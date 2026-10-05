"""Box-WAM training loop.

One process per GPU; the frozen DINOv3 encoder runs in float32 and the predictor under
autocast. Each epoch draws ``steps_per_epoch`` batches, then runs validation rollouts on the
held-out episodes and writes ``latest.pt`` (and ``best.pt`` when the rollout action error
improves) to ``<output_dir>/<run_name>/``.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from boxwam.config import ModelConfig, TrainConfig, to_dict
from boxwam.data.episodes import EpisodeDataset, find_episodes, split_episodes
from boxwam.data.loader import build_train_loader, build_val_loader
from boxwam.data.normalization import ActionStats
from boxwam.model.boxwam import BoxWAM
from boxwam.model.encoder import DinoPatchEncoder
from boxwam.training.checkpoint import load_checkpoint, load_model_weights, save_checkpoint, unwrap
from boxwam.training.loss import flow_matching_loss, prepare_batch, rollout_metrics
from boxwam.training.optim import Schedules, build_optimizer
from boxwam.utils.distributed import all_reduce_sum
from boxwam.utils.logging import Average, CSVLogger, append_jsonl, get_logger

AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_model(cfg: ModelConfig, *, encoder_dim: int, num_cameras: int, image_size: int, patch_size: int) -> BoxWAM:
    return BoxWAM(
        encoder_dim=encoder_dim,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        image_size=image_size,
        patch_size=patch_size,
        num_cameras=num_cameras,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.horizon,
        box_dim=cfg.box_dim,
        action_attend_box=cfg.action_attend_box,
        box_attend_action=cfg.box_attend_action,
        drop_path=cfg.drop_path,
        activation_checkpointing=cfg.activation_checkpointing,
    )


def run_training(cfg: TrainConfig, *, rank: int, world_size: int, device: torch.device) -> None:
    logger = get_logger(rank=rank)
    torch.manual_seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True
    amp_dtype = AMP_DTYPES[cfg.dtype]
    amp = amp_dtype != torch.float32 and device.type == "cuda"
    run_dir = cfg.run_dir
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as handle:
            import yaml

            yaml.safe_dump(to_dict(cfg), handle, sort_keys=False)

    # ---------------------------------------------------------------- data
    stats = ActionStats.load(cfg.data.stats, cfg.data.gripper_dims)
    if stats.action_dim != cfg.model.action_dim or stats.state_dim != cfg.model.state_dim:
        raise ValueError(
            f"statistics are for action_dim={stats.action_dim}/state_dim={stats.state_dim} but the model has "
            f"action_dim={cfg.model.action_dim}/state_dim={cfg.model.state_dim}"
        )
    episodes = find_episodes(cfg.data.episodes, cfg.data.task)
    train_episodes, val_episodes = split_episodes(episodes, cfg.data.val_fraction, cfg.seed)
    logger.info("Episodes: %d train, %d validation", len(train_episodes), len(val_episodes))
    dataset_kwargs = dict(
        cameras=cfg.data.cameras,
        horizon=cfg.model.horizon,
        load_boxes=cfg.model.box_dim > 0,
        use_valid_starts=cfg.data.use_valid_starts,
        preload=cfg.data.preload,
    )
    train_set = EpisodeDataset(train_episodes, **dataset_kwargs)
    train_loader, train_sampler = build_train_loader(
        train_set,
        batch_size=cfg.data.batch_size,
        steps_per_epoch=cfg.optim.steps_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
        num_workers=cfg.data.num_workers,
    )
    val_loader = None
    if val_episodes:
        val_set = EpisodeDataset(val_episodes, **dataset_kwargs)
        val_loader, _ = build_val_loader(
            val_set,
            batch_size=cfg.data.batch_size,
            rank=rank,
            world_size=world_size,
            seed=cfg.seed,
            num_workers=cfg.data.num_workers,
        )

    # --------------------------------------------------------------- model
    encoder = DinoPatchEncoder(cfg.model.encoder).to(device)
    model = build_model(
        cfg.model,
        encoder_dim=encoder.embed_dim,
        num_cameras=len(cfg.data.cameras),
        image_size=cfg.augmentation.crop_size,
        patch_size=encoder.patch_size,
    ).to(device)
    logger.info("Predictor parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
    optimizer = build_optimizer(model, cfg.optim)
    schedules = Schedules(cfg.optim)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype == torch.float16)

    start_epoch = 0
    best_action_se = float("inf")
    if cfg.init_from:
        load_model_weights(model, load_checkpoint(cfg.init_from))
        logger.info("Initialised weights from %s", cfg.init_from)
    if cfg.resume:
        checkpoint = load_checkpoint(cfg.resume)
        load_model_weights(model, checkpoint)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_action_se = float(checkpoint.get("best_action_se", best_action_se))
        logger.info("Resumed from %s at epoch %d", cfg.resume, start_epoch)

    if cfg.model.compile and device.type == "cuda":
        torch._dynamo.config.optimize_ddp = False
        model.compile()
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])

    csv_log = (
        CSVLogger(run_dir / "train_log.csv", ["epoch", "step", "loss", "action_loss", "box_loss", "lr", "step_ms"])
        if rank == 0
        else None
    )
    box_dim = cfg.model.box_dim

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, cfg.optim.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        meters = {name: Average() for name in ("loss", "action_loss", "box_loss")}
        progress = tqdm(
            train_loader, desc=f"epoch {epoch + 1}/{cfg.optim.epochs}", disable=rank != 0, dynamic_ncols=True
        )
        for step, raw in enumerate(progress):
            tick = time.perf_counter()
            global_step = epoch * cfg.optim.steps_per_epoch + step + 1
            lr, _ = schedules.apply(optimizer, global_step)
            batch = prepare_batch(
                raw, encoder=encoder, stats=stats, augmentation=cfg.augmentation, box_dim=box_dim, device=device
            )
            loss, parts = flow_matching_loss(model, batch, cfg.loss, amp_dtype=amp_dtype, amp=amp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if not torch.isfinite(loss):
                raise RuntimeError("loss is not finite")
            meters["loss"].update(loss.item())
            meters["action_loss"].update(parts["action_loss"])
            meters["box_loss"].update(parts["box_loss"])
            if rank == 0:
                progress.set_postfix(loss=f"{meters['loss'].mean:.4f}", lr=f"{lr:.1e}")
                csv_log.log(
                    epoch=epoch + 1,
                    step=global_step,
                    loss=loss.item(),
                    action_loss=parts["action_loss"],
                    box_loss=parts["box_loss"],
                    lr=lr,
                    step_ms=(time.perf_counter() - tick) * 1000,
                )

        record = {"epoch": epoch + 1, **{f"train/{name}": meter.mean for name, meter in meters.items()}}
        if val_loader is not None:
            record.update(
                validate(
                    model,
                    val_loader,
                    encoder=encoder,
                    stats=stats,
                    cfg=cfg,
                    device=device,
                    amp_dtype=amp_dtype,
                    amp=amp,
                )
            )
        if rank == 0:
            logger.info(" ".join(f"{key}={value:.4g}" for key, value in record.items()))
            append_jsonl(run_dir / "metrics.jsonl", record)
            common = dict(model=model, config=cfg, stats=stats, epoch=epoch + 1, metrics=record)
            save_checkpoint(run_dir / "latest.pt", optimizer=optimizer, **common)
            if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
                save_checkpoint(run_dir / f"epoch_{epoch + 1:03d}.pt", **common)
            action_se = record.get("val/action_mse", float("inf"))
            if action_se < best_action_se:
                best_action_se = action_se
                save_checkpoint(run_dir / "best.pt", **common)
                logger.info("New best validation action MSE %.4g -> best.pt", action_se)
        if dist.is_initialized():
            dist.barrier()


@torch.no_grad()
def validate(model, loader, *, encoder, stats, cfg: TrainConfig, device, amp_dtype, amp) -> dict[str, float]:
    """Flow losses and rollout errors on the validation episodes, averaged across all ranks."""
    model.eval()
    base = unwrap(model)
    sums = torch.zeros(5, device=device, dtype=torch.float64)  # loss, action_loss, box_loss, action_se, box_se
    count = torch.zeros(1, device=device, dtype=torch.float64)
    validation_aug = cfg.augmentation.geometry_only()
    for raw in loader:
        batch = prepare_batch(
            raw, encoder=encoder, stats=stats, augmentation=validation_aug, box_dim=cfg.model.box_dim, device=device
        )
        loss, parts = flow_matching_loss(base, batch, cfg.loss, amp_dtype=amp_dtype, amp=amp)
        rollout = rollout_metrics(base, batch, stats, num_steps=cfg.loss.val_steps, amp_dtype=amp_dtype, amp=amp)
        sums += torch.tensor(
            [
                loss.item() * batch.size,
                parts["action_loss"] * batch.size,
                parts["box_loss"] * batch.size,
                rollout["action_se"],
                rollout.get("box_se", 0.0),
            ],
            device=device,
            dtype=torch.float64,
        )
        count += batch.size
    all_reduce_sum(sums)
    all_reduce_sum(count)
    model.train()
    n = max(count.item(), 1.0)
    names = ["val/loss", "val/action_loss", "val/box_loss", "val/action_mse", "val/box_mse"]
    return {name: value / n for name, value in zip(names, sums.tolist())}
