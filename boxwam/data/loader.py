"""Data loaders and samplers for distributed training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from boxwam.data.episodes import EpisodeDataset


class BalancedEpisodeSampler(Sampler[int]):
    """Yield a fixed number of episode indices per rank and epoch.

    Episodes are repeated as evenly as possible so every epoch draws ``steps_per_epoch *
    batch_size`` windows per rank regardless of how many episodes exist. The order is a seeded
    permutation, so the sequence is reproducible and different for every epoch.
    """

    def __init__(self, num_episodes: int, samples_per_rank: int, *, rank: int, world_size: int, seed: int):
        self.num_episodes = num_episodes
        self.samples_per_rank = samples_per_rank
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_rank

    def __iter__(self):
        total = self.samples_per_rank * self.world_size
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(total, generator=generator) % self.num_episodes
        return iter(indices[self.rank :: self.world_size].tolist())


def build_train_loader(
    dataset: EpisodeDataset,
    *,
    batch_size: int,
    steps_per_epoch: int,
    rank: int,
    world_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, BalancedEpisodeSampler]:
    sampler = BalancedEpisodeSampler(
        len(dataset), steps_per_epoch * batch_size, rank=rank, world_size=world_size, seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )
    return loader, sampler


def build_val_loader(
    dataset: EpisodeDataset, *, batch_size: int, rank: int, world_size: int, seed: int, num_workers: int
) -> tuple[DataLoader, DistributedSampler]:
    """One random window per validation episode, split across ranks without shuffling."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=False)
    return loader, sampler
