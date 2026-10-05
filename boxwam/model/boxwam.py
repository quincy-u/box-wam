"""The Box-WAM predictor: a rectified-flow transformer over observation, action and box tokens.

Token sequence
--------------
::

    [ observation patch tokens | H action tokens | H box tokens ]
      num_cameras x grid x grid   one per step      one per step (only when box_dim > 0)

* Observation tokens are frozen DINOv3 patch features of the current frame(s), projected to
  the predictor width. They carry the robot state through adaptive LayerNorm.
* Action / box tokens are the noisy rectified-flow samples ``x_t`` projected to the predictor
  width. They additionally receive the flow timestep through adaptive LayerNorm.

Attention
---------
Observation tokens attend only to observation tokens, so they can be encoded once and cached
while the flow ODE is integrated. Action tokens always see observation and action tokens; box
tokens always see observation and box tokens. The two cross-links are configurable:

* ``action_attend_box`` lets action tokens read the (imagined) future box trajectory;
* ``box_attend_action`` lets box tokens read the action chunk.

The paper's Box-WAM design uses ``action_attend_box=True, box_attend_action=False``: the box
trajectory acts as an object-motion plan that guides the actions without depending on them.

Outputs
-------
Per-token velocities of the rectified flow: ``(B, H, action_dim)`` and ``(B, H, box_dim)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from boxwam.model.transformer import AdaLNBlock


class Velocity(NamedTuple):
    action: torch.Tensor
    box: torch.Tensor | None


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of flow timesteps ``t`` in ``[0, 1]`` (cosines first, then sines)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class BoxWAM(nn.Module):
    def __init__(
        self,
        *,
        encoder_dim: int = 1024,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        image_size: int = 224,
        patch_size: int = 16,
        num_cameras: int = 1,
        state_dim: int,
        action_dim: int,
        horizon: int,
        box_dim: int = 16,
        action_attend_box: bool = True,
        box_attend_action: bool = True,
        drop_path: float = 0.0,
        time_embed_dim: int = 128,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be a multiple of patch_size")
        if horizon < 1 or num_cameras < 1:
            raise ValueError("horizon and num_cameras must be positive")
        self.grid = image_size // patch_size
        self.num_cameras = num_cameras
        self.num_patches = num_cameras * self.grid * self.grid
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.box_dim = box_dim
        self.action_attend_box = action_attend_box
        self.box_attend_action = box_attend_action
        self.time_embed_dim = time_embed_dim
        self.activation_checkpointing = activation_checkpointing

        # Input projections. Attribute names match the released checkpoints.
        self.predictor_embed = nn.Linear(encoder_dim, embed_dim)
        self.action_flow_in_proj = nn.Linear(action_dim, embed_dim)
        self.obb_flow_in_proj = nn.Linear(box_dim, embed_dim) if box_dim else None
        self.state_adaln_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        drop_paths = torch.linspace(0, drop_path, depth).tolist()
        self.predictor_blocks = nn.ModuleList([AdaLNBlock(embed_dim, num_heads, drop_path=rate) for rate in drop_paths])

        # Output: project back to the encoder width, then decode velocities linearly.
        self.predictor_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(embed_dim, encoder_dim)
        self.action_decoder = nn.Linear(encoder_dim, action_dim)
        self.obb_decoder = nn.Linear(encoder_dim, box_dim) if box_dim else None

        # Rotary coordinates. Patch tokens use their (row, column) in a grid stacked over
        # cameras; action and box tokens sit on row 1 to the right of the image grid.
        patch_ids = torch.arange(self.num_patches)
        rows = torch.cat([patch_ids // self.grid, torch.ones(2 * horizon, dtype=torch.long)])
        cols = torch.cat([patch_ids % self.grid, self.grid + torch.arange(2 * horizon)])
        self.register_buffer("rows", rows.float(), persistent=False)
        self.register_buffer("cols", cols.float(), persistent=False)

        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(init)
        nn.init.zeros_(self.action_decoder.weight)
        nn.init.zeros_(self.action_decoder.bias)
        if self.obb_decoder is not None:
            nn.init.zeros_(self.obb_decoder.weight)
            nn.init.zeros_(self.obb_decoder.bias)
        # Depth-dependent rescaling of the residual branches (as in V-JEPA 2).
        for index, block in enumerate(self.predictor_blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))

    # --------------------------------------------------------------- helpers
    @property
    def has_boxes(self) -> bool:
        return self.box_dim > 0

    def _num_tokens(self, with_boxes: bool) -> int:
        return self.num_patches + self.horizon * (2 if with_boxes else 1)

    def _embed_flow_tokens(self, actions: torch.Tensor, boxes: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
        batch = actions.shape[0]
        if actions.shape != (batch, self.horizon, self.action_dim):
            raise ValueError(f"actions must be (B, {self.horizon}, {self.action_dim}), got {tuple(actions.shape)}")
        tokens = [self.action_flow_in_proj(actions.to(dtype))]
        if boxes is not None:
            if not self.has_boxes:
                raise ValueError("boxes were given but the model has box_dim=0")
            if boxes.shape != (batch, self.horizon, self.box_dim):
                raise ValueError(f"boxes must be (B, {self.horizon}, {self.box_dim}), got {tuple(boxes.shape)}")
            tokens.append(self.obb_flow_in_proj(boxes.to(dtype)))
        return torch.cat(tokens, dim=1)

    def _time_features(self, t: torch.Tensor) -> torch.Tensor:
        weight_dtype = self.time_mlp[0].weight.dtype
        return self.time_mlp(timestep_embedding(t.reshape(-1), self.time_embed_dim).to(weight_dtype))

    def _attention_mask(self, with_boxes: bool, device: torch.device) -> torch.Tensor:
        """Boolean mask ``(1, 1, T, T)`` for the full ``[observation | action | box]`` sequence."""
        ids = torch.arange(self._num_tokens(with_boxes), device=device)
        obs = ids < self.num_patches
        act = (ids >= self.num_patches) & (ids < self.num_patches + self.horizon)
        box = ids >= self.num_patches + self.horizon
        allowed = obs[:, None] & obs[None, :]
        allowed |= act[:, None] & (obs | act)[None, :]
        allowed |= box[:, None] & (obs | box)[None, :]
        if self.action_attend_box:
            allowed |= act[:, None] & box[None, :]
        if self.box_attend_action:
            allowed |= box[:, None] & act[None, :]
        return allowed[None, None]

    def _cached_attention_mask(self, with_boxes: bool, device: torch.device) -> torch.Tensor | None:
        """Mask for new ``[action | box]`` queries over ``[observation | action | box]`` keys.

        Returns ``None`` (dense attention) whenever every cross-link is allowed, which lets
        scaled_dot_product_attention pick its fastest kernel.
        """
        if not with_boxes or (self.action_attend_box and self.box_attend_action):
            return None
        new = torch.arange(2 * self.horizon, device=device)
        keys = torch.arange(self.num_patches + 2 * self.horizon, device=device)
        act_row = new < self.horizon
        box_row = ~act_row
        obs_col = keys < self.num_patches
        act_col = (keys >= self.num_patches) & (keys < self.num_patches + self.horizon)
        box_col = keys >= self.num_patches + self.horizon
        allowed = act_row[:, None] & (obs_col | act_col)[None, :]
        allowed |= box_row[:, None] & (obs_col | box_col)[None, :]
        if self.action_attend_box:
            allowed |= act_row[:, None] & box_col[None, :]
        if self.box_attend_action:
            allowed |= box_row[:, None] & act_col[None, :]
        return allowed[None, None]

    def _decode(self, x: torch.Tensor, first_action_token: int) -> Velocity:
        out = self.predictor_proj(self.predictor_norm(x))
        action_tokens = out[:, first_action_token : first_action_token + self.horizon]
        box_tokens = out[:, first_action_token + self.horizon : first_action_token + 2 * self.horizon]
        box = self.obb_decoder(box_tokens) if box_tokens.shape[1] else None
        return Velocity(self.action_decoder(action_tokens), box)

    def _check_inputs(self, patches: torch.Tensor, state: torch.Tensor) -> None:
        if patches.ndim != 3 or patches.shape[1] != self.num_patches:
            raise ValueError(f"patches must be (B, {self.num_patches}, encoder_dim), got {tuple(patches.shape)}")
        if state.shape != (patches.shape[0], self.state_dim):
            raise ValueError(f"state must be (B, {self.state_dim}), got {tuple(state.shape)}")

    # --------------------------------------------------------------- forward
    def forward(
        self,
        patches: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        boxes: torch.Tensor | None,
        t: torch.Tensor,
    ) -> Velocity:
        """Predict flow velocities in one pass over the full token sequence (training path).

        ``patches``: ``(B, num_patches, encoder_dim)`` DINOv3 features of the current frame(s).
        ``state``: ``(B, state_dim)`` normalised robot state.
        ``actions`` / ``boxes``: noisy flow samples ``(B, H, action_dim)`` / ``(B, H, box_dim)``.
        ``t``: ``(B,)`` flow timesteps in ``[0, 1]``.
        """
        self._check_inputs(patches, state)
        with_boxes = boxes is not None
        obs_tokens = self.predictor_embed(patches)
        x = torch.cat([obs_tokens, self._embed_flow_tokens(actions, boxes, obs_tokens.dtype)], dim=1)
        num_tokens = x.shape[1]

        cond = self.state_adaln_encoder(state)[:, None, :].expand(-1, num_tokens, -1).contiguous()
        cond[:, self.num_patches :] += self._time_features(t)[:, None, :]
        rows, cols = self.rows[:num_tokens], self.cols[:num_tokens]
        mask = self._attention_mask(with_boxes, x.device)

        for block in self.predictor_blocks:
            if self.activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cond, rows, cols, attn_mask=mask, use_reentrant=False)
            else:
                x = block(x, cond, rows, cols, attn_mask=mask)
        return self._decode(x, first_action_token=self.num_patches)

    @torch.no_grad()
    def encode_observation(self, patches: torch.Tensor, state: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run the observation tokens through every block and return their keys/values.

        Observation tokens never attend to action or box tokens and do not depend on the flow
        timestep, so this cache is valid for the whole ODE integration of one action chunk.
        """
        self._check_inputs(patches, state)
        x = self.predictor_embed(patches)
        cond = self.state_adaln_encoder(state)[:, None, :].expand(-1, self.num_patches, -1).contiguous()
        rows, cols = self.rows[: self.num_patches], self.cols[: self.num_patches]
        cache = []
        for block in self.predictor_blocks:
            x, k, v = block(x, cond, rows, cols, return_kv=True)
            cache.append((k, v))
        return cache

    def predict_velocity(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        state: torch.Tensor,
        actions: torch.Tensor,
        boxes: torch.Tensor | None,
        t: torch.Tensor,
    ) -> Velocity:
        """Predict velocities for noisy action/box tokens given cached observation keys/values."""
        if len(cache) != len(self.predictor_blocks):
            raise ValueError("cache depth does not match the number of blocks")
        with_boxes = boxes is not None
        x = self._embed_flow_tokens(actions, boxes, self.action_flow_in_proj.weight.dtype)
        num_new = x.shape[1]
        cond = (
            (self.state_adaln_encoder(state) + self._time_features(t))[:, None, :].expand(-1, num_new, -1).contiguous()
        )
        rows = self.rows[self.num_patches : self.num_patches + num_new]
        cols = self.cols[self.num_patches : self.num_patches + num_new]
        mask = self._cached_attention_mask(with_boxes, x.device)
        for block, kv in zip(self.predictor_blocks, cache):
            x = block(x, cond, rows, cols, attn_mask=mask, cache=kv)
        return self._decode(x, first_action_token=0)


WRAPPER_PREFIXES = ("module.", "_orig_mod.")  # added by DistributedDataParallel / torch.compile


def load_predictor_state(model: BoxWAM, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a state dict into ``model``, ignoring wrapper prefixes; every key must match."""
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
