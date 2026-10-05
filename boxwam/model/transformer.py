"""Transformer building blocks for the Box-WAM predictor.

Two pieces live here:

* :class:`RotaryAttention`: multi-head self-attention with a 2-D rotary position
  embedding (RoPE) over (row, column) token coordinates, plus an optional key/value
  cache so observation tokens can be encoded once per action chunk.
* :class:`AdaLNBlock`: a pre-norm transformer block whose LayerNorms are modulated by a
  per-token conditioning vector (adaptive LayerNorm, as in DiT).

The rotary layout is inherited from V-JEPA 2's 3-D RoPE, which splits each attention head
into three equal segments rotated by (frame, row, column) and leaves any remainder
unrotated. Box-WAM only sees a single frame, so the frame segment is rotated by a constant
zero angle, i.e. it is left untouched. Segment sizes and the sin/cos pairing are kept
exactly as in V-JEPA 2 so that released checkpoints load unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


def rotate(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of ``x`` by angles derived from ``positions``.

    ``x`` has shape ``(B, heads, T, D)`` with even ``D``; ``positions`` has shape ``(T,)``.
    The sin/cos tables are tiled as ``[s_0 .. s_{D/2-1}, s_0 .. s_{D/2-1}]`` while the rotation
    partner of ``x[..., 2i]`` is ``x[..., 2i+1]``. This pairing comes from the V-JEPA 2
    checkpoints Box-WAM was initialised from and must not be changed.

    Frequencies are computed in the dtype of ``x`` and multiplied with float32 positions,
    matching the numerics the released checkpoints were trained with under bfloat16 autocast.
    """
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE dimension must be even")
    exponent = torch.arange(dim // 2, device=x.device, dtype=x.dtype) / (dim / 2.0)
    freqs = 1.0 / 10000**exponent
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * freqs[None, :]
    sin = angles.sin().repeat(1, 2)
    cos = angles.cos().repeat(1, 2)
    first, second = x.unflatten(-1, (-1, 2)).unbind(-1)
    partner = torch.stack((-second, first), dim=-1).flatten(-2)
    return x * cos + partner * sin


class RotaryAttention(nn.Module):
    """Multi-head self-attention with 2-D RoPE and an optional key/value cache."""

    def __init__(self, dim: int, num_heads: int, *, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Width of each of the three V-JEPA 2 rotary segments (frame, row, column).
        self.rope_dim = 2 * ((self.head_dim // 3) // 2)
        if self.rope_dim == 0:
            raise ValueError("head dimension is too small for rotary embeddings")
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def _apply_rope(self, x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        r = self.rope_dim
        frame = x[..., :r]  # constant frame index 0: rotation by zero is the identity
        return torch.cat(
            [
                frame.to(torch.float32),
                rotate(x[..., r : 2 * r], rows),
                rotate(x[..., 2 * r : 3 * r], cols),
                x[..., 3 * r :],
            ],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ):
        """Attend over ``x`` (plus cached keys/values, if given).

        ``rows``/``cols`` are the rotary coordinates of the tokens in ``x``. ``cache`` holds
        already-rotated keys and values of earlier tokens, which are prepended along the
        sequence axis. With ``return_kv`` the rotated keys/values of ``x`` are returned so the
        caller can build such a cache.
        """
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).unflatten(-1, (3, self.num_heads, self.head_dim)).permute(2, 0, 3, 1, 4)
        q = self._apply_rope(q, rows, cols)
        k = self._apply_rope(k, rows, cols)
        new_k, new_v = k, v
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = self.proj(out.transpose(1, 2).reshape(batch, length, dim))
        if return_kv:
            return out, new_k, new_v
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AdaLNBlock(nn.Module):
    """Pre-norm transformer block with adaptive LayerNorm conditioning.

    ``cond`` is a per-token vector ``(B, T, dim)``; it produces shift/scale/gate triples for
    the attention and feed-forward branches (six chunks in total).
    """

    def __init__(
        self, dim: int, num_heads: int, *, mlp_ratio: float = 4.0, drop_path: float = 0.0, norm_eps: float = 1e-6
    ):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = RotaryAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(cond).chunk(6, dim=-1)
        attn = self.attn(
            self.norm1(x) * (1 + scale_a) + shift_a, rows, cols, attn_mask=attn_mask, cache=cache, return_kv=return_kv
        )
        if return_kv:
            attn, k, v = attn
        x = x + self.drop_path(gate_a * attn)
        x = x + self.drop_path(gate_m * self.mlp(self.norm2(x) * (1 + scale_m) + shift_m))
        if return_kv:
            return x, k, v
        return x
