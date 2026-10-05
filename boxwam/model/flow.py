"""Rectified-flow utilities shared by training and inference.

Convention: ``t = 0`` is pure Gaussian noise, ``t = 1`` is data. A sample along the
straight path is ``x_t = (1 - t) * noise + t * target`` and the velocity to regress is
``target - noise``. Inference integrates ``dx/dt = v(x_t, t)`` from 0 to 1 with Euler steps.
"""

from __future__ import annotations

import torch

from boxwam.model.boxwam import BoxWAM


def sample_timesteps(
    num: int,
    *,
    device: torch.device,
    scale: float = 0.999,
    beta: tuple[float, float] = (1.5, 1.0),
) -> torch.Tensor:
    """Draw ``num`` training timesteps ``t = scale * (1 - u)`` with ``u ~ Beta(a, b)``.

    With the default ``Beta(1.5, 1)`` this puts more mass on small ``t`` (noisier points), as
    in pi0.5. Set ``beta=(1, 1)`` for a uniform distribution.
    """
    if num <= 0:
        raise ValueError("num must be positive")
    alpha, b = (torch.as_tensor(value, device=device, dtype=torch.float32) for value in beta)
    u = torch.distributions.Beta(alpha, b).sample((num,))
    return (scale * (1.0 - u)).clamp(0.0, scale)


def interpolate(target: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the flow sample ``x_t`` and the velocity target for a batch of timesteps ``t``."""
    t = t.view(-1, *([1] * (target.ndim - 1)))
    return (1.0 - t) * noise + t * target, target - noise


def euler_schedule(num_steps: int) -> list[tuple[float, float]]:
    """Quadratic Euler schedule ``t_k = 1 - (1 - k / N)^2`` as ``(t_k, t_{k+1} - t_k)`` pairs.

    The step size shrinks towards ``t = 1`` where the trajectory needs the most resolution.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    times = [1.0 - (1.0 - k / num_steps) ** 2 for k in range(num_steps + 1)]
    return [(times[k], times[k + 1] - times[k]) for k in range(num_steps)]


@torch.no_grad()
def integrate(
    model: BoxWAM,
    cache: list[tuple[torch.Tensor, torch.Tensor]],
    state: torch.Tensor,
    actions: torch.Tensor,
    boxes: torch.Tensor | None,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Integrate the flow from noise to data with cached observation tokens.

    ``actions`` and ``boxes`` hold the initial Gaussian noise and are updated in place in
    float32. The denoised (normalised) action chunk and box trajectory are returned.
    """
    batch = state.shape[0]
    for time, dt in euler_schedule(num_steps):
        t = torch.full((batch,), time, device=state.device, dtype=torch.float32)
        velocity = model.predict_velocity(cache, state, actions, boxes, t)
        actions += dt * velocity.action.float()
        if boxes is not None:
            boxes += dt * velocity.box.float()
    return actions, boxes
