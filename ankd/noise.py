"""Synthetic noise families -- the student's entire training distribution.

Images are built on CPU in [0, 1] so DataLoader workers can produce them.
"""

from __future__ import annotations

import math
import random

import torch

from .config import NoiseCfg


def _colourise(grad: torch.Tensor, norm: bool = True) -> torch.Tensor:
    """Map a scalar field to a 3-channel image interpolating two random colours."""
    grad = grad.float()
    if norm:
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    color_a, color_b = torch.rand(3, 1, 1), torch.rand(3, 1, 1)
    return grad.unsqueeze(0) * color_a + (1 - grad.unsqueeze(0)) * color_b


def _perlin_grid(size: int, res: int) -> torch.Tensor:
    """Single 2D Perlin field, roughly in [-1, 1]. `size` must divide by `res`."""
    assert size % res == 0, "size must be divisible by res"
    d = size // res
    lin = torch.arange(0, res, 1.0 / d)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack((gy % 1, gx % 1), dim=-1)

    angles = 2 * math.pi * torch.rand(res + 1, res + 1)
    grads = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    tile = lambda g: g.repeat_interleave(d, 0).repeat_interleave(d, 1)

    g00, g10 = tile(grads[:-1, :-1]), tile(grads[1:, :-1])
    g01, g11 = tile(grads[:-1, 1:]), tile(grads[1:, 1:])
    n00 = (torch.stack((grid[..., 0], grid[..., 1]), -1) * g00).sum(-1)
    n10 = (torch.stack((grid[..., 0] - 1, grid[..., 1]), -1) * g10).sum(-1)
    n01 = (torch.stack((grid[..., 0], grid[..., 1] - 1), -1) * g01).sum(-1)
    n11 = (torch.stack((grid[..., 0] - 1, grid[..., 1] - 1), -1) * g11).sum(-1)

    t = 6 * grid ** 5 - 15 * grid ** 4 + 10 * grid ** 3          # fade curve
    return torch.lerp(torch.lerp(n00, n10, t[..., 0]),
                      torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def smooth_gradient(n: int, size: int = 32) -> torch.Tensor:
    """Smooth linear gradient at a random angle between two random colours."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size),
                            indexing="ij")
    imgs = torch.zeros(n, 3, size, size)
    for i in range(n):
        angle = random.uniform(0, 2 * math.pi)
        imgs[i] = _colourise(xx * math.cos(angle) + yy * math.sin(angle))
    return imgs.clamp(0.0, 1.0)


def perlin(n: int, size: int = 32) -> torch.Tensor:
    """Perlin field per image, at a coarse/medium/fine cell grid."""
    imgs = torch.zeros(n, 3, size, size)
    for i in range(n):
        imgs[i] = _colourise(_perlin_grid(size, random.choice([2, 4, 8])))
    return imgs.clamp(0.0, 1.0)


def uniform(n: int, size: int = 32) -> torch.Tensor:
    """Plain i.i.d. uniform noise -- no spatial structure, unlike the others."""
    return torch.rand(n, 3, size, size).clamp(0.0, 1.0)


def gabor(n: int, size: int = 32) -> torch.Tensor:
    """Gaussian-windowed grating, random orientation / frequency / phase."""
    imgs = torch.zeros(n, 3, size, size)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size),
                            indexing="ij")
    for i in range(n):
        theta = random.uniform(0, math.pi)
        freq = random.uniform(2.0, 8.0)
        phase = random.uniform(0, 2 * math.pi)
        sigma = random.uniform(0.3, 0.8)
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        window = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2))
        imgs[i] = _colourise(window * torch.cos(2 * math.pi * freq * x_theta + phase))
    return imgs.clamp(0.0, 1.0)


def checkerboard(n: int, size: int = 32) -> torch.Tensor:
    """Checkerboard with random cell size and random phase offset per image."""
    imgs = torch.zeros(n, 3, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for i in range(n):
        block = random.choice([2, 4, 8, 16])
        ox, oy = random.randint(0, block - 1), random.randint(0, block - 1)
        imgs[i] = _colourise((((xx + ox) // block) + ((yy + oy) // block)) % 2, norm=False)
    return imgs.clamp(0.0, 1.0)


FAMILIES = {
    "smooth_gradient": smooth_gradient,
    "perlin": perlin,
    "uniform": uniform,
    "gabor": gabor,
    "checkerboard": checkerboard,
}


def build_noise(cfg: NoiseCfg, size: int = 32) -> torch.Tensor:
    """Exactly cfg.num_samples images, family drawn per batch of cfg.per_call."""
    unknown = [f for f in cfg.families if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown noise families {unknown}; "
                         f"expected from {sorted(FAMILIES)}")
    fns = [FAMILIES[f] for f in cfg.families]
    n_calls = math.ceil(cfg.num_samples / cfg.per_call)
    batches = [random.choice(fns)(cfg.per_call, size) for _ in range(n_calls)]
    return torch.cat(batches, dim=0)[:cfg.num_samples]
