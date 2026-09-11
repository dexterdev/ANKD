"""Device selection, worker/precision settings, and the fast-path helpers.

Nothing here changes training math -- it only picks settings from the hardware
that is actually present, and reports what it found.
"""

from __future__ import annotations

import os
import random
from contextlib import nullcontext

import numpy as np
import torch

from .config import RuntimeCfg


class Runtime:
    """Resolved hardware settings, shared by both training loops."""

    def __init__(self, cfg: RuntimeCfg):
        self.cfg = cfg
        seed_everything(cfg.seed)

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self.is_cuda = self.device.type == "cuda"

        cpu_count = os.cpu_count() or 2
        self.num_workers = (max(0, min(16, cpu_count - 1))
                            if cfg.num_workers < 0 else cfg.num_workers)
        if cfg.prefetch_factor >= 0:
            self.prefetch_factor = cfg.prefetch_factor
        else:
            self.prefetch_factor = (2 if self.num_workers <= 2
                                    else (4 if self.num_workers <= 8 else 6))
        torch.set_num_threads(max(1, min(cpu_count, 32)))

        self.bf16 = self.is_cuda and cfg.amp and torch.cuda.is_bf16_supported()
        self.channels_last = cfg.channels_last and self.is_cuda

        if self.is_cuda:
            torch.backends.cudnn.benchmark = True       # fixed 32x32 inputs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            props = torch.cuda.get_device_properties(0)
            self.gpu_name = props.name
            self.vram_gb = props.total_memory / 1e9
            self.capability = f"{props.major}.{props.minor}"
        else:
            self.gpu_name, self.vram_gb, self.capability = "none", 0.0, "-"

        # The student batch ramp climbs to 2048; cap it at what this card holds.
        if cfg.max_batch > 0:
            self.max_batch = cfg.max_batch
        elif not self.is_cuda:
            self.max_batch = 256
        else:
            self.max_batch = (2048 if self.vram_gb >= 16
                              else (1024 if self.vram_gb >= 10 else 512))

    # -- reporting ---------------------------------------------------------- #
    def describe(self) -> str:
        if self.is_cuda:
            head = (f"GPU: {self.gpu_name}  {self.vram_gb:.1f} GB  "
                    f"compute capability {self.capability}")
            precision = "bf16" if self.bf16 else (
                "fp16 + GradScaler" if self.cfg.amp else "fp32")
        else:
            head = ("GPU: none found -- running on CPU. Full-length training will "
                    "be impractically slow here.")
            precision = "fp32"
        return (f"{head}\n"
                f"device={self.device}  precision={precision}  "
                f"num_workers={self.num_workers}  prefetch={self.prefetch_factor}  "
                f"max_batch={self.max_batch}")

    # -- fast paths --------------------------------------------------------- #
    def prepare(self, model):
        """channels_last is a memory-layout change only; conv math is identical."""
        return model.to(memory_format=torch.channels_last) if self.channels_last else model

    def to_input(self, x):
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x

    def autocast(self):
        """bf16 needs no GradScaler (it keeps fp32's exponent range); fp16 does."""
        if self.is_cuda and self.cfg.amp:
            if self.bf16:
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16), False
            return torch.amp.autocast("cuda"), True
        return nullcontext(), False

    def scaler(self, enabled: bool):
        return torch.amp.GradScaler("cuda", enabled=enabled)

    def loader_kwargs(self) -> dict:
        kwargs = dict(num_workers=self.num_workers, pin_memory=self.is_cuda)
        if self.num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=self.prefetch_factor)
        return kwargs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
