"""CIFAR-10 / CIFAR-100 loaders and the [0, 1] -> normalized wrapper."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from .config import DataCfg
from .runtime import Runtime

DATASETS = {
    "cifar10": dict(cls=torchvision.datasets.CIFAR10, num_classes=10,
                    folder="cifar-10-batches-py",
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(cls=torchvision.datasets.CIFAR100, num_classes=100,
                     folder="cifar-100-python",
                     mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
}


class Normalized(nn.Module):
    """Wraps a backbone so it accepts [0, 1] images and normalizes internally.

    The noise pipeline works in [0, 1] end to end, so normalization has to live
    inside the model rather than in a dataset transform.
    """

    def __init__(self, backbone: nn.Module, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x):
        return self.backbone((x - self.mean.to(x.dtype)) / self.std.to(x.dtype))


class CifarData:
    """Train/test loaders plus a clean-view train loader for train accuracy.

    `train_eval_loader` applies the *test* transform to the training set, so the
    train accuracy it reports measures fit rather than augmentation strength.
    """

    def __init__(self, cfg: DataCfg, runtime: Runtime, train_eval_samples: int = 10000):
        if cfg.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {cfg.dataset!r}; "
                             f"expected one of {sorted(DATASETS)}")
        spec = DATASETS[cfg.dataset]
        self.num_classes = spec["num_classes"]
        self.mean, self.std = spec["mean"], spec["std"]

        root = os.path.abspath(os.path.expanduser(cfg.root))
        if not cfg.download and not os.path.isdir(os.path.join(root, spec["folder"])):
            raise FileNotFoundError(
                f"{root} does not contain {spec['folder']!r}. torchvision resolves the "
                f"data as <root>/{spec['folder']}/..., so data.root must be the PARENT "
                f"of that folder, not the folder itself. Set data.download=true to fetch it."
            )

        test_tf = T.Compose([T.ToTensor()])
        base = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        extra = [T.RandAugment(num_ops=2, magnitude=9)] if cfg.randaug else []
        post = [T.RandomErasing(p=0.25)] if cfg.randaug else []
        train_tf = T.Compose(base + extra + [T.ToTensor()] + post)

        cls = spec["cls"]
        self.train_set = cls(root, train=True, download=cfg.download, transform=train_tf)
        self.test_set = cls(root, train=False, download=cfg.download, transform=test_tf)
        clean_train = cls(root, train=True, download=False, transform=test_tf)

        kwargs = runtime.loader_kwargs()
        self.train_loader = DataLoader(self.train_set, batch_size=cfg.teacher_batch,
                                       shuffle=True, **kwargs)
        self.test_loader = DataLoader(self.test_set, batch_size=cfg.eval_batch,
                                      shuffle=False, **kwargs)

        if 0 < train_eval_samples < len(clean_train):
            generator = torch.Generator().manual_seed(0)
            picks = torch.randperm(len(clean_train), generator=generator)[:train_eval_samples]
            clean_train = Subset(clean_train, picks.tolist())
        self.train_eval_loader = DataLoader(clean_train, batch_size=cfg.eval_batch,
                                            shuffle=False, **kwargs)

    def wrap(self, backbone: nn.Module) -> Normalized:
        return Normalized(backbone, self.mean, self.std)

    def describe(self) -> str:
        return (f"{len(self.train_set)} train / {len(self.test_set)} test  "
                f"({self.num_classes} classes)")
