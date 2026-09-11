"""The 17-op augmentation pool and the synthetic dataset it feeds.

Two independent switches, applied in order:
  1. geometric base  : RandomCrop(32, pad=4, reflect) + RandomHorizontalFlip
  2. random op stack : n_random_ops sampled without replacement from the pool

Both happen inside Dataset.__getitem__, i.e. before the tensor reaches the
teacher, so the teacher is queried on the augmented view.
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .config import AugmentCfg

base_geo_transform = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode="reflect"),
    T.RandomHorizontalFlip(),
])

_perspective = T.RandomPerspective(distortion_scale=0.35, p=1.0)
_zoom_crop = T.RandomResizedCrop(32, scale=(0.65, 1.0), ratio=(0.85, 1.15))
_color_jitter = T.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45, hue=0.12)
_cutout = T.RandomErasing(p=1.0, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0.0)


def op_rotate(img):
    return T.functional.rotate(img, random.uniform(-20, 20))


def op_affine(img):
    return T.functional.affine(
        img,
        angle=random.uniform(-12, 12),
        translate=(random.randint(-2, 2), random.randint(-2, 2)),
        scale=random.uniform(0.82, 1.18),
        shear=random.uniform(-12, 12),
    )


def op_gaussian_blur(img):
    return T.functional.gaussian_blur(img, kernel_size=random.choice([3, 5]),
                                      sigma=random.uniform(0.1, 2.2))


def op_equalize(img):
    return T.functional.equalize((img.clamp(0.0, 1.0) * 255).to(torch.uint8)).float() / 255.0


def op_posterize(img):
    img_u8 = (img.clamp(0.0, 1.0) * 255).to(torch.uint8)
    return T.functional.posterize(img_u8, random.choice([3, 4, 5, 6])).float() / 255.0


def op_gaussian_noise(img):
    return (img + torch.randn_like(img) * random.uniform(0.01, 0.08)).clamp(0.0, 1.0)


def op_salt_pepper(img):
    prob = random.uniform(0.01, 0.06)
    mask = torch.rand(1, img.shape[1], img.shape[2], device=img.device)
    out = img.clone()
    out[(mask < prob / 2).expand_as(img)] = 1.0
    out[(mask > 1 - prob / 2).expand_as(img)] = 0.0
    return out


def op_random_color_erase(img):
    out = img.clone()
    eh, ew = random.randint(4, 12), random.randint(4, 12)
    y0 = random.randint(0, img.shape[1] - eh)
    x0 = random.randint(0, img.shape[2] - ew)
    out[:, y0:y0 + eh, x0:x0 + ew] = torch.rand(3, 1, 1, device=img.device)
    return out


AUG_OPS = {
    "rotate": op_rotate,
    "affine": op_affine,
    "perspective": _perspective,
    "zoom_crop": _zoom_crop,
    "color_jitter": _color_jitter,
    "grayscale": lambda img: T.functional.rgb_to_grayscale(img, num_output_channels=3),
    "gaussian_blur": op_gaussian_blur,
    "sharpness": lambda img: T.functional.adjust_sharpness(img, random.uniform(0.0, 3.5)),
    "autocontrast": lambda img: T.functional.autocontrast(img.clamp(0.0, 1.0)),
    "equalize": op_equalize,
    "posterize": op_posterize,
    "solarize": lambda img: T.functional.solarize(img.clamp(0.0, 1.0), random.uniform(0.3, 0.9)),
    "invert": lambda img: T.functional.invert(img.clamp(0.0, 1.0)),
    "gaussian_noise": op_gaussian_noise,
    "salt_pepper": op_salt_pepper,
    "cutout": lambda img: _cutout(img.unsqueeze(0)).squeeze(0),
    "random_color_erase": op_random_color_erase,
}
AUG_OP_NAMES = list(AUG_OPS)
assert len(AUG_OP_NAMES) == 17, f"expected a 17-op pool, found {len(AUG_OP_NAMES)}"


def diverse_augment(img, use_geo: bool = True, n_random_ops: int = 4):
    img = img.clamp(0.0, 1.0)
    if use_geo:
        img = base_geo_transform(img)
    for name in random.sample(AUG_OP_NAMES, k=min(int(n_random_ops), len(AUG_OP_NAMES))):
        img = AUG_OPS[name](img).clamp(0.0, 1.0)
    return img


class SyntheticDataset(Dataset):
    """Yields the augmented noise image; the teacher labels it on the fly."""

    def __init__(self, imgs: torch.Tensor, cfg: AugmentCfg):
        self.imgs = imgs
        self.use_geo = cfg.use_geo
        self.n_random_ops = cfg.n_random_ops

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return diverse_augment(self.imgs[idx], self.use_geo, self.n_random_ops)
