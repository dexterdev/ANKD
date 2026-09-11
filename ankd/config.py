"""Configuration: dataclass defaults, YAML files, dotted command-line overrides.

Defaults here are the shared protocol -- the one from the original ResNet-34 ->
ResNet-18 notebook. Every experiment config overrides only what it must, and each
override is commented in the YAML with the reason.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, List

import yaml


@dataclass
class RuntimeCfg:
    seed: int = 0
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    num_workers: int = -1           # -1 -> derive from cpu count
    prefetch_factor: int = -1       # -1 -> derive from num_workers
    amp: bool = True                # bf16 where supported, else fp16 + GradScaler
    channels_last: bool = True
    max_batch: int = 0              # 0 -> derive from VRAM; caps the student ramp


@dataclass
class DataCfg:
    dataset: str = "cifar100"       # "cifar10" | "cifar100"
    root: str = "./data"            # must CONTAIN cifar-10-batches-py / cifar-100-python
    download: bool = True
    teacher_batch: int = 64
    eval_batch: int = 256
    randaug: bool = False           # teacher-only augmentation; student sees noise
    randaug_num_ops: int = 2        # only used when randaug is true
    randaug_magnitude: int = 9      # lower this for short teacher runs


@dataclass
class ModelCfg:
    teacher: str = "resnet34"
    student: str = "resnet18"


@dataclass
class TeacherCfg:
    epochs: int = 100
    optimizer: str = "sgd"          # "sgd" | "adamw"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    clip_grad: float = 0.0          # 0 disables clipping
    ckpt: str = "checkpoints/teacher.pt"
    eval_train_every: int = 1       # 0 disables train-accuracy reporting
    eval_train_samples: int = 10000  # 0 -> the full training set


@dataclass
class NoiseCfg:
    num_samples: int = 40000
    per_call: int = 100
    families: List[str] = field(default_factory=lambda: [
        "smooth_gradient", "perlin", "uniform", "gabor", "checkerboard"])
    regenerate_every: int = 0       # 0 = one fixed set; N = redraw every N epochs


@dataclass
class AugmentCfg:
    use_geo: bool = True
    n_random_ops: int = 4


@dataclass
class StudentCfg:
    epochs: int = 200
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    temperature: float = 20.0
    warmup_epochs: int = 0          # helps AdamW on transformers; 0 = off
    schedule: str = "ramp_aligned"  # "ramp_aligned" (shared protocol) | "cosine"
    ckpt_best: str = "checkpoints/student_best.pt"
    ckpt_last: str = "checkpoints/student_last.pt"
    eval_train_every: int = 1


@dataclass
class Config:
    name: str = "experiment"
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    student: StudentCfg = field(default_factory=StudentCfg)

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _coerce(value: Any, target_type: Any) -> Any:
    """Cast a string from the command line to the field's declared type."""
    if not isinstance(value, str):
        return value
    if target_type is bool:
        low = value.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"cannot read {value!r} as a boolean")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return value
    # List[str] and anything else: fall back to YAML's own parser
    return yaml.safe_load(value)


def _apply(node: Any, mapping: dict, path: str = "") -> None:
    """Recursively overwrite dataclass fields from a nested dict."""
    known = {f.name: f for f in fields(node)}
    for key, value in mapping.items():
        where = f"{path}{key}"
        if key not in known:
            raise KeyError(f"unknown config key {where!r}. "
                           f"Valid here: {sorted(known)}")
        current = getattr(node, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{where!r} is a section and needs a mapping")
            _apply(current, value, f"{where}.")
        else:
            setattr(node, key, _coerce(value, known[key].type))


def _set_dotted(cfg: Config, dotted: str, value: str) -> None:
    parts = dotted.split(".")
    nested: dict = {}
    cursor = nested
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    _apply(cfg, nested)


def load_config(argv: List[str] | None = None) -> tuple[Config, argparse.Namespace]:
    """Parse --config, --stage and any number of dotted --section.key overrides.

        python -m ankd.train --config configs/vit8_vit4_cifar10.yaml \
            --student.epochs 50 --noise.num_samples 4000
    """
    parser = argparse.ArgumentParser(
        prog="ankd", description="Data-free distillation through augmented noise",
        epilog="Any config field can be overridden as --section.key value, "
               "e.g. --teacher.lr 0.001 --augment.n_random_ops 8")
    parser.add_argument("--config", help="path to a YAML config file")
    parser.add_argument("--stage", default="all", choices=["all", "teacher", "student"],
                        help="run both stages, or just one")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved config and exit")
    args, extra = parser.parse_known_args(argv)

    cfg = Config()
    if args.config:
        with open(args.config) as handle:
            loaded = yaml.safe_load(handle) or {}
        _apply(cfg, loaded)

    # dotted overrides: --a.b value  (or --a.b=value)
    index = 0
    while index < len(extra):
        token = extra[index]
        if not token.startswith("--"):
            raise SystemExit(f"unexpected argument {token!r}")
        if "=" in token:
            dotted, value = token[2:].split("=", 1)
            index += 1
        else:
            dotted = token[2:]
            if index + 1 >= len(extra):
                raise SystemExit(f"{token} needs a value")
            value = extra[index + 1]
            index += 2
        _set_dotted(cfg, dotted, value)

    return cfg, args
