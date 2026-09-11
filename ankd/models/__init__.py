"""Model registry. `build_model(name, num_classes)` returns a bare backbone."""

from .alexnet import alexnet, alexnet_half
from .resnet import resnet18, resnet34
from .vit import vit4, vit8

REGISTRY = {
    "resnet34": resnet34,
    "resnet18": resnet18,
    "alexnet": alexnet,
    "alexnet_half": alexnet_half,
    "vit8": vit8,
    "vit4": vit4,
}


def build_model(name: str, num_classes: int):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(REGISTRY)}")
    return REGISTRY[name](num_classes=num_classes)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["REGISTRY", "build_model", "count_params",
           "resnet18", "resnet34", "alexnet", "alexnet_half", "vit8", "vit4"]
