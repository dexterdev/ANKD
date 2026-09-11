"""End-to-end smoke test: every config, both stages, tiny.

Substitutes a stub CIFAR so the whole pipeline can be exercised without the real
dataset. It checks that the code runs and that checkpoints round-trip -- it says
nothing about accuracy.

    python tests/smoke_test.py
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ankd import checkpoint, engine                                     # noqa: E402
from ankd.augment import AUG_OP_NAMES, SyntheticDataset                 # noqa: E402
from ankd.config import AugmentCfg, NoiseCfg, load_config               # noqa: E402
from ankd.models import build_model, count_params                       # noqa: E402
from ankd.noise import FAMILIES, build_noise                            # noqa: E402
from ankd.train import main                                             # noqa: E402

CONFIGS = ["configs/resnet34_resnet18_cifar100.yaml",
           "configs/resnet34_resnet18_cifar10.yaml",
           "configs/alexnet_alexnethalf_cifar10.yaml",
           "configs/vit8_vit4_cifar10.yaml"]

EXPECTED_PARAMS = {("resnet34", 100): 21_328_292, ("resnet18", 100): 11_220_132,
                   ("resnet34", 10): 21_282_122, ("resnet18", 10): 11_173_962,
                   ("alexnet", 10): 1_659_178, ("alexnet_half", 10): 417_434,
                   ("vit8", 10): 4_249_354, ("vit4", 10): 2_140_938}


def _stub_cifar(num_classes):
    class Stub(Dataset):
        def __init__(self, root, train=True, download=False, transform=None):
            self.n = 256 if train else 128
            self.transform = transform
            rng = np.random.RandomState(0 if train else 1)
            self.data = rng.randint(0, 256, (self.n, 32, 32, 3), dtype=np.uint8)
            self.targets = rng.randint(0, num_classes, self.n)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.transform(Image.fromarray(self.data[i])), int(self.targets[i])
    return Stub


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}")
    assert condition, label


def test_units():
    print("units:")
    check("17-op augmentation pool", len(AUG_OP_NAMES) == 17)
    check("5 noise families", len(FAMILIES) == 5)

    for (name, ncls), expected in EXPECTED_PARAMS.items():
        model = build_model(name, ncls).eval()
        out = model(torch.randn(2, 3, 32, 32))
        check(f"{name} builds, {expected:,} params, out {tuple(out.shape)}",
              count_params(model) == expected and out.shape == (2, ncls))

    check("vit4 is exactly half of vit8",
          len(build_model("vit4", 10).blocks) * 2 == len(build_model("vit8", 10).blocks))

    # the shared LR schedule must match the original notebook exactly
    def original(e):
        start = 0.001 if 25 <= e < 50 else 0.01
        ps = min(e // 25, 6) * 25
        pl = 50 if e >= 150 else 25
        cos = 0.5 * (1 + math.cos(math.pi * (e - ps) / max(pl - 1, 1)))
        return (1e-5 + (start - 1e-5) * cos) / 0.01
    check("batch_aligned_lr matches the reference protocol",
          all(abs(engine.batch_aligned_lr(e) - original(e)) < 1e-12 for e in range(260)))
    uncapped = [engine.batch_size_for(e, 4096) for e in (0, 25, 50, 75, 100, 125, 150)]
    check("batch ramp is 16 -> 2048", uncapped == [16, 64, 128, 256, 512, 1024, 2048])
    check("batch ramp caps at max_batch",
          [engine.batch_size_for(e, 512) for e in (0, 100, 200)] == [16, 512, 512])

    noise = build_noise(NoiseCfg(num_samples=150, per_call=100))
    check("build_noise returns exactly num_samples", noise.shape == (150, 3, 32, 32))
    check("noise stays in [0, 1]", 0.0 <= noise.min() and noise.max() <= 1.0)

    ds = SyntheticDataset(noise, AugmentCfg(n_random_ops=4))
    check("augmented sample shape", ds[0].shape == (3, 32, 32))


def test_checkpoint_roundtrip(tmp):
    print("checkpoint round-trip:")
    from ankd.data import Normalized
    model = Normalized(build_model("alexnet_half", 10), (0.5,) * 3, (0.25,) * 3)
    path = os.path.join(tmp, "ck.pt")
    checkpoint.save(model, path, val_acc=42.0, epoch=7)

    fresh = Normalized(build_model("alexnet_half", 10), (0.5,) * 3, (0.25,) * 3)
    meta = checkpoint.load(fresh, path, torch.device("cpu"))
    same = all(torch.equal(v, model.state_dict()[k]) for k, v in fresh.state_dict().items())
    check(f"wrapped save/load is bit-identical (meta={meta.get('val_acc')})", same)

    # a checkpoint saved from the bare backbone must still load into the wrapper
    bare = os.path.join(tmp, "bare.pt")
    torch.save({"model_state": model.backbone.state_dict(), "val_acc": 1.0}, bare)
    fresh2 = Normalized(build_model("alexnet_half", 10), (0.5,) * 3, (0.25,) * 3)
    meta2 = checkpoint.load(fresh2, bare, torch.device("cpu"))
    check(f"bare-backbone checkpoint loads ({meta2.get('_matched')})",
          meta2.get("_matched") == "add 'backbone.'")

    # a mismatched classifier must be named, not dumped as missing keys
    wrong = os.path.join(tmp, "wrong.pt")
    torch.save({"model_state": build_model("alexnet_half", 100).state_dict()}, wrong)
    try:
        checkpoint.load(fresh2, wrong, torch.device("cpu"))
        check("wrong num_classes rejected", False)
    except RuntimeError as exc:
        check("wrong num_classes reports a shape mismatch", "shapes" in str(exc))


def test_configs_end_to_end(tmp):
    # satisfy the data-root guard; the stub dataset never reads these
    for folder in ("cifar-10-batches-py", "cifar-100-python"):
        os.makedirs(os.path.join(tmp, folder), exist_ok=True)
    for path in CONFIGS:
        print(f"end-to-end: {path}")
        num_classes = 100 if "cifar100" in path else 10
        stub = _stub_cifar(num_classes)
        real10, real100 = torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100
        torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR100 = stub
        import ankd.data as data_module
        data_module.DATASETS["cifar10"]["cls"] = stub
        data_module.DATASETS["cifar100"]["cls"] = stub
        try:
            out = os.path.join(tmp, os.path.basename(path))
            code = main([
                "--config", path,
                "--data.root", tmp, "--data.download", "false",
                "--runtime.device", "cpu", "--runtime.num_workers", "0",
                "--teacher.epochs", "1", "--student.epochs", "2",
                "--teacher.eval_train_samples", "64",
                "--noise.num_samples", "200",
                "--teacher.ckpt", f"{out}_t.pt",
                "--student.ckpt_best", f"{out}_s_best.pt",
                "--student.ckpt_last", f"{out}_s_last.pt",
            ])
            check("exit code 0", code == 0)
            for suffix in ("_t.pt", "_s_best.pt", "_s_last.pt"):
                check(f"wrote {os.path.basename(out)}{suffix}",
                      os.path.isfile(f"{out}{suffix}"))
        finally:
            torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100 = real10, real100
            data_module.DATASETS["cifar10"]["cls"] = real10
            data_module.DATASETS["cifar100"]["cls"] = real100


def test_config_overrides():
    print("config:")
    cfg, _ = load_config(["--config", CONFIGS[0], "--student.epochs", "3",
                          "--augment.n_random_ops=8"])
    check("dotted override with a space", cfg.student.epochs == 3)
    check("dotted override with '='", cfg.augment.n_random_ops == 8)
    check("config default is 40k noise samples", cfg.noise.num_samples == 40000)
    try:
        load_config(["--nope.key", "1"])
        check("unknown key rejected", False)
    except KeyError:
        check("unknown key rejected", True)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="ankd-smoke-")
    try:
        test_units()
        test_config_overrides()
        test_checkpoint_roundtrip(tmp)
        test_configs_end_to_end(tmp)
        print("\nALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
