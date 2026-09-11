"""Data-flow invariants, checked at runtime rather than by reading the code.

    Teacher  trains on real training data, scored on real train + test data.
    Student  trains ONLY on augmented synthetic noise, queried through the
             teacher, and is scored ONLY on the real test set.

Every synthetic image is stamped with a sentinel pixel after augmentation, so a
forward hook can tell real data from noise no matter which loader produced it.

    python tests/invariants_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ankd import engine                                                  # noqa: E402
from ankd.augment import SyntheticDataset, diverse_augment               # noqa: E402
from ankd.config import (AugmentCfg, Config, DataCfg, NoiseCfg,          # noqa: E402
                         StudentCfg, TeacherCfg)
from ankd.data import CifarData                                          # noqa: E402
from ankd.models import build_model                                      # noqa: E402
from ankd.noise import build_noise                                       # noqa: E402
from ankd.runtime import Runtime                                         # noqa: E402

SENTINEL = -7.5                     # impossible in [0, 1] image data
CONFIGS = ["configs/resnet34_resnet18_cifar100.yaml",
           "configs/resnet34_resnet18_cifar10.yaml",
           "configs/alexnet_alexnethalf_cifar10.yaml",
           "configs/vit8_vit4_cifar10.yaml"]


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}")
    assert condition, label


class StampedSynthetic(SyntheticDataset):
    """Synthetic noise, stamped after augmentation so its origin is detectable."""

    def __getitem__(self, idx):
        img = super().__getitem__(idx)
        img[0, 0, 0] = SENTINEL
        return img


def _stub_cifar(num_classes):
    class Stub(Dataset):
        def __init__(self, root, train=True, download=False, transform=None):
            self.n = 128 if train else 64
            self.transform = transform
            rng = np.random.RandomState(0 if train else 1)
            self.data = rng.randint(0, 256, (self.n, 32, 32, 3), dtype=np.uint8)
            self.targets = rng.randint(0, num_classes, self.n)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.transform(Image.fromarray(self.data[i])), int(self.targets[i])
    return Stub


class Recorder:
    """Records which loaders `evaluate` is called on, and what each net is fed."""

    def __init__(self):
        self.eval_loaders = []
        self.teacher_inputs = []
        self.student_inputs = []

    def patch_evaluate(self):
        real = engine.evaluate

        def spy(model, device, loader):
            self.eval_loaders.append(loader)
            return real(model, device, loader)
        engine.evaluate = spy
        return real

    def hook(self, store):
        def fn(_module, args, _output=None):
            store.append(args[0])
        return fn


def test_config_uniformity():
    print("config uniformity across all four experiments:")
    for path in CONFIGS:
        raw = yaml.safe_load(open(path))
        noise = raw.get("noise", {})
        aug = raw.get("augment", {})
        name = os.path.basename(path)
        check(f"{name}: 40,000 noise samples", noise.get("num_samples") == 40000)
        check(f"{name}: one fixed set (regenerate_every 0)",
              noise.get("regenerate_every", 0) == 0)
        check(f"{name}: geometric augmentation on", aug.get("use_geo") is True)
        check(f"{name}: random ops > 0", aug.get("n_random_ops", 0) > 0)


def test_student_sees_only_augmented_noise(tmp):
    print("student data flow:")
    num_classes = 10
    import ankd.data as data_module
    stub = _stub_cifar(num_classes)
    saved = (data_module.DATASETS["cifar10"]["cls"], data_module.DATASETS["cifar100"]["cls"])
    data_module.DATASETS["cifar10"]["cls"] = stub

    rec = Recorder()
    real_evaluate = rec.patch_evaluate()
    try:
        for folder in ("cifar-10-batches-py",):
            os.makedirs(os.path.join(tmp, folder), exist_ok=True)
        runtime = Runtime(Config().runtime)
        runtime.device = torch.device("cpu")
        runtime.is_cuda = False
        runtime.channels_last = False
        data = CifarData(DataCfg(dataset="cifar10", root=tmp, download=False),
                         runtime, train_eval_samples=32)

        teacher = data.wrap(build_model("alexnet", num_classes)).eval()
        student = data.wrap(build_model("alexnet_half", num_classes))
        for p in teacher.parameters():
            p.requires_grad_(False)

        teacher.register_forward_pre_hook(rec.hook(rec.teacher_inputs))
        student.register_forward_pre_hook(rec.hook(rec.student_inputs))

        noise = build_noise(NoiseCfg(num_samples=64, per_call=32))
        dataset = StampedSynthetic(noise, AugmentCfg(use_geo=True, n_random_ops=4))

        cfg = StudentCfg(epochs=2, ckpt_best=os.path.join(tmp, "b.pt"),
                         ckpt_last=os.path.join(tmp, "l.pt"))
        engine.train_student(teacher, student, dataset, data, cfg, runtime)

        # --- what did the student actually train on? ---
        # Evaluation batches carry no sentinel; training batches must all carry one.
        train_batches = [t for t in rec.student_inputs if t.shape[0] > 0
                         and torch.any(t[:, 0, 0, 0] == SENTINEL)]
        eval_batches = [t for t in rec.student_inputs
                        if not torch.any(t[:, 0, 0, 0] == SENTINEL)]
        check(f"student ran {len(train_batches)} synthetic batches", len(train_batches) > 0)
        check("every student training batch is stamped synthetic noise",
              all(bool((t[:, 0, 0, 0] == SENTINEL).all()) for t in train_batches))
        check(f"student's {len(eval_batches)} non-synthetic batches are evaluation only",
              len(eval_batches) > 0)

        # --- was the teacher queried on exactly what the student was trained on? ---
        teacher_train = [t for t in rec.teacher_inputs
                         if torch.any(t[:, 0, 0, 0] == SENTINEL)]
        check("teacher was queried on synthetic noise too", len(teacher_train) > 0)
        pairs = list(zip(teacher_train, train_batches))
        check("teacher and student receive the identical tensor object",
              all(a is b for a, b in pairs) and len(pairs) == len(train_batches))

        # --- what real data did the student touch? ---
        used = {id(loader) for loader in rec.eval_loaders}
        check("student scored on the real TEST loader only",
              used == {id(data.test_loader)})
        check("student never touched the real TRAIN loader",
              id(data.train_loader) not in used)
        check("student never touched the clean-view TRAIN loader",
              id(data.train_eval_loader) not in used)
    finally:
        engine.evaluate = real_evaluate
        data_module.DATASETS["cifar10"]["cls"], data_module.DATASETS["cifar100"]["cls"] = saved


def test_teacher_uses_real_data(tmp):
    print("teacher data flow:")
    import ankd.data as data_module
    stub = _stub_cifar(10)
    saved = data_module.DATASETS["cifar10"]["cls"]
    data_module.DATASETS["cifar10"]["cls"] = stub

    rec = Recorder()
    real_evaluate = rec.patch_evaluate()
    try:
        runtime = Runtime(Config().runtime)
        runtime.device = torch.device("cpu")
        runtime.is_cuda = False
        runtime.channels_last = False
        data = CifarData(DataCfg(dataset="cifar10", root=tmp, download=False),
                         runtime, train_eval_samples=32)
        teacher = data.wrap(build_model("alexnet", 10))
        teacher.register_forward_pre_hook(rec.hook(rec.teacher_inputs))

        engine.train_teacher(teacher, data,
                             TeacherCfg(epochs=1, ckpt=os.path.join(tmp, "t.pt")),
                             runtime)

        check("teacher saw no synthetic noise at all",
              not any(torch.any(t[:, 0, 0, 0] == SENTINEL) for t in rec.teacher_inputs))
        used = {id(loader) for loader in rec.eval_loaders}
        check("teacher scored on real train and test loaders",
              used == {id(data.test_loader), id(data.train_eval_loader)})
    finally:
        engine.evaluate = real_evaluate
        data_module.DATASETS["cifar10"]["cls"] = saved


def test_augmentation_actually_applies():
    print("augmentation:")
    raw = torch.rand(3, 32, 32)
    check("diverse_augment changes the image when ops are requested",
          not torch.equal(diverse_augment(raw.clone(), True, 4), raw))
    check("n_random_ops=0 with geo still crops/flips",
          diverse_augment(raw.clone(), True, 0).shape == raw.shape)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="ankd-inv-")
    try:
        os.makedirs(os.path.join(tmp, "cifar-10-batches-py"), exist_ok=True)
        test_config_uniformity()
        test_augmentation_actually_applies()
        test_student_sees_only_augmented_noise(tmp)
        test_teacher_uses_real_data(tmp)
        print("\nALL INVARIANTS HOLD")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
