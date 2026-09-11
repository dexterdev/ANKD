"""Entry point: `python -m ankd.train --config configs/<experiment>.yaml`.

Stages:
    all      load the teacher checkpoint if present, else train it, then distil
    teacher  train the teacher, unconditionally
    student  distil, using an existing teacher checkpoint
"""

from __future__ import annotations

import os
import sys

from . import checkpoint, engine
from .augment import SyntheticDataset
from .config import load_config
from .data import CifarData
from .models import build_model, count_params
from .noise import build_noise
from .runtime import Runtime


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(argv=None) -> int:
    cfg, args = load_config(argv)
    if args.print_config:
        print(cfg.dump(), end="")
        return 0

    _rule(f"{cfg.name}  |  {cfg.model.teacher} -> {cfg.model.student}  "
          f"|  {cfg.data.dataset}")
    runtime = Runtime(cfg.runtime)
    print(runtime.describe())

    data = CifarData(cfg.data, runtime, cfg.teacher.eval_train_samples)
    print(f"data: {data.describe()}")

    teacher = data.wrap(build_model(cfg.model.teacher, data.num_classes)).to(runtime.device)
    student = data.wrap(build_model(cfg.model.student, data.num_classes)).to(runtime.device)
    t_params, s_params = count_params(teacher), count_params(student)
    print(f"teacher {cfg.model.teacher}: {t_params:,} params")
    print(f"student {cfg.model.student}: {s_params:,} params  "
          f"({s_params / t_params:.1%} of the teacher)")

    # ---- teacher ---------------------------------------------------------- #
    if args.stage in ("all", "teacher"):
        have_ckpt = os.path.exists(cfg.teacher.ckpt)
        if args.stage == "all" and have_ckpt:
            _rule("Teacher: loading checkpoint")
            meta = checkpoint.load(teacher, cfg.teacher.ckpt, runtime.device)
            print(f"loaded {cfg.teacher.ckpt}  {meta}")
        else:
            if have_ckpt:
                print(f"\nNote: overwriting existing checkpoint {cfg.teacher.ckpt}")
            _rule(f"Teacher: training from scratch for {cfg.teacher.epochs} epochs")
            engine.train_teacher(teacher, data, cfg.teacher, runtime)
            checkpoint.load(teacher, cfg.teacher.ckpt, runtime.device)  # best, not last
    else:
        if not os.path.exists(cfg.teacher.ckpt):
            print(f"error: stage 'student' needs a teacher checkpoint at "
                  f"{cfg.teacher.ckpt}; run --stage teacher first.", file=sys.stderr)
            return 2
        _rule("Teacher: loading checkpoint")
        meta = checkpoint.load(teacher, cfg.teacher.ckpt, runtime.device)
        print(f"loaded {cfg.teacher.ckpt}  {meta}")

    teacher.to(runtime.device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    t_train = engine.evaluate(teacher, runtime.device, data.train_eval_loader)
    t_test = engine.evaluate(teacher, runtime.device, data.test_loader)
    print(f"teacher accuracy: train={t_train:.2f}%  test={t_test:.2f}%")

    if args.stage == "teacher":
        return 0

    # ---- student ---------------------------------------------------------- #
    _rule(f"Student: distilling on {cfg.noise.num_samples:,} synthetic images")
    noise = build_noise(cfg.noise)
    dataset = SyntheticDataset(noise, cfg.augment)
    print(f"synthetic set: {tuple(noise.shape)}  "
          f"augment: geo={cfg.augment.use_geo} n_random_ops={cfg.augment.n_random_ops}")

    engine.temperature_report(teacher, dataset, runtime.device,
                              cfg.student.temperature, data.num_classes)

    _, best = engine.train_student(teacher, student, dataset, data, cfg.student, runtime)

    _rule("Summary")
    print(f"{cfg.name}")
    print(f"  teacher {cfg.model.teacher:14s} train={t_train:6.2f}%  test={t_test:6.2f}%")
    print(f"  student {cfg.model.student:14s} best test={best:.2f}%  "
          f"(retains {best / max(t_test, 1e-9):.1%} of the teacher)")
    print(f"  checkpoints: {cfg.student.ckpt_best}  |  {cfg.student.ckpt_last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
