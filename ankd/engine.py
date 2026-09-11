"""Training loops, shared by all four experiments.

Data-flow contract
------------------
This is the whole point of the method, so it is stated once here, enforced in the
loops below, and checked at runtime by ``tests/invariants_test.py``:

* the **teacher** trains on the real training set and is scored on real train and
  test data;
* the **student** trains *only* on augmented synthetic noise, labelled on the fly
  by the frozen teacher, and is scored *only* on the real test set;
* the teacher is queried on *exactly* the tensor the student is trained on -- the
  same augmented view, not the un-augmented noise behind it;
* the student has no train accuracy, because it has no training set on real data.

Distillation protocol
---------------------
Defaults are the protocol from the original ResNet-34 -> ResNet-18 notebook and
are identical across experiments: a 16 -> 2048 batch-size ramp (capped by VRAM), a
cosine learning-rate schedule that restarts at each ramp step, and a
temperature-scaled KL objective against the frozen teacher. The optimizer, an
optional warmup, and a single-cosine alternative to the restarting schedule are
configurable so an experiment can override what it must.
"""

from __future__ import annotations

import inspect
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import checkpoint
from .config import StudentCfg, TeacherCfg
from .runtime import Runtime

# --------------------------------------------------------------------------- #
# shared protocol
# --------------------------------------------------------------------------- #
BATCH_RAMP = [(25, 16), (50, 64), (75, 128), (100, 256), (125, 512), (150, 1024)]
RAMP_TAIL = 2048


def batch_size_for(epoch: int, max_batch: int) -> int:
    """The batch-size ramp, capped at what the GPU can hold."""
    for limit, size in BATCH_RAMP:
        if epoch < limit:
            return min(size, max_batch)
    return min(RAMP_TAIL, max_batch)


def batch_aligned_lr(epoch: int) -> float:
    """Cosine decay within each ramp phase, restarting at every batch-size change.

    Returns a LambdaLR multiplier expressed against lr=0.01, which is how the
    original notebook wrote it.
    """
    start = 0.001 if 25 <= epoch < 50 else 0.01
    end = 0.00001
    phase_start = min(epoch // 25, 6) * 25
    phase_len = 50 if epoch >= 150 else 25
    t = epoch - phase_start
    cos_factor = 0.5 * (1 + math.cos(math.pi * t / max(phase_len - 1, 1)))
    return (end + (start - end) * cos_factor) / 0.01


def student_lr_lambda(cfg: StudentCfg):
    """LambdaLR multiplier for the student, with optional warmup.

    ``ramp_aligned`` is the shared protocol above. ``cosine`` decays once across
    the whole run instead, which suits an AdamW/transformer student -- the
    restarts otherwise throw it back to full learning rate five times late in
    training.
    """
    warmup = max(cfg.warmup_epochs, 0)

    def lambda_fn(epoch: int) -> float:
        scale = (epoch + 1) / warmup if warmup and epoch < warmup else 1.0
        if cfg.schedule == "ramp_aligned":
            return batch_aligned_lr(epoch) * scale
        if cfg.schedule == "cosine":
            span = max(cfg.epochs - warmup, 1)
            progress = max(epoch - warmup, 0) / span
            return (0.001 + 0.999 * 0.5 * (1 + math.cos(math.pi * progress))) * scale
        raise ValueError(f"unknown student.schedule {cfg.schedule!r}; "
                         f"expected 'ramp_aligned' or 'cosine'")

    return lambda_fn


def kd_loss(x, student, teacher_logits, temperature):
    """Temperature-scaled KL between student and frozen teacher, scaled by T^2."""
    log_prob = F.log_softmax(student(x) / temperature, dim=-1)
    target = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return F.kl_div(log_prob, target, reduction="batchmean") * temperature ** 2


def make_optimizer(params, kind: str, lr: float, momentum: float, weight_decay: float,
                   runtime: Runtime):
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if kind == "sgd":
        extra = {}
        if runtime.is_cuda and "fused" in inspect.signature(
                torch.optim.SGD.__init__).parameters:
            extra["fused"] = True           # fused kernel; same update rule
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, **extra)
    raise ValueError(f"unknown optimizer {kind!r}; expected 'sgd' or 'adamw'")


@torch.no_grad()
def evaluate(model, device, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (torch.argmax(model(x), dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def _optimizer_step(loss, opt, scaler, needs_scaler, params=None, clip_grad=0.0):
    """One backward + step, with fp16 unscaling before clipping when scaling."""
    if needs_scaler:
        scaler.scale(loss).backward()
        if clip_grad:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, clip_grad)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(params, clip_grad)
        opt.step()


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


# --------------------------------------------------------------------------- #
# teacher -- real data only
# --------------------------------------------------------------------------- #
def train_teacher(teacher, data, cfg: TeacherCfg, runtime: Runtime):
    """Supervised training on the real dataset, saving the best epoch as it goes.

    Three accuracies are reported, because "train accuracy" is ambiguous here:

    ``run``    accuracy over the training pass itself. Free, since the logits are
               already computed, but measured on augmented inputs while the
               weights are still moving, so it largely tracks how hard the
               augmentation is.
    ``train``  a separate pass over the clean-view loader, which applies the test
               transform to the training set. This is the one to compare against
               test accuracy: the gap between them is overfitting.
    ``test``   the held-out set; this is what selects the checkpoint.
    """
    device = runtime.device
    teacher = runtime.prepare(teacher.to(device))
    params = list(teacher.parameters())
    opt = make_optimizer(params, cfg.optimizer, cfg.lr, cfg.momentum,
                         cfg.weight_decay, runtime)

    def lr_lambda(epoch):
        if cfg.warmup_epochs and epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        span = max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - cfg.warmup_epochs) / span))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    autocast_ctx, needs_scaler = runtime.autocast()
    scaler = runtime.scaler(needs_scaler)

    best_acc, started = 0.0, time.time()
    for epoch in range(cfg.epochs):
        teacher.train()
        epoch_loss, correct, seen = 0.0, 0, 0

        for x, y in data.train_loader:          # real training data
            x = runtime.to_input(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = teacher(x)
                loss = criterion(logits, y)
            _optimizer_step(loss, opt, scaler, needs_scaler, params, cfg.clip_grad)

            epoch_loss += loss.item()
            correct += (logits.detach().argmax(dim=1) == y).sum().item()
            seen += y.size(0)
        sched.step()

        run_acc = 100.0 * correct / max(seen, 1)
        test_acc = evaluate(teacher, device, data.test_loader)
        measure_train = (cfg.eval_train_every
                         and (epoch + 1) % cfg.eval_train_every == 0)
        train_acc = (evaluate(teacher, device, data.train_eval_loader)
                     if measure_train else None)

        if test_acc > best_acc:
            best_acc = test_acc
            # val_acc duplicates test_acc so checkpoints written before the rename
            # still load; both names mean the held-out score.
            checkpoint.save(teacher, cfg.ckpt, val_acc=test_acc, test_acc=test_acc,
                            train_acc=train_acc, run_acc=run_acc, epoch=epoch + 1)

        train_part = f"train={train_acc:.2f}%  " if train_acc is not None else ""
        gap_part = f"  gap={train_acc - test_acc:+.2f}" if train_acc is not None else ""
        print(f"[teacher] epoch {epoch + 1}/{cfg.epochs}  run={run_acc:.2f}%  "
              f"{train_part}test={test_acc:.2f}%  best={best_acc:.2f}%  "
              f"loss={epoch_loss / len(data.train_loader):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {_fmt(time.time() - started)}{gap_part}")

    print(f"[teacher] done in {_fmt(time.time() - started)}. "
          f"best test={best_acc:.2f}% -> {cfg.ckpt}")
    return teacher, best_acc


# --------------------------------------------------------------------------- #
# student -- synthetic noise only
# --------------------------------------------------------------------------- #
def train_student(teacher, student, dataset, data, cfg: StudentCfg, runtime: Runtime,
                  regenerate=None, regenerate_every: int = 0):
    """Distil the frozen teacher into the student on augmented synthetic noise.

    Upholds the data-flow contract in this module's docstring: `dataset` is the
    student's entire training distribution, the teacher is queried on the very
    tensor the student learns from, and `data` is touched only for the test-set
    score.

    `regenerate` is an optional callable returning a fresh noise tensor, redrawn
    every `regenerate_every` epochs. It is still synthetic noise; it never brings
    real data into training.
    """
    device = runtime.device
    student = runtime.prepare(student.to(device))
    teacher = runtime.prepare(teacher.to(device))
    teacher.eval()          # dropout/BN are active in train mode -- keep targets fixed

    opt = make_optimizer(student.parameters(), cfg.optimizer, cfg.lr, cfg.momentum,
                         cfg.weight_decay, runtime)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=student_lr_lambda(cfg))
    autocast_ctx, needs_scaler = runtime.autocast()
    scaler = runtime.scaler(needs_scaler)

    loader_cache: dict[int, DataLoader] = {}
    loader_kwargs = runtime.loader_kwargs()

    def loader_for(epoch: int) -> DataLoader:
        size = batch_size_for(epoch, runtime.max_batch)
        if size not in loader_cache:
            # drop_last keeps the batch shape constant so cudnn.benchmark stays hot
            loader_cache[size] = DataLoader(dataset, batch_size=size, shuffle=True,
                                            drop_last=True, **loader_kwargs)
        return loader_cache[size]

    best_acc, started = 0.0, time.time()
    for epoch in range(cfg.epochs):
        if regenerate and regenerate_every and epoch and epoch % regenerate_every == 0:
            # The loader cache must be dropped as well as the images replaced:
            # persistent workers hold their own pickled copy of the dataset, so
            # mutating it in the parent alone would never reach them.
            dataset.imgs = regenerate()
            loader_cache.clear()
            print(f"[student] epoch {epoch + 1}: redrew "
                  f"{dataset.imgs.shape[0]:,} synthetic images")

        student.train()
        loader = loader_for(epoch)
        epoch_loss = 0.0

        for x in loader:                        # synthetic noise, already augmented
            x = runtime.to_input(x.to(device, non_blocking=True))
            with torch.no_grad(), autocast_ctx:   # teacher frozen; no_grad also keeps
                targets = teacher(x)              # autograd off the input side
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                loss = kd_loss(x, student, targets, cfg.temperature)   # same x
            _optimizer_step(loss, opt, scaler, needs_scaler)
            epoch_loss += loss.item()
        sched.step()

        test_acc = evaluate(student, device, data.test_loader)   # real TEST set only

        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint.save(student, cfg.ckpt_best, val_acc=test_acc, test_acc=test_acc,
                            epoch=epoch + 1, temperature=cfg.temperature)

        print(f"[student] epoch {epoch + 1}/{cfg.epochs}  test={test_acc:.2f}%  "
              f"best={best_acc:.2f}%  loss={epoch_loss / len(loader):.4f}  "
              f"bs={batch_size_for(epoch, runtime.max_batch)}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {_fmt(time.time() - started)}")

    final_acc = evaluate(student, device, data.test_loader)
    checkpoint.save(student, cfg.ckpt_last, val_acc=final_acc, test_acc=final_acc,
                    epoch=cfg.epochs, temperature=cfg.temperature)
    print(f"[student] done in {_fmt(time.time() - started)}. "
          f"best test={best_acc:.2f}% -> {cfg.ckpt_best}")
    print(f"[student] final epoch={final_acc:.2f}% -> {cfg.ckpt_last}")
    return student, best_acc


@torch.no_grad()
def temperature_report(teacher, dataset, device, temperature, num_classes, n=256):
    """How peaked are the teacher's targets on noise at this temperature?

    With few classes a large T compresses the targets towards uniform. That does
    not zero the gradient -- the T^2 factor compensates, and the argmax survives --
    but it narrows the margin the student has to fit, so it is worth seeing.
    """
    teacher.eval()
    probe = torch.stack([dataset[i] for i in range(min(n, len(dataset)))]).to(device)
    logits = teacher(probe)
    flat = F.softmax(logits / temperature, dim=-1).max(-1).values.mean().item()
    sharp = F.softmax(logits, dim=-1).max(-1).values.mean().item()
    print(f"teacher on noise -- mean max prob @ T={temperature:g}: {flat:.4f}   "
          f"@ T=1: {sharp:.4f}   (chance = {1 / num_classes:.4f})")
    if flat < 1.5 / num_classes:
        print("  WARNING: targets are close to uniform at this temperature; the "
              "margin to fit is very narrow. Consider lowering student.temperature.")
    return flat, sharp
