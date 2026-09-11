"""Training loops, shared by all three experiments.

The distillation protocol is the one from the original ResNet-34 -> ResNet-18
notebook and is identical across experiments: a 16 -> 2048 batch-size ramp, a
cosine LR schedule that resets at each ramp step, and a temperature-scaled KL
objective against the frozen teacher. Only the optimizer and a few teacher-side
knobs are configurable, so each experiment can override what it must.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import checkpoint
from .config import StudentCfg, TeacherCfg
from .runtime import Runtime

# Shared distillation protocol ---------------------------------------------- #
BATCH_RAMP = [(25, 16), (50, 64), (75, 128), (100, 256), (125, 512), (150, 1024)]
RAMP_TAIL = 2048


def batch_size_for(epoch: int, max_batch: int) -> int:
    """The ramp, capped at what the GPU can hold."""
    for limit, size in BATCH_RAMP:
        if epoch < limit:
            return min(size, max_batch)
    return min(RAMP_TAIL, max_batch)


def batch_aligned_lr(epoch: int) -> float:
    """Cosine decay within each phase, resetting at every batch-size change so the
    schedule lines up with the ramp. Returns a LambdaLR multiplier on lr=0.01."""
    start = 0.001 if 25 <= epoch < 50 else 0.01
    end = 0.00001
    phase_start = min(epoch // 25, 6) * 25
    phase_len = 50 if epoch >= 150 else 25
    t = epoch - phase_start
    cos_factor = 0.5 * (1 + math.cos(math.pi * t / max(phase_len - 1, 1)))
    return (end + (start - end) * cos_factor) / 0.01


def student_lr_lambda(cfg: StudentCfg):
    """LambdaLR multiplier for the student.

    "ramp_aligned" is the shared protocol: cosine within each batch-size phase,
    restarting at every ramp step. "cosine" decays once across the whole run,
    which suits an AdamW/transformer student -- the restarts otherwise throw it
    back to full learning rate five times late in training.
    """
    warmup = max(cfg.warmup_epochs, 0)

    def lambda_fn(epoch: int) -> float:
        if warmup and epoch < warmup:
            scale = (epoch + 1) / warmup
        else:
            scale = 1.0
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
        if runtime.is_cuda:
            import inspect
            if "fused" in inspect.signature(torch.optim.SGD.__init__).parameters:
                extra["fused"] = True       # fused kernel; same update rule
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


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


# --------------------------------------------------------------------------- #
# teacher
# --------------------------------------------------------------------------- #
def train_teacher(teacher, data, cfg: TeacherCfg, runtime: Runtime):
    """Supervised training on the real dataset, saving the best epoch as it goes.

    Reports train and test accuracy each epoch. Train accuracy is measured on a
    clean (un-augmented) view of the training set, so it reflects fit rather than
    augmentation strength, and the train/test gap is readable as overfitting.
    """
    device = runtime.device
    teacher = runtime.prepare(teacher.to(device))
    opt = make_optimizer(teacher.parameters(), cfg.optimizer, cfg.lr,
                         cfg.momentum, cfg.weight_decay, runtime)

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
        epoch_loss = 0.0
        for x, y in data.train_loader:
            x = runtime.to_input(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                loss = criterion(teacher(x), y)
            if needs_scaler:
                scaler.scale(loss).backward()
                if cfg.clip_grad:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(teacher.parameters(), cfg.clip_grad)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.clip_grad:
                    nn.utils.clip_grad_norm_(teacher.parameters(), cfg.clip_grad)
                opt.step()
            epoch_loss += loss.item()
        sched.step()

        test_acc = evaluate(teacher, device, data.test_loader)
        show_train = (cfg.eval_train_every
                      and (epoch + 1) % cfg.eval_train_every == 0)
        train_acc = evaluate(teacher, device, data.train_eval_loader) if show_train else None

        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint.save(teacher, cfg.ckpt, val_acc=test_acc,
                            train_acc=train_acc, epoch=epoch + 1)

        train_part = f"train={train_acc:.2f}%  " if train_acc is not None else ""
        print(f"[teacher] epoch {epoch + 1}/{cfg.epochs}  {train_part}"
              f"test={test_acc:.2f}%  best={best_acc:.2f}%  "
              f"loss={epoch_loss / len(data.train_loader):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {_fmt(time.time() - started)}")

    print(f"[teacher] done in {_fmt(time.time() - started)}. best test={best_acc:.2f}% "
          f"-> {cfg.ckpt}")
    return teacher, best_acc


# --------------------------------------------------------------------------- #
# student
# --------------------------------------------------------------------------- #
def train_student(teacher, student, dataset, data, cfg: StudentCfg, runtime: Runtime,
                  regenerate=None, regenerate_every: int = 0):
    """Distil the frozen teacher into the student on synthetic noise.

    The student never sees a real training image; the CIFAR loaders are used only
    to score it. The best-scoring epoch is written to disk as it goes, so a long
    run that drifts or is interrupted still leaves a usable student.
    """
    device = runtime.device
    student = runtime.prepare(student.to(device))
    teacher = runtime.prepare(teacher.to(device))
    teacher.eval()          # dropout/BN are active in train mode -- keep targets fixed

    opt = make_optimizer(student.parameters(), cfg.optimizer, cfg.lr,
                         cfg.momentum, cfg.weight_decay, runtime)
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
            # Fresh noise widens what the teacher is queried on. The loader cache
            # must be dropped too: persistent workers hold their own pickled copy
            # of the dataset, so mutating it in the parent alone would not reach them.
            dataset.imgs = regenerate()
            loader_cache.clear()
            print(f"[student] epoch {epoch + 1}: redrew {dataset.imgs.shape[0]:,} "
                  f"synthetic images")

        student.train()
        loader = loader_for(epoch)
        epoch_loss = 0.0
        for x in loader:
            x = runtime.to_input(x.to(device, non_blocking=True))
            with torch.no_grad(), autocast_ctx:   # teacher frozen; no_grad also keeps
                targets = teacher(x)              # autograd off the input side
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                loss = kd_loss(x, student, targets, cfg.temperature)
            if needs_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            epoch_loss += loss.item()
        sched.step()

        test_acc = evaluate(student, device, data.test_loader)
        show_train = cfg.eval_train_every and (epoch + 1) % cfg.eval_train_every == 0
        train_acc = evaluate(student, device, data.train_eval_loader) if show_train else None

        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint.save(student, cfg.ckpt_best, val_acc=test_acc,
                            train_acc=train_acc, epoch=epoch + 1,
                            temperature=cfg.temperature)

        train_part = f"train={train_acc:.2f}%  " if train_acc is not None else ""
        print(f"[student] epoch {epoch + 1}/{cfg.epochs}  {train_part}"
              f"test={test_acc:.2f}%  best={best_acc:.2f}%  "
              f"loss={epoch_loss / len(loader):.4f}  "
              f"bs={batch_size_for(epoch, runtime.max_batch)}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {_fmt(time.time() - started)}")

    final_acc = evaluate(student, device, data.test_loader)
    checkpoint.save(student, cfg.ckpt_last, val_acc=final_acc, epoch=cfg.epochs,
                    temperature=cfg.temperature)
    print(f"[student] done in {_fmt(time.time() - started)}. "
          f"best test={best_acc:.2f}% -> {cfg.ckpt_best}")
    print(f"[student] final epoch={final_acc:.2f}% -> {cfg.ckpt_last}")
    return student, best_acc


@torch.no_grad()
def temperature_report(teacher, dataset, device, temperature, num_classes, n=256):
    """How peaked are the teacher's targets at this temperature?

    With few classes a large T flattens the softmax so far that there is almost
    no signal to distil -- if the reported value sits at chance, lower T.
    """
    teacher.eval()
    probe = torch.stack([dataset[i] for i in range(min(n, len(dataset)))]).to(device)
    logits = teacher(probe)
    flat = F.softmax(logits / temperature, dim=-1).max(-1).values.mean().item()
    sharp = F.softmax(logits, dim=-1).max(-1).values.mean().item()
    print(f"teacher on noise -- mean max prob @ T={temperature:g}: {flat:.4f}   "
          f"@ T=1: {sharp:.4f}   (chance = {1 / num_classes:.4f})")
    if flat < 1.5 / num_classes:
        print("  WARNING: targets are close to uniform at this temperature; "
              "there is little signal to distil. Consider lowering student.temperature.")
    return flat, sharp
