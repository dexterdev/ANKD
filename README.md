# ANKD — data-free distillation through augmented noise

The student never sees a real training image. Synthetic noise is pushed through a
17-op augmentation stack, a frozen teacher labels the augmented view on the fly, and
the student matches those labels with a temperature-scaled KL objective. The real
dataset is used only to train the teacher and to score both networks.

Four experiments share one protocol:

| config | teacher | student | dataset | teacher params | student params |
|---|---|---|---|---|---|
| `resnet34_resnet18_cifar100.yaml` | ResNet-34 | ResNet-18 | CIFAR-100 | 21,328,292 | 11,220,132 |
| `resnet34_resnet18_cifar10.yaml` | ResNet-34 | ResNet-18 | CIFAR-10 | 21,282,122 | 11,173,962 |
| `alexnet_alexnethalf_cifar10.yaml` | AlexNet | AlexNet-Half | CIFAR-10 | 1,659,178 | 417,434 |
| `vit8_vit4_cifar10.yaml` | ViT-8 | ViT-4 | CIFAR-10 | 4,249,354 | 2,140,938 |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# teacher (trained if no checkpoint exists) then distillation
python -m ankd.train --config configs/vit8_vit4_cifar10.yaml

# one stage at a time
python -m ankd.train --config configs/alexnet_alexnethalf_cifar10.yaml --stage teacher
python -m ankd.train --config configs/alexnet_alexnethalf_cifar10.yaml --stage student

# override any config field from the command line
python -m ankd.train --config configs/vit8_vit4_cifar10.yaml \
    --student.epochs 50 --student.temperature 8 --noise.num_samples 4000

# see the resolved config without running anything
python -m ankd.train --config configs/vit8_vit4_cifar10.yaml --print-config
```

`--stage all` (the default) loads the teacher checkpoint if one exists and trains it
otherwise. `--stage teacher` always retrains.

## Data

`data.root` must be the directory that **contains** `cifar-10-batches-py` /
`cifar-100-python`, not that folder itself — torchvision resolves the data as
`<root>/<folder>/...`. Pointing it at the extracted folder is what produces
*"Dataset not found or corrupted"*. Set `data.download: true` to let torchvision
fetch it.

## The shared protocol

Defaults live in `ankd/config.py` and are the protocol from the original ResNet-34 →
ResNet-18 experiment. Each config overrides only what it must, and every deviation is
commented in the YAML with its reason.

* **Noise** — 40,000 synthetic images drawn from five families (smooth gradient,
  Perlin, uniform, Gabor, checkerboard), built in `[0, 1]` on CPU.
* **Augmentation** — `RandomCrop(32, pad=4, reflect)` + `RandomHorizontalFlip`, then 4
  ops sampled without replacement from a 17-op pool. Applied inside
  `Dataset.__getitem__`, so the teacher is queried on the augmented view.
* **Objective** — `KL(student/T ‖ teacher/T) · T²` at `T = 20`.
* **Schedule** — batch size ramps 16 → 2048 across 200 epochs; the LR is a cosine decay
  that resets at each ramp step. The ramp is capped by detected VRAM.
* **Teacher** — SGD at 0.01, momentum 0.9, weight decay 5e-4, cosine over 100 epochs.

Every run reports **train and test accuracy** each epoch for both networks. Train
accuracy is measured on a clean (un-augmented) view of the training set, so the
train/test gap reads as overfitting rather than augmentation strength. Set
`teacher.eval_train_every: 0` to skip it, or `teacher.eval_train_samples: 0` to use the
full training set instead of a 10k subsample.

### Where ViT-8 deviates, and why

A from-scratch ViT does not converge under the shared teacher recipe. Its **teacher**
block therefore uses AdamW at 1e-3, 5 warmup epochs into cosine, weight decay 0.05,
label smoothing 0.1, gradient clipping at 1.0, RandAugment, and batch 128 over 200
epochs. The distillation half is untouched, so the four experiments stay comparable.

`ViT-8` and `ViT-4` differ only in depth — 8 attention blocks against 4, with patch
size, width, heads and MLP ratio held fixed, so the student is the teacher with half
the blocks.

## Checkpoints

Teacher and student both write the best-scoring epoch as training proceeds; the student
additionally writes its final epoch. All use one container format
(`model_state` plus `val_acc` / `train_acc` / `epoch` / `temperature`), and
`ankd.checkpoint.load` tolerates a bare `state_dict`, other container keys, a `module.`
prefix from `DataParallel`, and the `backbone.` prefix the normalization wrapper adds.
When nothing matches it reports whether the mismatch is shapes (wrong class count) or
unrelated names (a different architecture).

## Temperature

Before distilling, the run prints how peaked the teacher's targets are at the chosen
temperature and warns if they are near uniform. With 10 classes, `T = 20` can flatten
the softmax so far that there is almost no signal left to distil — if you see that
warning, lower `student.temperature` (4–8 is a reasonable range).

## Layout

```
ankd/
  config.py      dataclass defaults, YAML loading, dotted CLI overrides
  runtime.py     device/worker/precision selection, GPU report, fast paths
  data.py        CIFAR loaders, clean-view train loader, normalization wrapper
  models/        resnet.py, alexnet.py, vit.py + registry
  noise.py       the five synthetic families
  augment.py     17-op pool and the synthetic dataset
  engine.py      teacher and student training loops, shared protocol
  checkpoint.py  save/load with container and prefix reconciliation
  train.py       entry point
configs/         one YAML per experiment
tests/           end-to-end smoke test against a stub CIFAR
notebooks/       the original notebooks this package replaces
```

## Tests

```bash
python tests/smoke_test.py
```

Runs every config through both stages against a stub CIFAR, checks parameter counts,
that the LR schedule matches the reference protocol exactly, and that checkpoints
round-trip. It exercises the code; it says nothing about accuracy.

## Reference

The AlexNet pair is from the ZSKD supplementary, Table 2 — Nayak et al.,
*Zero-Shot Knowledge Distillation in Deep Networks*, ICML 2019
([paper](http://proceedings.mlr.press/v97/nayak19a/nayak19a.pdf),
[code](https://github.com/vcl-iisc/ZSKD)).
