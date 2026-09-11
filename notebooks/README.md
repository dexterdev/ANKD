# Original notebooks

Historical, kept for reference. The `ankd` package supersedes them and is what the
configs run — start there, not here.

* `distillation_through_augmented_noise_CIFAR100_optimizedA100(2).ipynb` — the original
  ResNet-34 → ResNet-18 experiment on CIFAR-100, whose protocol the package adopts as
  its shared default.
* `distillation_through_augmented_noise_CIFAR10_AlexNet_to_AlexNetHalf.ipynb`
* `distillation_through_augmented_noise_CIFAR10_ViT8_to_ViT4.ipynb`

Where the package differs:

* 40,000 noise samples for the student (the notebooks used 150,000);
* 4 random augmentation ops (the notebooks used 8);
* `run` / `train` / `test` accuracy reported for the teacher every epoch, train
  accuracy measured on a clean view of the training set;
* the student is scored on the real test set only, and the data-flow contract that
  guarantees it is checked by `tests/invariants_test.py`;
* student checkpoints (best and final) for every experiment.
