# Original notebooks

Kept for reference; the `ankd` package supersedes them and is what the configs run.

* `distillation_through_augmented_noise_CIFAR100_optimizedA100(2).ipynb` — the original
  ResNet-34 → ResNet-18 experiment on CIFAR-100, whose protocol the package adopts as
  its shared default.
* `distillation_through_augmented_noise_CIFAR10_AlexNet_to_AlexNetHalf.ipynb`
* `distillation_through_augmented_noise_CIFAR10_ViT8_to_ViT4.ipynb`

Differences in the package: 40,000 noise samples (was 150,000), 4 random augmentation
ops (was 8), train **and** test accuracy reported every epoch for both networks, and
student checkpoints written for every experiment.
