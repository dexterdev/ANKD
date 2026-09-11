"""ANKD -- data-free knowledge distillation through augmented noise.

The student never sees a real training image: synthetic noise is pushed through
an augmentation stack, a frozen teacher labels the augmented view on the fly, and
the student matches those labels with a temperature-scaled KL objective.
"""

__version__ = "0.1.0"
