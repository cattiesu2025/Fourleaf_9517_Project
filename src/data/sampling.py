"""Sampling helpers for long-tail class imbalance experiments."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def class_counts(targets: Sequence[int]) -> dict[int, int]:
    """Return per-class sample counts."""
    return dict(Counter(int(t) for t in targets))


def make_weighted_sampler(
    targets: Sequence[int],
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Create an inverse-frequency sampler for long-tail training.

    The sampler keeps the original CSV unchanged and balances classes during
    training by sampling minority-class examples more often.
    """
    if not targets:
        raise ValueError("targets must not be empty when building a weighted sampler")

    counts = class_counts(targets)
    weights = torch.DoubleTensor([1.0 / counts[int(target)] for target in targets])
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples or len(targets),
        replacement=replacement,
    )
