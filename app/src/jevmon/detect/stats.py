"""Small statistics helpers for context: smoothed surprisal, percentile rank, fork-rate baseline.

These compute numbers the model reads. They never assign severity.
"""

import math
from dataclasses import dataclass


def surprisal(count: int, total: int, distinct: int, alpha: float = 0.5) -> float:
    """-log p under add-alpha smoothing. Unseen events get high surprisal; cold start gives 0."""
    vocab = distinct + 1
    probability = (count + alpha) / (total + alpha * vocab)
    return -math.log(max(probability, 1e-12))


def percentile(values: list[float], q: float) -> float | None:
    """Value at quantile `q` in [0, 1], or None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def percentile_rank(values: list[float], value: float) -> float | None:
    """Percentage of samples at or below `value`, or None for an empty sample."""
    if not values:
        return None
    below = sum(1 for v in values if v <= value)
    return 100.0 * below / len(values)


@dataclass(slots=True)
class ForkRateBaseline:
    """Exponentially weighted baseline of the fork rate. Statistical context, not a threshold."""

    alpha: float = 0.3
    samples: int = 0
    ewma: float = 0.0

    def update(self, rate: float) -> None:
        self.samples += 1
        if self.samples == 1:
            self.ewma = rate
        else:
            self.ewma = self.alpha * rate + (1 - self.alpha) * self.ewma

    def ratio(self, rate: float) -> float:
        if self.ewma <= 0:
            return 0.0
        return rate / self.ewma
