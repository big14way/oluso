"""Interval estimates used in every results table.

- Wilson score interval for a single rate.
- Cluster bootstrap for rates pooled across languages: the same JailbreakBench item
  (its `index`) appears in every language, so items are resampled by index, not by row.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ratio(
    clusters: list, num: list[float], den: list[float], n_boot: int = 2000, seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float]]:  # fmt: skip
    """sum(num)/sum(den) with a percentile CI from resampling whole clusters."""
    by: dict = defaultdict(lambda: [0.0, 0.0])
    for c, a, b in zip(clusters, num, den, strict=True):
        by[c][0] += a
        by[c][1] += b
    keys = list(by)
    arr = np.array([by[k] for k in keys])
    point = arr[:, 0].sum() / arr[:, 1].sum() if arr[:, 1].sum() else float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(keys), len(keys))].sum(axis=0)
        if s[1]:
            stats.append(s[0] / s[1])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), (float(lo), float(hi))
