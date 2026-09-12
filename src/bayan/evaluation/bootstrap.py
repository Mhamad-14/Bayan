"""Lab 6: bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def _array(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if arr.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """Return mean point estimate and percentile bootstrap CI."""
    values = _array(values)

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[indices].mean(axis=1)

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return point, min(lo, point), max(hi, point)


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """Return paired mean difference a-b and percentile bootstrap CI."""
    a = _array(a)
    b = _array(b)

    if len(a) != len(b):
        raise ValueError("paired inputs must have equal length")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    diff = a - b
    delta = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[indices].mean(axis=1)

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return delta, min(lo, delta), max(hi, delta)
