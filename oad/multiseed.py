"""
multiseed.py -- reusable variance reporting (ROADMAP Phase A3).

Small-sample 95% confidence intervals (Student-t) for a metric across seeds, plus
a PAIRED interval for a system-vs-system difference measured on the same seeds
(the correct test when comparing two configurations on a shared seed set).

No experiment-specific logic lives here; any driver can import it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from scipy.stats import t as _t


@dataclass
class Stat:
    mean: float
    ci: float          # half-width of the 95% CI (mean +/- ci)
    sd: float
    n: int
    lo: float
    hi: float

    def fmt(self, d: int = 1) -> str:
        return f"{self.mean:.{d}f} +/- {self.ci:.{d}f}"


def _tcrit(n: int) -> float:
    return float(_t.ppf(0.975, df=max(1, n - 1)))


def ci95(values: Sequence[float]) -> Stat:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return Stat(m, 0.0, 0.0, n, m, m)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    half = _tcrit(n) * sd / math.sqrt(n)
    return Stat(m, half, sd, n, m - half, m + half)


def paired_ci95(a: Sequence[float], b: Sequence[float]) -> Stat:
    """95% CI on the paired difference (a_i - b_i). Same seed order assumed."""
    if len(a) != len(b):
        raise ValueError("paired_ci95 needs equal-length, seed-aligned inputs")
    return ci95([ai - bi for ai, bi in zip(a, b)])


def summarize_runs(runs: List[Dict[str, float]], keys: Sequence[str]) -> Dict[str, Stat]:
    """runs: list of per-seed metric dicts. Returns {key: Stat} over seeds."""
    return {k: ci95([r[k] for r in runs]) for k in keys}
