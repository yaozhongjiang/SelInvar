"""Task-level paired inference.

THE INDEPENDENT UNIT IS THE TASK, NOT THE TRIAL. Every task contributes seven
renderings to every arm, and those seven share a latent state, a price and a
reservation utility. Treating them as independent would shrink every interval by
roughly sqrt(7). All resampling here is over task ids, carrying each task's whole
block of rows with it.

Comparisons between arms are paired on the task for the same reason: the arms
see identical rendered messages, so the between-task variance -- which is large,
since listing prices span four orders of magnitude -- cancels out of the
difference.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _task_means(df: pd.DataFrame, metric: str) -> pd.Series:
    """Per-task mean of `metric`, the quantity that gets resampled."""
    return df.groupby("task_id")[metric].mean()


def bootstrap_mean(df: pd.DataFrame, metric: str, reps: int = 5000,
                   seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """Task-level bootstrap CI for the mean of `metric`."""
    x = _task_means(df, metric).values.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_tasks": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, x.size, size=(reps, x.size))
    means = x[draws].mean(axis=1)
    return {"mean": float(x.mean()),
            "ci_low": float(np.quantile(means, alpha / 2)),
            "ci_high": float(np.quantile(means, 1 - alpha / 2)),
            "n_tasks": int(x.size)}


def paired_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str,
                reps: int = 5000, seed: int = 42,
                alpha: float = 0.05) -> Dict[str, float]:
    """Paired bootstrap CI and permutation p-value for mean(a) - mean(b).

    Tasks present in only one arm are dropped, and the number kept is reported,
    so a silently shrinking intersection cannot masquerade as a clean result.
    """
    a, b = _task_means(df_a, metric), _task_means(df_b, metric)
    common = a.index.intersection(b.index)
    a, b = a.loc[common].values.astype(float), b.loc[common].values.astype(float)
    d = a - b
    n = d.size
    if n == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n_tasks": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    boot = d[idx].mean(axis=1)

    # Permutation on the sign of each paired difference: the exchangeability
    # that holds under "the two arms are the same policy on this task".
    signs = rng.choice([-1.0, 1.0], size=(reps, n))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())

    return {"diff": float(d.mean()),
            "ci_low": float(np.quantile(boot, alpha / 2)),
            "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
            "p_value": p, "n_tasks": int(n)}


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Step-down correction; returns the reject/keep mask in input order."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def attack_effect(df: pd.DataFrame, buyer: str, metric: str = "accepted",
                  reps: int = 5000, seed: int = 42) -> pd.DataFrame:
    """RQ1: per-attack paired change against the same arm's clean rendering."""
    sub = df[df.buyer_id == buyer]
    clean = sub[sub.attack_id == "clean"]
    rows = []
    for aid in sorted(sub.attack_id.unique()):
        if aid == "clean":
            continue
        r = paired_diff(sub[sub.attack_id == aid], clean, metric, reps, seed)
        rows.append({"buyer_id": buyer, "attack_id": aid, "metric": metric, **r})
    out = pd.DataFrame(rows)
    if len(out):
        out["reject_holm"] = holm_bonferroni(out.p_value.values)
    return out


def arm_comparison(df: pd.DataFrame, reference: str, metric: str,
                   others: Optional[Sequence[str]] = None,
                   reps: int = 5000, seed: int = 42) -> pd.DataFrame:
    """RQ2: `reference` minus each other arm, paired on the task."""
    arms = list(others) if others else [b for b in sorted(df.buyer_id.unique())
                                        if b != reference]
    ref = df[df.buyer_id == reference]
    rows = []
    for b in arms:
        r = paired_diff(ref, df[df.buyer_id == b], metric, reps, seed)
        rows.append({"reference": reference, "other": b, "metric": metric, **r})
    out = pd.DataFrame(rows)
    if len(out):
        out["reject_holm"] = holm_bonferroni(out.p_value.values)
    return out
