"""Precompute the robust bound as a function of the ambiguity radius.

Calibrating the four coefficients of eps(m) by re-running the LP for every
candidate is hopeless at benchmark scale: one config over a 1,300-task split is
~9,000 transport problems. But the gate's decision depends on the radius only
through `robust_lower_bound(signal, prefs, eps)`, so that curve can be tabulated
once per rendered signal and every later config evaluated by interpolation.

The LP value is concave and non-increasing in eps (it is the optimal value of a
linear program in its right-hand side), so linear interpolation between grid
points sits *below* the true curve: the approximation is conservative, never
optimistic. On a 0.02 grid the gap is under 1e-3 utility units.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .attacks import render
from .buyers import (_clauses, bound_curve, disclosed_impact, safe_price,
                     uncertainty_scores)
from .dataset import true_utility

RADIUS_GRID = np.round(np.arange(0.0, 1.5001, 0.02), 4)


def precompute(tasks, attacks: Sequence[str], seed: int = 42, repeat: int = 0,
               grid: np.ndarray = RADIUS_GRID,
               informative_evidence: bool = True,
               progress: bool = True) -> Dict[str, Any]:
    """Tabulate everything the gate needs, independent of its coefficients."""
    from .experiment import trial_seed

    rows: List[Dict[str, Any]] = []
    curves: List[np.ndarray] = []
    for i, t in enumerate(tasks):
        rng = np.random.default_rng(trial_seed(t.task_id, "render", repeat, seed))
        for aid in attacks:
            sig = render(t, aid, rng, informative_evidence=informative_evidence)
            # Urgency decoupling happens before the economic evaluator, so the
            # tabulated curve already reflects it.
            s = dict(sig)
            s["expires_in_minutes"] = None
            s["urgency_text"] = ""

            u_x, u_phi, u_s = uncertainty_scores(s)
            penalty = safe_price(s) - float(s["headline_price"])
            nominal = sum(float(c["amount"]) * float(c["p_claimed"])
                          for c in _clauses(s))
            u_true = true_utility(sig, t.oracle)
            res = float(t.oracle["reservation_utility"])
            rows.append({
                "task_id": t.task_id, "attack_id": aid, "category": t.category,
                "listing_price": float(t.listing_price),
                "u_x": u_x, "u_phi": u_phi, "u_s": u_s,
                "canon_penalty": max(0.0, penalty - nominal),
                "reservation": res, "true_utility": u_true,
                "is_good": int(u_true >= res),
            })
            curves.append(bound_curve(s, t.buyer_preferences, grid))
        if progress and (i + 1) % 100 == 0:
            print(f"  precomputed {i+1}/{len(tasks)} tasks", flush=True)

    return {"table": pd.DataFrame(rows), "curves": np.vstack(curves), "grid": np.asarray(grid)}


def evaluate_config(pre: Dict[str, Any], eps0: float, lambda_x: float,
                    lambda_phi: float, lambda_s: float, delta: float = 0.0,
                    canonicalize: bool = True,
                    fixed_radius: float | None = None) -> Dict[str, float]:
    """Metrics for one coefficient vector, vectorized over all renderings."""
    tab, curves, grid = pre["table"], pre["curves"], pre["grid"]
    if fixed_radius is not None:
        eps = np.full(len(tab), float(fixed_radius))
    else:
        eps = (eps0 + lambda_x * tab.u_x.values + lambda_phi * tab.u_phi.values
               + lambda_s * tab.u_s.values)
    eps = np.clip(eps, grid[0], grid[-1])

    # Per-row interpolation along that row's own curve.
    idx = np.searchsorted(grid, eps).clip(1, len(grid) - 1)
    g0, g1 = grid[idx - 1], grid[idx]
    w = np.where(g1 > g0, (eps - g0) / np.maximum(g1 - g0, 1e-12), 0.0)
    r = np.arange(len(tab))
    lb = curves[r, idx - 1] * (1 - w) + curves[r, idx] * w

    if canonicalize:
        lb = lb - tab.canon_penalty.values

    accept = (lb >= tab.reservation.values + delta).astype(int)
    good = tab.is_good.values
    L = np.maximum(tab.listing_price.values, 1e-9)
    surplus = tab.true_utility.values - tab.reservation.values
    regret = (np.maximum(0.0, surplus) - np.where(accept == 1, surplus, 0.0)) / L

    return {
        "HAR": float(np.mean(accept * (1 - good))),
        "valid": float(np.mean(accept * good)),
        "missed": float(np.mean((1 - accept) * good)),
        "regret": float(np.mean(regret)),
        "accept_rate": float(np.mean(accept)),
        "radius": float(np.mean(eps)),
    }
