"""Shared construction of the calibration matrices used by the certification runs.

Both ``run_conformal.py`` (coverage) and ``run_strategic_certificate.py``
(decision risk) need the same three things per message: the radius-to-bound
curve, the observable features, and the realized distortion. Keeping one copy
matters more than usual here -- the two scripts are compared against each other
in the study, and a drift between two ``build`` implementations would show up as
a methodological difference that does not exist.

Urgency is stripped before the bound is computed. The scripted gate never reads
the urgency channel, so leaving it in would let a rendering difference that no
policy consumes perturb the curve.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.attacks import ATTACKS
from src.attacks import render as render_cb
from src.buyers import (_clauses, bound_curve, disclosed_impact, safe_price,
                        uncertainty_scores)
from src.dataset_dnd import DND_ATTACKS
from src.dataset_dnd import render as render_dnd
from src.experiment import trial_seed

GRID = np.round(np.arange(0.0, 1.2001, 0.01), 4)
FEATURES = ["u_x", "u_phi", "u_s", "n_clause", "disclosed", "ask_vs_mkt", "claim_q"]

ENVIRONMENTS = {
    "CraigslistBargain": ("data/tasks.jsonl", "data/splits.json", list(ATTACKS), render_cb),
    "AmazonHistoryPrice": ("data/tasks_ahp.jsonl", "data/splits_ahp.json", list(ATTACKS), render_cb),
    "DealOrNoDeal": ("data/tasks_dnd.jsonl", "data/splits_dnd.json", list(DND_ATTACKS), render_dnd),
}


def true_expected_price(sig: Dict[str, Any]) -> float:
    p = float(sig["headline_price"])
    for c in (sig.get("conditions") or []):
        p += float(c["amount"]) * float(c["p_true"])
    b = sig.get("bonus")
    if b:
        p -= float(b["amount"]) * float(b["p_true"])
    return p


def build(task_ids: Iterable[str], tasks_by_id: Dict[str, Any],
          attacks: List[str], render_fn: Callable) -> Tuple[np.ndarray, pd.DataFrame]:
    """Bound curves, observable features, and the realized distortion per message."""
    curves, rows = [], []
    for tid in task_ids:
        t = tasks_by_id[tid]
        rng = np.random.default_rng(trial_seed(tid, "render", 0, 42))
        v0 = float(t.buyer_preferences["value_at_quality_0"])
        span = float(t.buyer_preferences["value_at_quality_1"]) - v0
        for a in attacks:
            sig = render_fn(t, a, rng)
            s = dict(sig)
            s["expires_in_minutes"] = None
            s["urgency_text"] = ""
            curves.append(bound_curve(s, t.buyer_preferences, GRID))

            pen = safe_price(s) - float(s["headline_price"])
            nom = sum(float(c["amount"]) * float(c["p_claimed"]) for c in _clauses(s))
            u_x, u_phi, u_s = uncertainty_scores(s)

            # v(q) = value_at_quality_0 + span*q; dropping the intercept here while
            # the oracle's true_value keeps it makes every distortion look negative.
            claim_val = v0 + float(sig["claimed_quality"]) * span - disclosed_impact(sig)
            true_val = float(t.oracle["true_value"])
            ideal = max(0.0, (claim_val - true_val) / max(span, 1e-9))
            mkt = float(t.oracle.get("market_price") or 0.0)
            rows.append({
                "task": tid, "attack": a,
                "u_x": u_x, "u_phi": u_phi, "u_s": u_s,
                "n_clause": len(_clauses(s)),
                "disclosed": int(bool(sig.get("disclosed_attributes"))),
                "ask_vs_mkt": (float(sig["headline_price"]) / mkt) if mkt > 0 else 1.0,
                "claim_q": float(sig["claimed_quality"]),
                "canon": max(0.0, pen - nom),
                "res": float(t.oracle["reservation_utility"]),
                "L": max(float(t.listing_price), 1e-9),
                "ideal": ideal,
                "u_true": true_val - true_expected_price(sig),
            })
    return np.vstack(curves), pd.DataFrame(rows)


def robust_bound(C: np.ndarray, M: pd.DataFrame, eps) -> np.ndarray:
    """Worst-case utility at radius ``eps``, interpolated along the precomputed curve."""
    eps = np.clip(np.asarray(eps, float), GRID[0], GRID[-1])
    i = np.searchsorted(GRID, eps).clip(1, len(GRID) - 1)
    g0, g1 = GRID[i - 1], GRID[i]
    w = np.where(g1 > g0, (eps - g0) / np.maximum(g1 - g0, 1e-12), 0.0)
    r = np.arange(len(M))
    return C[r, i - 1] * (1 - w) + C[r, i] * w - M.canon.values


def decisions(C: np.ndarray, M: pd.DataFrame, eps) -> Tuple[np.ndarray, np.ndarray]:
    """Per-message (harmful acceptance, valid acceptance) indicators at radius ``eps``."""
    acc = (robust_bound(C, M, eps) >= M.res.values).astype(int)
    good = (M.u_true.values >= M.res.values).astype(int)
    return acc * (1 - good), acc * good
