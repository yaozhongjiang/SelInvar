"""Attackability-gated ambiguity radius.

WHAT IT CHANGES. Every calibrated-radius method we build on estimates *how much*
to distrust a message and widens the ambiguity set in proportion: the radius is a
conformal region around the state \\cite{johnstone2021conformal}, a set reshaped
for decision loss, or a learned contextual width. On this benchmark that target
is not learnable to the precision the decision needs -- a gradient-boosted
predictor of the realized distortion reaches correlation 0.66/0.58/0.59 on the
three environments, more capacity makes it worse, and even handing the learner
the attack family identity tops out near 0.75, while the frontier does not move
until roughly 0.93.

WHY THE TARGET IS WRONG. Let A be the seller's action set and let

    H_j(eps) = max_{a in A} 1[ gate accepts (j,a) at radius eps  and  deal j is bad ]

be the policy-uniform harm indicator on task j. H_j is non-increasing in eps and
binary, so it is a step: H_j(eps) = 1[eps < rho_j] for

    rho_j = inf{ eps : H_j(eps) = 0 },

the smallest radius that defeats *every* seller action on that task. The
policy-uniform risk of a per-task rule eps(.) is then exactly

    R(eps) = (1/n) sum_j 1[ eps_j < rho_j ].                                (1)

Equation (1) says a task with rho_j = 0 contributes zero risk at *any* radius,
including zero. The entire risk budget is carried by S = {j : rho_j > 0}, which
is 13.4% / 7.8% / 26.9% of tasks here. A scalar radius must clear the budget
using one value for every task, so it pays utility on the complement of S where
no payment buys anything -- and the utility it pays there is not small, because
raising the radius also rejects honest good deals.

WHAT IS LEARNABLE. The magnitude of rho is not much easier to predict than the
distortion was (correlation 0.46/0.09/0.81). Its *support* is: whether a task is
attackable at all is predicted at AUC 0.934/0.890/0.995 from the same
observables. So the decision-relevant latent is binary, and it is the one that
(1) actually gates the risk.

THE RULE. With g(m) an estimate of P(rho_j > 0 | m),

    eps_j = eps_hi * 1[ g(m_j) >= tau ],                                    (2)

whose risk decomposes, by (1), into a false-negative term and a residual term:

    R(tau, eps_hi) = P(rho > 0, g < tau)  +  P(rho > eps_hi, g >= tau).     (3)

The first is the classifier missing an attackable task; the second is a flagged
task the radius did not cover. R is non-increasing in eps_hi at fixed tau and
non-decreasing in tau at fixed eps_hi, so it is non-increasing along any path
that lowers tau and raises eps_hi. Certifying along such a path by
fixed-sequence Learn-then-Test needs no multiplicity correction.

THREE THINGS VALIDITY DEPENDS ON, one of which we got wrong first.

  * The max over A in H_j is taken *before* aggregating over tasks, so the
    certificate bounds the risk of every seller policy on A, adaptive or not.
    That construction is \\citet{csillag2024strategic}'s, transferred here from a
    conformity threshold to the radius of a robust program.
  * Risks are aggregated to tasks before the concentration bound: renderings of
    one listing are not independent observations.
  * The classifier must not be fitted on the data used to certify. Our first
    version scored the calibration split with a classifier fitted on it and
    certified alpha = 0.10 while delivering 0.1056 on test. Cross-fitting gives
    every calibration task an out-of-fold score, which restores validity without
    halving the sample -- the split version could not certify the smallest
    environment at all.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.certify import hoeffding_p, task_risk

OFFSET_GRID = np.round(np.arange(0.0, 0.6001, 0.01), 4)
TAU_GRID = np.round(np.arange(0.05, 0.9501, 0.05), 3)


def safe_radius(harm_at: Callable[[float], np.ndarray], tasks: np.ndarray,
                grid: Sequence[float]) -> Dict[str, float]:
    """rho_j for each task: the smallest grid radius with no harmful acceptance.

    ``harm_at(eps)`` returns the per-message harmful-acceptance indicator at a
    constant radius; the maximum over a task's messages is the policy-uniform
    indicator of Eq. (1). A task never made safe on the grid is assigned the
    largest radius on it, which is conservative in the direction that matters.
    """
    tasks = np.asarray(tasks)
    stack = np.column_stack([harm_at(float(e)) for e in grid]).astype(bool)
    out: Dict[str, float] = {}
    for t, idx in pd.Series(np.arange(len(tasks))).groupby(tasks).groups.items():
        any_harm = stack[np.asarray(idx)].any(axis=0)
        safe = ~any_harm
        out[t] = float(grid[int(np.argmax(safe))]) if safe.any() else float(grid[-1])
    return out


def crossfit_scores(X: pd.DataFrame, y: np.ndarray, make_model: Callable,
                    folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(rho>0 | m), so no task is scored by a model that saw it.

    Split conformal would also work but costs half the calibration set, which on
    the smallest environment leaves too few tasks to certify anything.
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    assign = rng.permutation(np.arange(n) % folds)
    out = np.zeros(n, dtype=float)
    for k in range(folds):
        tr, te = assign != k, assign == k
        if y[tr].min() == y[tr].max():          # degenerate fold: fall back to base rate
            out[te] = float(y[tr].mean())
            continue
        out[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def certify_gate(risk_of: Callable[[Optional[float], float], float],
                 utility_of: Callable[[Optional[float], float], float],
                 n_tasks: int, alpha: float, delta: float = 0.10,
                 taus: Sequence[float] = TAU_GRID,
                 offsets: Sequence[float] = OFFSET_GRID,
                 include_scalar: bool = True
                 ) -> Optional[Tuple[Optional[float], float]]:
    """First configuration on the utility-ordered path whose certificate holds.

    Returns ``(tau, eps_hi)`` with ``tau=None`` denoting the scalar rule, or None
    if nothing on the grid certifies at this sample size -- which is a real
    answer, not a failure to converge, and must not be replaced by the widest
    radius on the grid.
    """
    cands = [(None, float(e)) for e in offsets] if include_scalar else []
    cands += [(float(t), float(e)) for t in taus for e in offsets]
    cands.sort(key=lambda c: -utility_of(*c))
    for tau, eps in cands:
        if hoeffding_p(risk_of(tau, eps), alpha, n_tasks) <= delta:
            return tau, eps
    return None


def apply_gate(scores: np.ndarray, tau: Optional[float], eps_hi: float) -> np.ndarray:
    """Eq. (2); ``tau=None`` gives the scalar rule the gate is compared against."""
    if tau is None:
        return np.full(len(scores), eps_hi, dtype=float)
    return np.where(np.asarray(scores) >= tau, eps_hi, 0.0)
