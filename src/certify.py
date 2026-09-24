"""Finite-sample certification of the gate's decision risk.

WHY THIS EXISTS. The safety property of Sec. 4.7 is conditional on assumption
A1 -- the true state lies inside the ambiguity set -- and the coverage audit
shows A1 failing on a quarter of messages. Split conformal repairs A1 by
calibrating the radius so that

    P( true state inside the ambiguity set ) >= 1 - alpha,

which is what conformal uncertainty sets for robust optimization have done since
Johnstone & Cox (2021). Two things go wrong when that recipe is transplanted
into a negotiation, and this module is about both.

FIRST: coverage is the wrong target. What a buyer needs bounded is the rate of
*harmful acceptances*, and covering the state is sufficient but not necessary
for a correct decision -- a message whose bound sits far below the reservation
utility is rejected whether or not the ambiguity set happens to contain the
truth. Certifying coverage therefore buys safety the buyer already had and pays
for it in foregone trade. We certify the decision risk directly, by
Learn-then-Test rather than by a conformal quantile.

SECOND, and this is the part no calibration recipe survives on its own: split
conformal certifies the deployment risk only under exchangeability between the
calibration and deployment message distributions. A seller that best-responds to
the *published* radius rule changes the deployment distribution as a function of
the certificate itself. This is not a mild violation. Measured across three
environments, the realized risk under a fitted best response is about three
times the risk on the calibration mixture, and at the levels a deployment would
actually ask for, the certificate is violated outright.

THE REPAIR. Take the per-task maximum over the seller's action set *before*
aggregating across tasks. A mixture never exceeds the maximum of its own
support, so the resulting risk dominates that of every seller policy over that
action set -- adaptive, randomized, or otherwise. The certificate then holds
uniformly over the seller's policy space, and needs no fixed point, no model of
what the seller optimizes, and no assumption that the seller is even rational.
Its price is paid only where it is needed: where no policy can exploit the gate,
the worst-case and mixture certificates select the same radius and the bound is
free.

WHAT THIS IS NOT. The two certificates draw radii from one parametric family, so
they trace the same safety--utility frontier; the worst-case one is not a
sharper mechanism, it is a *valid* one. The comparison below is between a
certificate that holds and a certificate that does not, not between two points
on a frontier.

TESTING PROCEDURE. The risk is monotone non-increasing in the radius offset, so
offsets are tested in increasing order and the first rejection is returned.
Fixed-sequence testing controls the family-wise error with no multiplicity
penalty; an earlier Bonferroni version of this search cost so much power that
the smallest environment could not certify at any level. Risks are aggregated to
tasks before the Hoeffding bound is applied, because the seven renderings of a
task are one listing and the message count is not the effective sample size.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Offsets searched, in the units the radius lives in (fractions of the latent
# quality range). The scan is monotone, so the cost is one pass, not a grid.
OFFSET_GRID = np.round(np.arange(0.0, 0.8001, 0.005), 4)


def task_risk(tasks: np.ndarray, harm: np.ndarray, uniform: bool) -> np.ndarray:
    """Per-task risk, aggregated over the renderings of that task.

    ``uniform=False`` averages, giving the risk under the attack mixture the
    calibration split happens to contain. ``uniform=True`` maximizes, giving a
    quantity that dominates the risk of every seller policy supported on those
    renderings -- which is what makes the certificate policy-uniform.
    """
    s = pd.Series(np.asarray(harm, float)).groupby(np.asarray(tasks))
    return (s.max() if uniform else s.mean()).values


def hoeffding_p(risk: float, alpha: float, n: int) -> float:
    """p-value for H0: risk >= alpha, from Hoeffding on [0,1]-bounded task risks."""
    if risk >= alpha:
        return 1.0
    return float(np.exp(-2.0 * n * (alpha - risk) ** 2))


def certify(harm_at: Callable[[float], np.ndarray], tasks: np.ndarray,
            alpha: float, uniform: bool, delta: float = 0.10,
            grid: Sequence[float] = OFFSET_GRID) -> Optional[float]:
    """Smallest offset whose risk certificate holds at level alpha, or None.

    ``harm_at(t)`` returns the per-message harmful-acceptance indicator at
    radius offset ``t``. None means no offset in the grid certifies: with these
    calibration sizes that happens at tight alpha, and it is a real answer --
    a finite-sample guarantee has a task-count price -- not a failure to
    converge.
    """
    n = int(pd.Series(np.asarray(tasks)).nunique())
    for t in grid:
        r = float(task_risk(tasks, harm_at(float(t)), uniform).mean())
        if hoeffding_p(r, alpha, n) <= delta:
            return float(t)
    return None


def best_response(harm: np.ndarray, evidence: np.ndarray, attack: np.ndarray) -> Dict[str, str]:
    """Seller policy: per observable evidence level, the most harmful family.

    Conditioning on evidence rather than on the task keeps the policy
    deployable -- the seller knows what it can prove about itself, not which
    buyer it drew. The per-task maximum is the non-deployable upper bound, and
    it is exactly what ``uniform=True`` certifies against.
    """
    g = (pd.DataFrame({"ev": evidence, "a": attack, "h": np.asarray(harm, float)})
         .groupby(["ev", "a"]).h.mean().reset_index())
    return g.sort_values("h", ascending=False).groupby("ev").first().a.to_dict()


def policy_mask(policy: Dict[str, str], evidence: np.ndarray,
                attack: np.ndarray) -> np.ndarray:
    """Messages the seller's policy actually sends."""
    return np.array([policy.get(e) == a for e, a in zip(evidence, attack)], dtype=bool)
