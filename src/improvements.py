"""Genuine utility improvements: the sensitivity half of selective invariance.

WHY THIS EXISTS. Every rendering in `attacks.py` is revenue-neutral by
construction, so the whole benchmark measures one half of what a decision rule
must do: not move when the seller changes presentation without changing the
economics. That half is easy to pass by being insensitive. A rule that rejects
everything scores a perfect zero on every attack channel.

The question the method is actually answering is two-sided --- be invariant to
utility-preserving strategic variation, and stay responsive to real utility
variation --- and only the first side was ever tested against a paired
counterfactual. These renderings supply the second. They are honest: the seller
genuinely lowers what it will settle for, so the buyer's true expected utility
rises by exactly the amount conceded and `effective_price` falls with it. There
is nothing to detect and nothing to close.

WHAT THE CRITERION PREDICTS. The closure operators act on the completion set and
on payoff-irrelevant fields; neither touches the headline the seller is actually
conceding. So a closing rule must track a genuine concession about as well as
the oracle does, while moving by exactly zero on the revenue-neutral channels.
The contrast between those two numbers, measured on the same arms, is what
distinguishes selective invariance from plain insensitivity. A closing rule that
tracked improvements much worse than the oracle would be buying its invariance
with dead weight, which is the failure mode the harmful/valid pair was always
meant to expose but could not, because no rendering ever improved anything.

MAGNITUDES. Concessions are a fixed fraction of the listing price rather than of
the offer, so the same nominal generosity is applied regardless of how
aggressively the seller priced, and the two levels straddle the jitter width the
offers were drawn with.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

# Fractions of the listing price the seller genuinely gives up.
IMPROVEMENTS = {"genuine_discount_small": 0.05, "genuine_discount_large": 0.15}


def render(task, improvement_id: str, rng: np.random.Generator,
           informative_evidence: bool = True) -> Dict[str, Any]:
    """Honest rendering of a genuinely better deal.

    Only `headline_price` moves, so `effective_price` and therefore the true
    expected utility move with it by the full amount. Nothing else in the signal
    changes, which keeps the comparison paired at the message level with the
    clean rendering the attacks are also paired against.
    """
    if improvement_id == "clean":
        return copy.deepcopy(task.clean_signal)
    if improvement_id not in IMPROVEMENTS:
        raise ValueError(f"unknown improvement {improvement_id!r}")
    s = copy.deepcopy(task.clean_signal)
    concession = IMPROVEMENTS[improvement_id] * float(task.listing_price)
    s["headline_price"] = round(float(s["headline_price"]) - concession, 2)
    return s


def concession_of(task, improvement_id: str) -> float:
    """True money the seller gives up, for checking that utility really moved."""
    if improvement_id == "clean":
        return 0.0
    return IMPROVEMENTS[improvement_id] * float(task.listing_price)
