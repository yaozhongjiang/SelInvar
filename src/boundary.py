"""The failing side of Proposition 1's precondition, made measurable.

WHY THIS EXISTS. Proposition 1 is a characterisation: closure is exact when the
attacked completion set contains the clean one, and there is a price function and
a threshold under which it fails when that inclusion breaks. Every attack family
in `attacks.py` only ever *adds* unresolved conditions, so all of them satisfy
the precondition and none of them can exhibit the failing side. A reader is then
asked to take the necessity half on the proof alone while the sufficiency half
is measured, which is the wrong way round for the claim the study leans on.

WHAT THIS RENDERS. A contract manipulation that makes the two completion sets
*incomparable* rather than nested: it removes one condition the clean rendering
really carried and adds a different one of matched expected cost. Neither set
contains the other, the seller's true expected settlement is unchanged to the
same tolerance as the calibrated families, and the closure's guarantee therefore
does not apply. The prediction is that the contract-channel effect under closure
stops being exactly zero.

WHY THE CLEAN SIDE NEEDS CONDITIONS. An earlier version of this experiment
removed clauses from the attacked rendering while the clean rendering carried
none, so the attacked set was still a superset of the empty set and the
precondition held after all -- the run measured nothing. The clean rendering here
is given two real conditions first, so that removing one is possible.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

BOUNDARY_ATTACKS = ("clean", "nested_addition", "incomparable_swap")


def _condition(kind: str, amount: float, p_true: float,
               p_claimed: float | None = None) -> Dict[str, Any]:
    """A clause carries what the seller claims and what actually binds.

    Honest renderings state the two equal; the contract channel is exactly the
    gap between them, and closure works by taking the worst case over the set
    rather than by trusting `p_claimed`.
    """
    # The amount is kept at full precision: rounding it while the expected-cost
    # match was computed from the unrounded probability broke revenue neutrality
    # by up to 5e-2 on high-priced listings.
    return {"kind": kind, "amount": float(amount),
            "p_true": round(float(p_true), 6),
            "p_claimed": round(float(p_true if p_claimed is None else p_claimed), 6)}


def seed_clean(task, rng: np.random.Generator) -> Dict[str, Any]:
    """Clean rendering carrying two genuine conditions.

    Without these the attacked set cannot fail to contain the clean one, and the
    experiment silently degenerates into the nested case it is meant to contrast.
    """
    s = copy.deepcopy(task.clean_signal)
    L = float(task.listing_price)
    s["conditions"] = [
        _condition("restocking_fee", 0.04 * L, float(rng.uniform(0.30, 0.50))),
        _condition("delivery_surcharge", 0.03 * L, float(rng.uniform(0.30, 0.50))),
    ]
    return s


def render(task, attack_id: str, rng: np.random.Generator,
           informative_evidence: bool = True) -> Dict[str, Any]:
    if attack_id not in BOUNDARY_ATTACKS:
        raise ValueError(f"unknown boundary attack {attack_id!r}")
    s = seed_clean(task, rng)
    if attack_id == "clean":
        return s
    L = float(task.listing_price)
    conds: List[Dict[str, Any]] = list(s["conditions"])

    if attack_id == "nested_addition":
        # The calibrated regime: only adds, so the attacked set contains the
        # clean one and Proposition 1 applies. Included as the control.
        conds.append(_condition("handling_fee", 0.03 * L,
                                float(rng.uniform(0.30, 0.50))))
    else:
        # Incomparable: drop one real condition and add a different one of
        # matched *expected* cost but a higher binding probability, hence a
        # smaller amount. Expected settlement is unchanged, so the rendering is
        # revenue-neutral exactly as the calibrated families are; but the worst
        # case over the completion set falls, because the sup is over amounts
        # rather than over expectations. Drawing the new probability at random
        # instead leaves the direction to chance and the effect averages away --
        # a first version did that and measured exactly zero.
        dropped = conds.pop(0)
        cost = dropped["amount"] * dropped["p_true"]
        p_new = round(min(0.95, dropped["p_true"] * float(rng.uniform(1.6, 2.2))), 6)
        conds.append(_condition("expedite_fee", cost / p_new, p_new))
    s["conditions"] = conds
    return s


def expected_settlement(signal: Dict[str, Any]) -> float:
    """True expected price implied by the rendered contract, for the neutrality check."""
    p = float(signal["headline_price"])
    for c in signal.get("conditions") or []:
        p += float(c["amount"]) * float(c["p_true"])
    b = signal.get("bonus")
    if b:
        p -= float(b["amount"]) * float(b["p_true"])
    return p
