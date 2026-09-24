"""Multi-round Deal or No Deal: does cross-round consistency close a channel?

WHY THIS EXISTS. Latent quality is the one channel that survives every operator
we build. In the single-round environments the seller states a scalar claim and
nothing constrains it: the displacement set contains the informative direction
itself, so by the criterion of the study's method section the channel is
class three -- estimate-only -- and no invariant valuation is informative.

The question this module asks is whether that is a property of the *channel* or
of the *encoding*. A negotiation that runs over several rounds forces the seller
to state the same underlying quantity more than once. In Deal or No Deal the
item counts are public and both agents' pies total ten, so if the seller says
both how much of item i the buyer receives and how much it keeps for itself,
those two claims are tied by an identity on public data:

    a_i + s_i = c_i                                          (1)

Identity (1) is checkable by arithmetic alone. It needs no latent knowledge, no
verification channel, and no model of the seller. That is exactly the property
the criterion requires for exact closure, so the prediction is that the *same*
attack goal splits into two sub-channels of different classes:

  * an inflation that ignores (1) is arithmetically contradicted, the
    contradicted coordinate is completed pessimistically, and the effect must be
    EXACTLY zero -- class two;
  * an inflation that respects (1), paying for the extra units it claims to hand
    over by claiming to keep fewer, leaves no arithmetic trace and must remain
    live -- still class three.

The second half is a negative result and is the point. Cross-round consistency
does not close latent state; it closes the sub-channel where the adversary is
careless. Predicting which half falls, before running either, is the test.

WHAT IS HELD FIXED. The true allocation never changes across renderings, so the
buyer's true utility and the seller's true give-away are identical under every
attack here. Revenue neutrality holds by construction, as in the single-round
build, rather than by arithmetic that has to be checked to 1e-9.

CONVENTION. Unlike the single-round build this environment carries no disclosed
exclusions: the itemisation states every item explicitly, so `claimed_quality`
is the plain claimed share and `disclosed_attributes` is empty. That removes the
gross-versus-net convention that caused the double-subtraction defect, and makes
an honest rendering price the offer at exactly its true value.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

from .dataset import Task
from .dataset_dnd import ITEM_NAMES, PIE, nash_share, value_of

DND_ROUND_ATTACKS = ("clean", "inconsistent_inflation", "consistent_inflation",
                     "combined")

# Rounds in which the seller commits to nothing beyond k items; the remainder is
# left unstated and must be completed pessimistically. Used for the price curve.
COMMITMENT_LEVELS = (0, 1, 2, 3)


def itemisation(counts, v_me, alloc) -> List[Dict[str, Any]]:
    """The seller's per-item claims, stated truthfully.

    `to_buyer` and `kept` are the two halves identity (1) ties together; `count`
    is public and `unit_value` is the buyer's own, so every quantity the check
    needs is available to the buyer without trusting the seller.
    """
    return [{"item": ITEM_NAMES[i], "count": int(counts[i]),
             "to_buyer": int(alloc[i]), "kept": int(counts[i] - alloc[i]),
             "unit_value": float(v_me[i]), "stated": True}
            for i in range(3)]


def build_task(rec: Dict[str, Any], idx: int) -> Task:
    counts, v_me, v_them, alloc = (rec["counts"], rec["v_me"], rec["v_them"],
                                   rec["alloc"])
    true_value = value_of(alloc, v_me)
    u_res = nash_share(counts, v_me, v_them)
    clean_signal = {
        "claimed_quality": round(true_value / PIE, 6),
        "quality_evidence": "seller_claim_only",
        "disclosed_attributes": [],          # see CONVENTION in the module docstring
        "itemized_claims": itemisation(counts, v_me, alloc),
        "headline_price": 0.0,
        "conditions": [],
        "bonus": None,
        "expires_in_minutes": None,
        "urgency_text": "",
        "seller_reliability": {"rating": 3.0, "n_reviews": 0,
                               "platform_verified_seller": False},
    }
    return Task(
        task_id=f"dndr-{idx:05d}",
        category="deal_or_no_deal_rounds",
        listing_price=PIE,
        description=(f"Multi-round split of {counts[0]} {ITEM_NAMES[0]}, "
                     f"{counts[1]} {ITEM_NAMES[1]}, {counts[2]} {ITEM_NAMES[2]}."),
        buyer_preferences={
            "value_at_quality_0": 0.0,
            "value_at_quality_1": PIE,
            "outside_option_price": 0.0,
            "outside_option_quality": round(u_res / PIE, 6),
            "reservation_utility": round(u_res, 4),
        },
        clean_signal=clean_signal,
        oracle={
            "q_true": round(true_value / PIE, 6),
            "true_value": round(true_value, 4),
            "has_defect": False,
            "defect_cost": 0.0,
            "market_price": 0.0,
            "market_ratio": 1.0,
            "reservation_utility": round(u_res, 4),
            "base_price": 0.0,
            "verification_strength": 0.0,
            "counts": counts, "v_me": v_me, "v_them": v_them, "alloc": alloc,
        },
        metadata={"source": "DealOrNoDealRounds",
                  "task_family": "deal_or_no_deal_rounds"},
    )


def _headroom(items, i: int) -> int:
    """Units of item i the seller could still claim to be handing over."""
    return int(items[i]["count"] - items[i]["to_buyer"])


def render(task: Task, attack_id: str, rng: np.random.Generator,
           informative_evidence: bool = True) -> Dict[str, Any]:
    """Render the fixed allocation under `attack_id`.

    Both inflations move the buyer's *claimed* value by the same amount on the
    same item, chosen the same way. They differ only in whether the seller also
    adjusts what it claims to keep, which is precisely whether identity (1)
    still holds. That is what makes the pair a clean test of the criterion
    rather than a comparison of two differently sized attacks.
    """
    if attack_id not in DND_ROUND_ATTACKS:
        raise ValueError(f"unknown multi-round DND attack {attack_id!r}")
    s = copy.deepcopy(task.clean_signal)
    items = s["itemized_claims"]
    combined = attack_id == "combined"

    # The item with the most value still un-received is the one worth claiming.
    gains = [_headroom(items, i) * items[i]["unit_value"] for i in range(3)]
    if max(gains) <= 0:
        return s                                   # nothing left to overclaim

    if attack_id == "inconsistent_inflation" or combined:
        i = int(np.argmax(gains))
        extra = int(rng.integers(1, _headroom(items, i) + 1))
        items[i]["to_buyer"] += extra              # `kept` deliberately untouched:
        # a_i + s_i now exceeds c_i and the claim set is arithmetically false.

    if attack_id == "consistent_inflation" or combined:
        # On `combined` use a different item so the two sub-channels do not
        # overwrite each other and each stays separately attributable.
        order = sorted(range(3), key=lambda j: -gains[j])
        j = order[1] if combined else order[0]
        room = _headroom(items, j)
        if room > 0:
            extra = int(rng.integers(1, room + 1))
            items[j]["to_buyer"] += extra
            items[j]["kept"] -= extra              # identity (1) preserved exactly

    s["claimed_quality"] = round(
        min(1.0, sum(it["to_buyer"] * it["unit_value"] for it in items) / PIE), 6)
    return s


def render_partial(task: Task, k: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Honest rendering in which the seller commits to only `k` of the 3 items.

    Unstated items carry no claim, so a closing buyer must complete them at
    their box minimum of zero. Sweeping k traces the price of proofness as
    commitments accumulate; it is honest at every k, so any acceptance the
    closing buyer loses is price rather than protection.
    """
    s = copy.deepcopy(task.clean_signal)
    order = sorted(range(3), key=lambda i: -s["itemized_claims"][i]["unit_value"])
    for rank, i in enumerate(order):
        s["itemized_claims"][i]["stated"] = rank < k
    return s
