"""A simulated contract channel for Deal or No Deal.

WHY. The contract result is measured on CraigslistBargain and AmazonHistoryPrice,
both of which *construct* the latent quality, so a reader may reasonably ask
whether exact contract closure is an artefact of that construction rather than a
property of the operator. Deal or No Deal records both agents' true per-item
values, so its oracle is a dataset fact; if the same closure appears there, the
result does not depend on a sampled label.

THE CONSTRUCTION. Deal or No Deal has no price, so a contract clause is built
from the allocation itself: the seller withholds value `delta` from the base
split and promises it back contingently, quoting probability `p_claimed` while
the true probability is `p_true`. The clause amount is set to `delta / p_true`,
so the buyer's TRUE expected utility is unchanged to floating-point tolerance --
the same revenue-neutrality the other environments enforce -- while a buyer that
prices the clause at its claimed probability sees `delta * (p_claimed/p_true)`
instead of `delta` and overvalues the offer.

The clause is written into the shared `conditions` field, so the conservative
price rule and the gate act on it unmodified; nothing environment-specific is
added to any policy.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

PIE = 10.0


def render_contract(task, attack_id: str, rng: np.random.Generator) -> Dict[str, Any]:
    """Clean signal, or one with a revenue-neutral contingent clause."""
    s = dict(task.clean_signal)
    s["disclosed_attributes"] = list(s.get("disclosed_attributes") or [])
    s["conditions"] = []
    if attack_id == "clean":
        return s
    if attack_id not in ("contract_complexity", "tail_incentive"):
        raise ValueError(f"unknown contract attack {attack_id!r}")

    true_value = float(task.oracle["true_value"])
    delta = float(min(0.30 * true_value, 3.0))
    if delta <= 0:
        return s
    # Both clauses must OVERSTATE the odds to be attacks. An understated clause
    # makes a claimed-probability buyer more conservative, not less, so quoting
    # p_claimed < p_true would measure the machinery's direction rather than a
    # manipulation: contract_complexity overstates mildly on a likely event,
    # tail_incentive overstates heavily on an unlikely one.
    p_true = 0.40 if attack_id == "contract_complexity" else 0.05
    p_claimed = 0.75 if attack_id == "contract_complexity" else 0.60

    # withhold delta from the headline claim, promise it back contingently
    gross = float(s["claimed_quality"]) * PIE
    s["claimed_quality"] = round(max(0.0, (gross - delta) / PIE), 6)
    s["conditions"] = [{
        "kind": "contingent_topup",
        "amount": round(-delta / p_true, 6),   # negative: a benefit, not a charge
        "p_true": p_true,
        "p_claimed": p_claimed,
        "claimed": True,
        "verifiable": False,
    }]
    return s


def true_utility(task, signal: Dict[str, Any]) -> float:
    """Buyer's true expected utility: the clause pays at its TRUE probability."""
    v = float(signal["claimed_quality"]) * PIE
    v -= sum(float(a.get("value_impact", 0.0))
             for a in (signal.get("disclosed_attributes") or []))
    for c in (signal.get("conditions") or []):
        v -= float(c["amount"]) * float(c["p_true"])
    return v
