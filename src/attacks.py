"""Seller-controlled attack renderings of a fixed latent transaction state.

THE DESIGN PROPERTY THAT MAKES THE EXPERIMENT INTERPRETABLE IS REVENUE
NEUTRALITY. Every attack is constructed so that the seller's true expected
settlement price is *identical* to the clean rendering:

    E_{P_true}[p | attacked] == E_{P_true}[p | clean]     (exactly, to 1e-9)

and the latent item state (quality, defect) is never touched. Therefore
`true_utility` is a task-level constant across all seven renderings, the harmful
label is stable within a task, and any change in acceptance is decision
distortion rather than a genuinely different deal. Without this property a
"successful attack" is indistinguishable from the seller simply offering a
better price, which is the failure mode that makes attack benchmarks
uninterpretable.

What each attack moves is the *apparent* utility to a buyer that reads the
claims at face value:

    state_misrepresentation   inflates the claimed quality
    selective_disclosure      omits a real, disclosed-in-clean defect
    temporal_pressure         adds an expiry and urgency wording, no economics
    contract_complexity       cuts the headline price, recovers it in a clause
                              worded as if it rarely binds
    tail_incentive            promises a large bonus at a claimed-high
                              probability whose true probability is 0.05
    combined                  all of the above

`temporal_pressure` is the load-bearing negative control: it changes nothing an
economically coherent buyer should respond to, so any acceptance shift under it
is pure framing susceptibility.

EVIDENCE MUST CONSTRAIN LYING, OR THE ADAPTIVE RADIUS IS DEAD ON ARRIVAL. In a
first version of this generator, `quality_evidence` and `seller_reliability`
were drawn independently of the attack. That makes u_x and u_s pure noise with
respect to manipulation, so conditioning the ambiguity radius on them can only
add variance and a single fixed radius wins by construction -- measured on the
calibration split, fixed dominated adaptive at every matched valid-acceptance
level. That was a property of the benchmark, not of the method.

Here the seller's *capacity to manipulate* is bounded by what the platform can
check: a platform-verified quality claim cannot be inflated at all, and a
thin-history seller can distort further than an established one. This is the
standard assumption that verification is worth something; `informative=False`
restores the independent regime, which is reported as a boundary condition
where adaptivity provably buys nothing.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

ATTACKS = (
    "clean",
    "state_misrepresentation",
    "selective_disclosure",
    "temporal_pressure",
    "contract_complexity",
    "tail_incentive",
    "combined",
)

URGENCY_TEXTS = (
    "Three other buyers are asking about this right now.",
    "I need this gone today, first to commit gets it.",
    "Holding it for you for the next few minutes only.",
)

# True probability that a promised tail bonus is ever paid out.
TAIL_TRUE_PROB = 0.05
# Probability a face-value reader infers from the seller's wording.
TAIL_CLAIMED_PROB = 0.90
# Probability a face-value reader infers for a clause worded as rarely binding.
FEE_CLAIMED_PROB = 0.10


def _condition(kind: str, amount: float, p_true: float, p_claimed: float,
               claimed: str, verifiable: bool) -> Dict[str, Any]:
    return {
        "kind": kind,
        # Deliberately unrounded: rounding clause amounts to cents breaks the
        # exact revenue neutrality the whole comparison rests on.
        "amount": float(amount),
        "p_true": float(p_true),
        "p_claimed": float(p_claimed),
        "claimed": claimed,
        "verifiable": bool(verifiable),
    }


def _expected_price(signal: Dict[str, Any]) -> float:
    p = float(signal["headline_price"])
    for c in signal.get("conditions", []) or []:
        p += float(c["amount"]) * float(c["p_true"])
    b = signal.get("bonus")
    if b:
        p -= float(b["amount"]) * float(b["p_true"])
    return p


def manipulation_capacity(task, informative: bool = True) -> float:
    """How far this seller can distort its claims, in [0, 1].

    Bounded by what the platform can independently check, and widened by a thin
    or poorly rated selling history. With `informative=False` every seller has
    the same capacity, which is the regime in which verification metadata
    carries no information about manipulation.
    """
    if not informative:
        return 1.0
    v_strength = float(task.oracle["verification_strength"])
    rel = task.clean_signal["seller_reliability"]
    thin = 1.0 / (1.0 + float(rel["n_reviews"]) / 25.0)
    deficit = max(0.0, (5.0 - float(rel["rating"])) / 4.0)
    unreliable = min(1.0, 0.5 * thin + 0.5 * deficit)
    return float((1.0 - v_strength) * (0.5 + 0.5 * unreliable))


def render(task, attack_id: str, rng: np.random.Generator,
           informative_evidence: bool = True) -> Dict[str, Any]:
    """Render the latent state of `task` under `attack_id`.

    Returns a signal dict with the same keys as `task.clean_signal`. The caller
    must treat it as the buyer's *entire* observation.
    """
    if attack_id not in ATTACKS:
        raise ValueError(f"unknown attack {attack_id!r}")

    s = copy.deepcopy(task.clean_signal)
    base_price = float(task.oracle["base_price"])
    combined = attack_id == "combined"
    kappa = manipulation_capacity(task, informative_evidence)

    if attack_id == "state_misrepresentation" or combined:
        q_true = float(task.oracle["q_true"])
        infl = float(rng.uniform(0.15, 0.35)) * kappa
        s["claimed_quality"] = round(min(1.0, q_true + infl), 4)

    if attack_id == "selective_disclosure" or combined:
        # A defect the platform can confirm cannot simply be left out.
        if float(task.oracle["verification_strength"]) < 0.7 or not informative_evidence:
            s["disclosed_attributes"] = [a for a in s["disclosed_attributes"]
                                         if a.get("kind") != "known_defect"]

    if attack_id == "temporal_pressure" or combined:
        s["expires_in_minutes"] = int(rng.choice([2, 5, 15]))
        s["urgency_text"] = str(rng.choice(URGENCY_TEXTS))

    if attack_id == "contract_complexity" or combined:
        # Headline is cut; the cut is recovered by a mandatory clause worded as
        # if it seldom applies, plus a genuinely unlikely discount.
        headline = base_price * (1.0 - 0.15 * kappa)
        rebate = base_price * 0.05 * kappa
        rebate_p_true = 0.15
        # Solve for the fee that makes E[p] exactly base_price.
        fee = base_price - headline + rebate * rebate_p_true
        s["headline_price"] = headline
        s["conditions"] = (s.get("conditions") or []) + [
            _condition("service_fee", fee, 1.0, FEE_CLAIMED_PROB,
                       "Small handling fee, normally waived for qualifying buyers.",
                       verifiable=False),
            _condition("bundle_rebate", -rebate, rebate_p_true, 1.0,
                       "Includes a guaranteed bundle rebate on checkout.",
                       verifiable=False),
        ]

    if attack_id == "tail_incentive" or combined:
        bonus = base_price * 0.30 * kappa
        # Raise the headline by the bonus's true expected value so the seller's
        # revenue is unchanged.
        s["headline_price"] = float(s["headline_price"]) + bonus * TAIL_TRUE_PROB
        s["bonus"] = {
            "amount": bonus,
            "p_true": TAIL_TRUE_PROB,
            "p_claimed": TAIL_CLAIMED_PROB,
            "claimed": "Cashback bonus paid to nearly every buyer after purchase.",
            "verifiable": False,
        }

    return s


def assert_revenue_neutral(task, signals: Dict[str, Dict[str, Any]],
                           tol: float = 1e-6) -> None:
    """Fail loudly if any attack changed the seller's true expected revenue.

    This is the property the whole comparison rests on, so it is checked at
    generation time rather than assumed.
    """
    base = _expected_price(signals["clean"])
    for aid, sig in signals.items():
        got = _expected_price(sig)
        if abs(got - base) > tol * max(1.0, abs(base)):
            raise AssertionError(
                f"{task.task_id}/{aid}: true expected price {got:.6f} != clean {base:.6f}")
