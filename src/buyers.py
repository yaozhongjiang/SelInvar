"""Buyer decision policies.

Each policy is a distinct decision *mechanism*, so that the baseline matrix in
the study tests something. In the original scaffold `dro` and `robust_gate`
shared one code path and `conservative` and `oracle` shared another, which
reduced seven arms to two and made every comparison vacuous.

    oracle              sees the latent state; upper reference, not deployable
    claimed_eu          expected utility under the claims, taken at face value
    conservative        canonical contract, worst case over unresolved clauses
    verify_then_decide  hard evidence filter, then conservative arithmetic
    dro_fixed           Wasserstein DRO at one radius for every message
    robust_gate         ours: canonicalization + urgency decoupling +
                        verification-conditioned adaptive radius + margin

Only `robust_gate` adapts its radius to the message. `dro_fixed` is the
ablation that isolates that adaptation from the robust machinery itself.

A NOTE ON WHAT THE DRO ACTUALLY BUYS. With utility linear in a scalar latent
quality, a W1 ball reduces analytically to a downward shift of the claimed
quality, and the LP would be theatre. The ambiguity set here is therefore over
the *joint* support of (quality, which contract clauses bind), with ground
metric |q - q'| + kappa * hamming(omega, omega')/k. Utility is linear in q but
arbitrary in omega, so the worst case is a genuine transport problem and the LP
in `dro_solver` is doing real work. `check_dro.py` cross-checks it against the
analytic quality-only shift on contracts with no clauses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dro_solver import wasserstein_worst_case

# Cost, in the ground metric, of flipping every contract clause at once.
CLAUSE_FLIP_COST = 0.5
# Quality support. The endpoints 0 and 1 must always be present: with a grid
# that only spans q_claim +/- 0.6 the transport problem runs out of places to
# move mass, so the "worst case" saturates and the bound stops being a lower
# bound at large radii -- silently anti-conservative exactly where the gate is
# supposed to be most cautious.
QUALITY_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Maximum clauses carried into the joint support (2**k realizations).
MAX_CLAUSES = 3


@dataclass
class GateConfig:
    """Parameters of the adaptive robust gate (study Sec. 4.5-4.6)."""
    eps0: float = 0.05          # base radius
    lambda_x: float = 0.35      # weight on unverified state claims (u_x)
    lambda_phi: float = 0.30    # weight on unresolved contract mass (u_phi)
    lambda_s: float = 0.20      # weight on seller reliability uncertainty (u_s)
    delta: float = 0.0          # evaluation-error margin in Eq. 14
    fixed_radius: float = 0.15  # radius used by the dro_fixed ablation
    # Ablation switches; all True reproduces the full method.
    use_state_uncertainty: bool = True
    use_contract_canonicalization: bool = True
    use_urgency_decoupling: bool = True
    use_adaptive_radius: bool = True


# --------------------------------------------------------------------------
# reading a signal
# --------------------------------------------------------------------------

def value_fn(prefs: Dict[str, Any]):
    """The buyer's own (correctly specified) quality -> value mapping.

    Reads `dataset.QUALITY_CURVE` at call time so the buyer and the oracle stay
    the same function under the sensitivity study. Binding it at import would
    leave the buyer affine while the oracle curved, which is not a coherent
    utility model and would make the study measure a specification error.
    """
    from . import dataset as _ds
    v0 = float(prefs["value_at_quality_0"])
    v1 = float(prefs["value_at_quality_1"])
    return lambda q: v0 + (v1 - v0) * (float(q) ** _ds.QUALITY_CURVE)


def disclosed_impact(signal: Dict[str, Any]) -> float:
    """Total value reduction from defects the seller actually disclosed.

    An honest rendering discloses them; `selective_disclosure` omits them, and
    the buyer's value estimate is then too high by exactly this amount.
    """
    return float(sum(float(a.get("value_impact", 0.0))
                     for a in (signal.get("disclosed_attributes") or [])))


def claimed_value(signal: Dict[str, Any], prefs: Dict[str, Any]) -> float:
    """Value the buyer reads off the signal, net of disclosed defects."""
    return value_fn(prefs)(signal["claimed_quality"]) - disclosed_impact(signal)


def _clauses(signal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Contract clauses plus the tail bonus, in one uniform representation.

    A promised bonus is a clause with a negative amount: it reduces what the
    buyer pays, if it ever materialises.
    """
    out = list(signal.get("conditions") or [])
    b = signal.get("bonus")
    if b:
        out = out + [{
            "kind": "tail_bonus",
            "amount": -float(b["amount"]),
            "p_true": float(b["p_true"]),
            "p_claimed": float(b["p_claimed"]),
            "verifiable": bool(b.get("verifiable", False)),
        }]
    return out[:MAX_CLAUSES]


def apparent_price(signal: Dict[str, Any]) -> float:
    """Price a face-value reader infers, weighting clauses by claimed odds."""
    p = float(signal["headline_price"])
    for c in _clauses(signal):
        p += float(c["amount"]) * float(c["p_claimed"])
    return p


def safe_price(signal: Dict[str, Any]) -> float:
    """Conservative effective price: sup over unresolved clause realizations.

    Costs to the buyer are assumed to bind; benefits are credited only when
    they are immediately verifiable. This is `p_safe` of Sec. 4.3.
    """
    p = float(signal["headline_price"])
    for c in _clauses(signal):
        amt = float(c["amount"])
        if amt > 0:
            p += amt                      # a charge: assume it applies
        elif c.get("verifiable") and float(c["p_true"]) >= 1.0:
            p += amt                      # a guaranteed, checkable benefit
    return p


def uncertainty_scores(signal: Dict[str, Any]) -> Tuple[float, float, float]:
    """(u_x, u_phi, u_s) driving the adaptive radius of Sec. 4.5."""
    from .dataset import VERIFICATION_STRENGTH
    u_x = 1.0 - VERIFICATION_STRENGTH.get(signal.get("quality_evidence", "seller_claim_only"), 0.0)

    headline = max(abs(float(signal["headline_price"])), 1e-9)
    unresolved = sum(abs(float(c["amount"])) for c in _clauses(signal)
                     if not c.get("verifiable", False))
    u_phi = float(min(1.0, unresolved / headline))

    rel = signal.get("seller_reliability") or {}
    n = float(rel.get("n_reviews", 0))
    rating = float(rel.get("rating", 3.0))
    thin_history = 1.0 / (1.0 + n / 25.0)
    rating_deficit = max(0.0, (5.0 - rating) / 4.0)
    u_s = float(min(1.0, 0.5 * thin_history + 0.5 * rating_deficit))
    return u_x, u_phi, u_s


def adaptive_radius(signal: Dict[str, Any], cfg: GateConfig) -> float:
    u_x, u_phi, u_s = uncertainty_scores(signal)
    if not cfg.use_state_uncertainty:
        u_x = 0.0
    return float(cfg.eps0 + cfg.lambda_x * u_x + cfg.lambda_phi * u_phi
                 + cfg.lambda_s * u_s)


# --------------------------------------------------------------------------
# ambiguity set and robust lower bound
# --------------------------------------------------------------------------

def build_support(signal: Dict[str, Any], prefs: Dict[str, Any],
                  price_fn) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete joint support over (quality, clause realization).

    Returns (utilities, distance matrix, nominal weights). The nominal measure
    is the claim taken at face value: a point mass on the claimed quality,
    times independent clause draws at their claimed probabilities.
    """
    v = value_fn(prefs)
    impact = disclosed_impact(signal)
    q_claim = float(signal["claimed_quality"])
    qs = sorted(set(QUALITY_GRID) | {float(np.clip(q_claim, 0.0, 1.0))})
    clauses = _clauses(signal)
    k = len(clauses)
    omegas = list(product([0, 1], repeat=k)) if k else [()]

    base = float(signal["headline_price"])
    utils, weights, points = [], [], []
    for qi, q in enumerate(qs):
        w_q = 1.0 if abs(q - q_claim) < 1e-12 else 0.0
        for om in omegas:
            p = base + sum(float(clauses[i]["amount"]) for i in range(k) if om[i])
            utils.append(v(q) - impact - p)
            w_om = 1.0
            for i in range(k):
                pc = float(clauses[i]["p_claimed"])
                w_om *= pc if om[i] else (1.0 - pc)
            weights.append(w_q * w_om)
            points.append((q, om))

    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:                       # claimed quality off-grid; fall back
        w = np.ones(len(points)) / len(points)
    w = w / w.sum()

    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        qi, omi = points[i]
        for j in range(n):
            qj, omj = points[j]
            ham = (sum(a != b for a, b in zip(omi, omj)) / k) if k else 0.0
            D[i, j] = abs(qi - qj) + CLAUSE_FLIP_COST * ham
    return np.asarray(utils, dtype=float), D, w


def robust_lower_bound(signal: Dict[str, Any], prefs: Dict[str, Any],
                       radius: float) -> float:
    """inf over the W1 ball of the expected utility (Eq. 13)."""
    utils, D, w = build_support(signal, prefs, None)
    val, _ = wasserstein_worst_case(utils, D, epsilon=float(radius), weights=w)
    return float(val)


def bound_curve(signal: Dict[str, Any], prefs: Dict[str, Any],
                radii: np.ndarray, n_dual: int = 600) -> np.ndarray:
    """The whole radius -> lower-bound curve in one pass, via Kantorovich duality.

        inf_{W1(P,Phat)<=eps} E_P[u] = sup_{lambda>=0} [ g(lambda) - lambda*eps ]
        g(lambda) = sum_i phat_i * min_j ( u_j + lambda * d_ij )

    Calibrating eps(m) needs this curve for every rendering on a fine radius
    grid; solving one LP per radius is ~76x more work than solving the dual
    once. Because the sup is taken over a finite lambda grid the result is a
    lower bound on the true value, i.e. conservative in the safe direction.
    """
    utils, D, w = build_support(signal, prefs, None)
    # Beyond the steepest utility gain per unit transport distance no mass is
    # worth moving, so g(lambda) is flat there and the sup is already attained.
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (utils[:, None] - utils[None, :]) / np.where(D > 0, D, np.inf)
    lam_max = float(np.nanmax(np.where(np.isfinite(rate), rate, 0.0)))
    lam_max = max(lam_max, 1e-9) * 1.05
    lam = np.concatenate([[0.0], np.linspace(lam_max / n_dual, lam_max, n_dual)])

    # g(lambda) for every lambda at once: (n_lam, n, n) -> min over j -> dot w.
    inner = utils[None, None, :] + lam[:, None, None] * D[None, :, :]
    g = (inner.min(axis=2) * w[None, :]).sum(axis=1)

    radii = np.asarray(radii, dtype=float)
    return (g[:, None] - lam[:, None] * radii[None, :]).max(axis=0)


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------

def decide_oracle(task, signal, cfg: GateConfig) -> Tuple[str, Dict[str, Any]]:
    from .dataset import true_utility
    u = true_utility(signal, task.oracle)
    res = float(task.oracle["reservation_utility"])
    return ("ACCEPT" if u >= res else "REJECT"), {"u_hat": u, "radius": None}


def decide_claimed_eu(task, signal, cfg: GateConfig) -> Tuple[str, Dict[str, Any]]:
    u = claimed_value(signal, task.buyer_preferences) - apparent_price(signal)
    res = float(task.buyer_preferences["reservation_utility"])
    return ("ACCEPT" if u >= res else "REJECT"), {"u_hat": u, "radius": None}


def decide_conservative(task, signal, cfg: GateConfig) -> Tuple[str, Dict[str, Any]]:
    u = claimed_value(signal, task.buyer_preferences) - safe_price(signal)
    res = float(task.buyer_preferences["reservation_utility"])
    return ("ACCEPT" if u >= res else "REJECT"), {"u_hat": u, "radius": None}


VERIFY_OK = ("platform_verified", "third_party_report")


def decide_verify_then_decide(task, signal, cfg: GateConfig) -> Tuple[str, Dict[str, Any]]:
    if signal.get("quality_evidence") not in VERIFY_OK:
        return "REJECT", {"u_hat": None, "radius": None, "reason": "unverified_claim"}
    return decide_conservative(task, signal, cfg)


def decide_dro_fixed(task, signal, cfg: GateConfig) -> Tuple[str, Dict[str, Any]]:
    lb = robust_lower_bound(signal, task.buyer_preferences, cfg.fixed_radius)
    res = float(task.buyer_preferences["reservation_utility"])
    return ("ACCEPT" if lb >= res + cfg.delta else "REJECT"), {
        "u_hat": lb, "radius": cfg.fixed_radius}


def decide_robust_gate(task, signal, cfg: GateConfig) -> Tuple[str, Dict[str, Any]]:
    """The study's gate: canonicalize, decouple urgency, adapt the radius."""
    sig = dict(signal)
    if cfg.use_urgency_decoupling:
        # Timing is stripped before the economic evaluator ever sees it, which
        # is what makes dU/d(tau_clock) = 0 hold by construction (Eq. 10).
        sig["expires_in_minutes"] = None
        sig["urgency_text"] = ""

    eps = adaptive_radius(sig, cfg) if cfg.use_adaptive_radius else cfg.fixed_radius
    lb = robust_lower_bound(sig, task.buyer_preferences, eps)

    if cfg.use_contract_canonicalization:
        # The robust bound is taken at the conservative effective price rather
        # than the headline: unresolved clauses are charged, not credited.
        penalty = safe_price(sig) - float(sig["headline_price"])
        nominal_penalty = sum(float(c["amount"]) * float(c["p_claimed"])
                              for c in _clauses(sig))
        lb -= max(0.0, penalty - nominal_penalty)

    res = float(task.buyer_preferences["reservation_utility"])
    return ("ACCEPT" if lb >= res + cfg.delta else "REJECT"), {
        "u_hat": lb, "radius": eps}


def decide_gate_from_appraisal(task, signal, cfg: GateConfig,
                               estimated_value: float) -> Tuple[str, Dict[str, Any]]:
    """The gate applied on top of an LLM's valuation instead of the raw claim.

    Only the *state* channel is replaced: the model's estimated value is mapped
    back through the buyer's own value scale to an implied latent quality, and
    everything else -- contract canonicalization, the verification-conditioned
    radius, the reservation comparison -- runs exactly as in `decide_robust_gate`.

    This is the arm in which urgency decoupling is not vacuous. The operator
    strips the urgency channel *before the model appraises*, so if the model's
    valuation is urgency-contaminated the gate inherits that error when the
    operator is off, and does not when it is on.
    """
    prefs = task.buyer_preferences
    v0 = float(prefs["value_at_quality_0"])
    v1 = float(prefs["value_at_quality_1"])
    implied_q = float(np.clip((float(estimated_value) - v0) / max(v1 - v0, 1e-9), 0.0, 1.0))

    sig = dict(signal)
    sig["claimed_quality"] = implied_q
    # The appraisal already accounts for disclosed defects; charging them again
    # here would double-count them.
    sig["disclosed_attributes"] = []

    action, meta = decide_robust_gate(task, sig, cfg)
    meta = {**meta, "implied_quality": implied_q, "estimated_value": float(estimated_value)}
    return action, meta


# --- cross-round consistency closure -----------------------------------------
# The multi-round environment states each item twice: how much the buyer
# receives and how much the seller keeps. Public counts tie the two by
# a_i + s_i = c_i, an identity on data the buyer already holds. This operator is
# the same pessimistic completion the contract channel uses, applied to that
# identity: a coordinate whose two claims disagree is not merely distrusted, it
# is completed at its box minimum of zero. A coordinate the seller never stated
# is completed the same way, which is what makes the price curve monotone.


def consistent_items(signal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Item claims that are stated and survive the arithmetic check."""
    return [it for it in (signal.get("itemized_claims") or [])
            if it.get("stated", True)
            and int(it["to_buyer"]) + int(it["kept"]) == int(it["count"])
            and 0 <= int(it["to_buyer"]) <= int(it["count"])]


def closed_itemized_value(signal: Dict[str, Any]) -> float:
    """Pessimistic completion over the box, given the corroborated claims.

    min over {a in prod [0, c_i] : a_i pinned where item i is corroborated}
    equals the corroborated part alone, since every free coordinate's minimum
    is zero. No search is needed and the value is exact, not a bound.
    """
    return float(sum(int(it["to_buyer"]) * float(it["unit_value"])
                     for it in consistent_items(signal)))


def decide_consistency_closed(task, signal, cfg: GateConfig) -> Tuple[str, Dict[str, Any]]:
    """Accept on the value the seller's own claims corroborate, and nothing more.

    Falls back to the ordinary claimed valuation when the environment supplies
    no itemisation, so the arm is well defined everywhere and only differs where
    a cross-round identity actually exists.
    """
    if not (signal.get("itemized_claims") or []):
        return decide_claimed_eu(task, signal, cfg)
    u = closed_itemized_value(signal) - apparent_price(signal)
    res = float(task.buyer_preferences["reservation_utility"])
    return ("ACCEPT" if u >= res else "REJECT"), {"u_hat": u, "radius": None}


SCRIPTED = {
    "oracle": decide_oracle,
    "claimed_eu": decide_claimed_eu,
    "conservative": decide_conservative,
    "verify_then_decide": decide_verify_then_decide,
    "dro_fixed": decide_dro_fixed,
    "robust_gate": decide_robust_gate,
}

# Arms that exist only where a cross-round identity does. Kept out of SCRIPTED so
# the registered six-arm set of the three main environments is unchanged.
EXTRA_ARMS = {"consistency_closed": decide_consistency_closed}
