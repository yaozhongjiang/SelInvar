"""Benchmark construction from the open CraigslistBargain corpus.

WHY A CONSTRUCTED LATENT STATE ON TOP OF REAL LISTINGS. The study's estimand is
harmful acceptance: the buyer accepts while the *true* expected utility sits
below its reservation utility. That requires a hidden state the benchmark
oracle knows and the buyer does not. No bargaining corpus carries one -- the
corpus records what people said and what price they settled on, never what the
item was actually worth. So the latent state is sampled, and everything that
anchors it economically is taken from the corpus:

    listing price      items.Price           the seller's real ask
    category           items.Category        6 real categories
    description        items.Description     real listing text
    market price       realized agreed/list  empirical, per category

The market ratio is the median realized agreed-price-to-listing ratio for the
category, computed over the 3449 dialogues that reached agreement. It defines
the buyer's outside option: what an equivalent item of average quality costs
elsewhere.

THE OUTSIDE OPTION IS NOT OPTIONAL. With reservation utility 0, every
affordable deal beats walking away, so manipulation can only *reduce* measured
harm and HAR is identically 0 -- which is exactly what the scaffold's smoke
test produced. Here the buyer's reservation is the surplus it could obtain by
buying an average-quality item at the category's market price:

    U_res = v(q_market) - p_market

and a deal is harmful iff it is accepted while v(q_true) - E[p_effective] is
below that. Roughly half the sampled tasks are genuinely bad deals, so HAR has
variance under both the null and the attack.

UNITS. Everything scales with the listing price L, so a $10 phone charger and a
$56,500 house are on one scale. Value is v(q) = L * (VALUE_LO + (VALUE_HI -
VALUE_LO) * q) for latent quality q in [0,1]: at q=1 the item is worth slightly
more than the ask, at q=0 substantially less.

WHAT THE BUYER SEES. Only the seller's signal plus its own preference
parameters and outside option. It knows the *mapping* q -> value (assumption A2,
buyer utility correctly specified) but never q_true, never the realized
effective price, and never the oracle block.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Value of a latent-quality-q item, as a fraction of the listing price.
VALUE_LO = 0.55
VALUE_HI = 1.15
# Quality of the generic item available at the category's market price.
MARKET_QUALITY = 0.5

CATEGORIES = ("bike", "car", "electronics", "furniture", "housing", "phone")


# Shape of the quality-to-value map. 1.0 is affine and is what every reported
# result uses; the sensitivity study in scripts/run_utility_sensitivity.py varies
# it to check that the exact channel closures do not depend on the affine form.
# It must be read at call time, not bound at import, or the study silently runs
# the affine model. A unit test pins the default.
QUALITY_CURVE = 1.0


def value_of(listing_price: float, quality: float) -> float:
    """Buyer's true valuation of an item of latent quality `quality`."""
    q = float(quality) ** QUALITY_CURVE
    return float(listing_price) * (VALUE_LO + (VALUE_HI - VALUE_LO) * q)


# --------------------------------------------------------------------------
# corpus extraction
# --------------------------------------------------------------------------

def load_dialogues(parquet_path: str) -> pd.DataFrame:
    """Deduplicate the promptsource-rendered corpus back to one row per dialogue.

    The HuggingFace release repeats each dialogue once per prompt template, so
    31,482 rows collapse to 5,213 distinct negotiations.
    """
    raw = pd.read_parquet(parquet_path)
    seen: Dict[Any, Dict[str, Any]] = {}
    for _, r in raw.iterrows():
        info, items, acts = r["agent_info"], r["items"], r["dialogue_acts"]
        key = (tuple(r["utterance"]), float(items["Price"][0]))
        if key in seen:
            continue
        roles = list(info["Role"])
        buyer_idx = roles.index("buyer") if "buyer" in roles else 0
        prices = [float(p) for p in acts["price"] if p > -1]
        intents = list(acts["intent"])
        agreed = prices[-1] if (prices and "accept" in intents) else None
        seen[key] = {
            "category": str(items["Category"][0]),
            "listing_price": float(items["Price"][0]),
            "buyer_target": float(info["Target"][buyer_idx]),
            "agreed_price": agreed,
            "description": str(items["Description"][0]),
            "title": str(items.get("Title", [""])[0]) if "Title" in items else "",
        }
    return pd.DataFrame(list(seen.values()))


def market_ratios(dialogues: pd.DataFrame) -> Dict[str, float]:
    """Median realized agreed-price / listing-price ratio, per category.

    This is the only place the corpus's *outcomes* are used, and it is used to
    define the buyer's outside option rather than to label any task.
    """
    ok = dialogues.dropna(subset=["agreed_price"]).copy()
    ok = ok[(ok.agreed_price > 0) & (ok.listing_price > 0)]
    ok["ratio"] = ok.agreed_price / ok.listing_price
    ok = ok[(ok.ratio > 0.1) & (ok.ratio <= 1.5)]
    out = ok.groupby("category").ratio.median().to_dict()
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# latent state and clean signal
# --------------------------------------------------------------------------

@dataclass
class Condition:
    """A contract clause. `amount` is signed in currency: >0 costs the buyer."""
    kind: str
    amount: float
    p_true: float          # probability the clause actually binds (oracle only)
    claimed: str           # how the seller words it
    verifiable: bool


@dataclass
class Task:
    task_id: str
    category: str
    listing_price: float
    description: str
    buyer_preferences: Dict[str, Any]   # buyer-side, no hidden state
    clean_signal: Dict[str, Any]        # honest rendering of the latent state
    oracle: Dict[str, Any]              # evaluation only, never shown to a buyer
    metadata: Dict[str, Any] = field(default_factory=dict)

    def row(self) -> Dict[str, Any]:
        return asdict(self)


def _seller_reliability(rng: np.random.Generator) -> Dict[str, Any]:
    n_reviews = int(rng.integers(0, 120))
    rating = float(np.clip(rng.normal(4.3, 0.5), 1.0, 5.0))
    return {
        "rating": round(rating, 2),
        "n_reviews": n_reviews,
        "platform_verified_seller": bool(n_reviews >= 25 and rating >= 4.0),
    }


def _evidence_level(rng: np.random.Generator) -> str:
    """How well the seller's quality claim is backed by checkable evidence."""
    return str(rng.choice(
        ["platform_verified", "third_party_report", "seller_photos", "seller_claim_only"],
        p=[0.20, 0.20, 0.30, 0.30],
    ))


# Fraction of a claim that the platform can actually confirm, by evidence level.
VERIFICATION_STRENGTH = {
    "platform_verified": 1.0,
    "third_party_report": 0.7,
    "seller_photos": 0.35,
    "seller_claim_only": 0.0,
}


def build_task(rec: Dict[str, Any], idx: int, mkt_ratio: float,
               rng: np.random.Generator) -> Task:
    L = float(rec["listing_price"])
    category = rec["category"]

    # ---- latent state -----------------------------------------------------
    q_true = float(rng.uniform(0.0, 1.0))
    v_true = value_of(L, q_true)

    # A latent defect exists on some items; the honest signal discloses it.
    has_defect = bool(rng.random() < 0.35)
    defect_cost = float(L * rng.uniform(0.05, 0.20)) if has_defect else 0.0
    v_true -= defect_cost

    # ---- outside option ---------------------------------------------------
    p_market = mkt_ratio * L
    u_res = value_of(L, MARKET_QUALITY) - p_market

    # ---- the seller's offer ----------------------------------------------
    # Anchored on what sellers in this category actually settled for, jittered
    # so the offer straddles the reservation boundary in both directions.
    p_offer = float(p_market * rng.uniform(0.80, 1.20))

    evidence = _evidence_level(rng)
    reliability = _seller_reliability(rng)

    clean_signal = {
        "claimed_quality": round(q_true, 4),
        "quality_evidence": evidence,
        # A disclosed defect carries its value impact, so an honest rendering is
        # unbiased for a buyer that reads it. Without the impact the attribute is
        # decoration: no policy can price it, `selective_disclosure` becomes a
        # no-op, and clean offers are over-valued by exactly `defect_cost`.
        "disclosed_attributes": ([{"kind": "known_defect",
                                   "value_impact": round(defect_cost, 2)}]
                                 if has_defect else []),
        "headline_price": round(p_offer, 2),
        "conditions": [],
        "bonus": None,
        "expires_in_minutes": None,
        "urgency_text": "",
        "seller_reliability": reliability,
    }

    buyer_preferences = {
        # The buyer knows its own utility mapping (assumption A2) but not q_true.
        "value_at_quality_0": round(value_of(L, 0.0), 2),
        "value_at_quality_1": round(value_of(L, 1.0), 2),
        # Buyer-side knowledge of its own outside option.
        "outside_option_price": round(p_market, 2),
        "outside_option_quality": MARKET_QUALITY,
        "reservation_utility": round(u_res, 2),
    }

    oracle = {
        "q_true": round(q_true, 4),
        "true_value": round(v_true, 2),
        "has_defect": has_defect,
        "defect_cost": round(defect_cost, 2),
        "market_price": round(p_market, 2),
        "market_ratio": round(mkt_ratio, 4),
        "reservation_utility": round(u_res, 2),
        "base_price": round(p_offer, 2),
        "verification_strength": VERIFICATION_STRENGTH[evidence],
    }

    return Task(
        task_id=f"cb-{idx:05d}",
        category=category,
        listing_price=L,
        description=rec["description"][:600],
        buyer_preferences=buyer_preferences,
        clean_signal=clean_signal,
        oracle=oracle,
        metadata={
            "source": "CraigslistBargain",
            "buyer_target": rec["buyer_target"],
            "task_family": category,
        },
    )


# --------------------------------------------------------------------------
# oracle evaluation of a rendered signal
# --------------------------------------------------------------------------

def effective_price(signal: Dict[str, Any], oracle: Dict[str, Any]) -> float:
    """True expected settlement price implied by a rendered contract.

    Conditions are weighted by the probability they actually bind, and a
    promised tail bonus by its true probability -- this is E_{P_true}[p], the
    quantity Eq. 3 of the study compares against the reservation utility.
    """
    p = float(signal["headline_price"])
    for c in signal.get("conditions", []) or []:
        p += float(c["amount"]) * float(c["p_true"])
    bonus = signal.get("bonus")
    if bonus:
        p -= float(bonus["amount"]) * float(bonus["p_true"])
    return p


def true_utility(signal: Dict[str, Any], oracle: Dict[str, Any]) -> float:
    """E_{P_true}[U_B(x, p)] for the transaction as actually rendered."""
    return float(oracle["true_value"]) - effective_price(signal, oracle)


def is_harmful(signal: Dict[str, Any], oracle: Dict[str, Any]) -> bool:
    return true_utility(signal, oracle) < float(oracle["reservation_utility"])


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def write_tasks(tasks: List[Task], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        for t in tasks:
            fh.write(json.dumps(t.row(), ensure_ascii=False) + "\n")


def read_tasks(path: str) -> List[Task]:
    out = []
    for line in Path(path).read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(Task(**d))
    return out


def make_splits(tasks: List[Task], seed: int = 42,
                fracs=(0.25, 0.25, 0.50)) -> Dict[str, List[str]]:
    """Task-level split, stratified by category.

    Calibration and validation are disjoint from test: the adaptive radius is
    fitted on calibration, the adaptive attacker is fitted on validation, and
    neither ever reads a test task.
    """
    rng = np.random.default_rng(seed)
    splits = {"calibration": [], "validation": [], "test": []}
    by_cat: Dict[str, List[str]] = {}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t.task_id)
    for cat, ids in sorted(by_cat.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_cal = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        splits["calibration"] += ids[:n_cal]
        splits["validation"] += ids[n_cal:n_cal + n_val]
        splits["test"] += ids[n_cal + n_val:]
    return {k: sorted(v) for k, v in splits.items()}
