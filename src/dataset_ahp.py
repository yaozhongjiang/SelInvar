"""Third environment: AmazonHistoryPrice (Xia et al., 2024), open on GitHub.

WHY IT IS WORTH ADDING. The Craigslist build has to *derive* the buyer's outside
option: there is no market price in that corpus, so we take the empirical median
realized agreed-to-listing ratio per category and call that the market. Here the
market price is recorded. Each product carries the seller's current ask, the
list price, and the historical `average_price` scraped from a price tracker --
so the outside option, the single quantity the reservation utility is built on,
is an observed fact rather than a corpus statistic.

    title           product description
    current_price   what the seller asks now      -> the offer
    list_price      MSRP
    average_price   historical average            -> the outside option
    category        17 real categories
    badge           the tracker's own verdict, e.g. "Good Deal"

What it does not supply is the buyer's private valuation, so the latent quality
is sampled here exactly as in the Craigslist build; this environment improves
the outside option, not the oracle. Deal or No Deal is the one that fixes the
oracle.

Unlike Deal or No Deal, this environment has prices and contracts, so all seven
attack families apply and the Craigslist renderer is reused unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.dataset import (MARKET_QUALITY, VALUE_HI, VALUE_LO,
                         VERIFICATION_STRENGTH, Task, value_of)

MONEY = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def parse_money(x: Any) -> Optional[float]:
    if x is None:
        return None
    m = MONEY.search(str(x).replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    return v if v > 0 else None


def load_products(data_dir: str) -> List[Dict[str, Any]]:
    """Read every category file and keep records with a usable price pair.

    A record is usable only if it has both an ask and a historical average: the
    average *is* the outside option, and without it the reservation utility
    would have to be invented, which is the thing this environment is here to
    avoid.
    """
    out: List[Dict[str, Any]] = []
    for f in sorted(Path(data_dir).glob("*.json")):
        try:
            blob = json.loads(f.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            continue
        records = blob.values() if isinstance(blob, dict) else blob
        for r in records:
            if not isinstance(r, dict):
                continue
            ask = parse_money(r.get("current_price"))
            avg = parse_money(r.get("average_price"))
            if ask is None or avg is None:
                continue
            # Guard against scrape artefacts: a 100x gap is a parsing failure,
            # not a bargain.
            if not (0.05 <= ask / avg <= 20.0):
                continue
            out.append({
                "title": str(r.get("title") or r.get("amazon_title") or "")[:600],
                "ask": ask,
                "market": avg,
                "list_price": parse_money(r.get("list_price")),
                "category": str(r.get("category") or f.stem),
                "badge": str(r.get("badge") or ""),
            })
    return out


def build_task(rec: Dict[str, Any], idx: int, rng: np.random.Generator) -> Task:
    ask, market = float(rec["ask"]), float(rec["market"])
    # Value is expressed against the historical average, so the scale is the
    # market rather than one seller's asking price.
    L = market

    q_true = float(rng.uniform(0.0, 1.0))
    v_true = value_of(L, q_true)
    has_defect = bool(rng.random() < 0.35)
    defect_cost = float(L * rng.uniform(0.05, 0.20)) if has_defect else 0.0
    v_true -= defect_cost

    # The outside option is the recorded historical average price.
    u_res = value_of(L, MARKET_QUALITY) - market

    evidence = str(rng.choice(
        ["platform_verified", "third_party_report", "seller_photos",
         "seller_claim_only"], p=[0.20, 0.20, 0.30, 0.30]))
    n_reviews = int(rng.integers(0, 120))
    rating = float(np.clip(rng.normal(4.3, 0.5), 1.0, 5.0))

    clean_signal = {
        "claimed_quality": round(q_true, 4),
        "quality_evidence": evidence,
        "disclosed_attributes": ([{"kind": "known_defect",
                                   "value_impact": round(defect_cost, 2)}]
                                 if has_defect else []),
        "headline_price": ask,
        "conditions": [],
        "bonus": None,
        "expires_in_minutes": None,
        "urgency_text": "",
        "seller_reliability": {"rating": round(rating, 2), "n_reviews": n_reviews,
                               "platform_verified_seller":
                                   bool(n_reviews >= 25 and rating >= 4.0)},
    }
    return Task(
        task_id=f"ahp-{idx:05d}",
        category=rec["category"],
        listing_price=L,
        description=rec["title"],
        buyer_preferences={
            "value_at_quality_0": round(value_of(L, 0.0), 2),
            "value_at_quality_1": round(value_of(L, 1.0), 2),
            "outside_option_price": round(market, 2),
            "outside_option_quality": MARKET_QUALITY,
            "reservation_utility": round(u_res, 2),
        },
        clean_signal=clean_signal,
        oracle={
            "q_true": round(q_true, 4),
            "true_value": round(v_true, 2),
            "has_defect": has_defect,
            "defect_cost": round(defect_cost, 2),
            "market_price": round(market, 2),
            "market_ratio": round(ask / market, 4),
            "reservation_utility": round(u_res, 2),
            "base_price": ask,
            "verification_strength": VERIFICATION_STRENGTH[evidence],
        },
        metadata={"source": "AmazonHistoryPrice", "task_family": rec["category"],
                  "badge": rec["badge"], "list_price": rec["list_price"]},
    )
