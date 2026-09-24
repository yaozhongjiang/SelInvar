"""External OOD check on BazaarBench, reported separately from the main benchmark.

    python scripts/run_ood_bazaarbench.py --pressure-db <recovered.db>

WHY THIS IS A REAL EXTERNAL TEST. BazaarBench ships a *counterfactual fork*:
both arms share one simulated marketplace up to tick 372, then one applies
seller-side pressure to the same 20 treated buyer agents while the other does
not. Same world, same agents, same listings, different seller behaviour -- the
same estimand as our clean/attacked pairing, produced by someone else's
benchmark, with different models and multi-turn dialogue instead of a
single-shot gate.

It is NOT our benchmark and is never pooled with it. There is no hidden-state
oracle, so harmful acceptance in our sense cannot be computed; the outcome is
BazaarBench's own `market_premium` (final price minus its market baseline) and
the price actually paid. Its treatment is BazaarBench's "pressure", not our
attack families. The unit is the treated agent, so only large effects are
detectable.

THREE DATA DEFECTS, ALL FOUND BY INSPECTION AND ALL REPORTED RATHER THAN
WORKED AROUND:

1. The pressure-arm database fails `PRAGMA quick_check` outright. It is
   reconstructed with `sqlite3 .recover`; how much that lost is *measured*
   below by comparing pre-fork rows, which must be identical across arms
   because the arms share that history.
2. Every mental-price column (`buyer_drift`, `buyer_after_chat`, ...) is 100%
   NULL even in the intact baseline database, so BazaarBench's own
   valuation-drift instrumentation carries no data in this release. The
   decision-distortion question it was built to answer cannot be asked here.
3. Thread ids are assigned per-arm after the fork, so joining pressure-arm
   thread ids through the baseline database silently maps them to *different*
   listings. Doing that produced a spurious "pressure raises offers to 4.3x the
   asking price" (p=0.05), with offers of $220 against a $4 listing. Each arm
   is therefore joined only through its own tables, and the fraction of
   post-fork threads is reported so the hazard stays visible.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FORK_TICK = 372
BASE_JSON = "llm_judge/L2-baseline_qwen_no-swap.json"
PRESSURE_JSON = "llm_judge/L2-falsif_qwen_pressure-no-swap.json"
BASE_DB = "level2/L2-baseline_qwen_no-swap.db"


def con(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def tables(path: Path) -> Dict[str, Any]:
    c = con(path)
    threads = {r[0]: {"listing_id": r[1], "created_at_tick": r[2]}
               for r in c.execute(
                   "select thread_id, listing_id, created_at_tick from threads")}
    listings = {r[0]: {"price_usd": r[1] / 100.0, "category": r[2]}
                for r in c.execute("select listing_id, price_cents, category from listings")}
    tx = pd.read_sql(
        "select buyer_agent_id, accept_tick, final_price_cents, "
        "market_baseline_cents, market_premium from transaction_utility", c)
    return {"threads": threads, "listings": listings, "tx": tx}


def extract_offers(path: Path, tb: Dict[str, Any], arm: str) -> pd.DataFrame:
    """Offers by treated agents, joined ONLY through this arm's own tables."""
    d = json.loads(path.read_text())
    rows: List[Dict[str, Any]] = []
    for agent, recs in d["agents"].items():
        for rec in recs:
            for act in (rec.get("actions") or []):
                if act.get("kind") != "make_offer":
                    continue
                th = tb["threads"].get(act.get("thread_id"))
                li = tb["listings"].get(th["listing_id"]) if th else None
                rows.append({
                    "arm": arm, "agent_id": int(agent), "tick": rec.get("tick"),
                    "thread_id": act.get("thread_id"),
                    "offer_usd": float(act.get("amount_usd")),
                    "list_price_usd": li["price_usd"] if li else np.nan,
                    "category": li["category"] if li else None,
                    "joined": li is not None,
                    "thread_post_fork": (th["created_at_tick"] > FORK_TICK) if th else None,
                })
    d2 = pd.DataFrame(rows)
    d2["offer_ratio"] = d2.offer_usd / d2.list_price_usd.replace(0, np.nan)
    return d2


def agent_paired(d: pd.DataFrame, metric: str, reps: int = 5000,
                 seed: int = 42) -> Dict[str, float]:
    """Agent-level paired difference (pressure - baseline); the fork treats agents."""
    piv = d.pivot_table(index="agent_id", columns="arm", values=metric,
                        aggfunc="mean").dropna()
    if piv.empty or not {"baseline", "pressure"} <= set(piv.columns):
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n_agents": 0}
    diff = (piv["pressure"] - piv["baseline"]).values
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, diff.size, size=(reps, diff.size))].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(reps, diff.size))
    null = (signs * diff).mean(axis=1)
    return {"diff": float(diff.mean()),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "p_value": float((np.abs(null) >= abs(diff.mean()) - 1e-15).mean()),
            "n_agents": int(diff.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bazaar", default="../../frameshield_r/data/bazaarbench")
    ap.add_argument("--pressure-db", required=True,
                    help="recovered pressure-arm DB (the shipped one is corrupt)")
    ap.add_argument("--output", default="outputs/processed/ood_bazaarbench.csv")
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    bz = (root / args.bazaar).resolve()
    tb_base = tables(bz / BASE_DB)
    tb_press = tables(Path(args.pressure_db))

    # ---- how much did the recovery lose? ---------------------------------
    # Pre-fork history is shared, so pre-fork transaction counts must match.
    pre_b = int((tb_base["tx"].accept_tick <= FORK_TICK).sum())
    pre_p = int((tb_press["tx"].accept_tick <= FORK_TICK).sum())
    print("=== recovery integrity (pre-fork rows must be identical) ===")
    print(f"  pre-fork transactions: baseline {pre_b}, pressure {pre_p} "
          f"-> recovered {pre_p / max(pre_b, 1):.1%}")
    if pre_p < pre_b:
        print(f"  WARNING: {pre_b - pre_p} pre-fork rows lost by .recover; "
              f"post-fork pressure counts are a LOWER BOUND")

    # ---- is the treatment window present at all? -------------------------
    # Everything below is meaningless if the pressure arm has no data after the
    # fork, so this is checked before any comparison is computed.
    max_tick_p = int(tb_press["tx"].accept_tick.max())
    post_p = int((tb_press["tx"].accept_tick > FORK_TICK).sum())
    print(f"\n=== treatment window (fork at tick {FORK_TICK}) ===")
    print(f"  baseline  max accept_tick {int(tb_base['tx'].accept_tick.max())}, "
          f"post-fork transactions {int((tb_base['tx'].accept_tick > FORK_TICK).sum())}")
    print(f"  pressure  max accept_tick {max_tick_p}, post-fork transactions {post_p}")
    if post_p == 0:
        print("\n  BLOCKED: the pressure arm contains no data after the fork.")
        print("  The shipped database is corrupt and `.recover` salvages only up to")
        print(f"  tick ~{max_tick_p}, which is BEFORE the fork at {FORK_TICK}, so the entire")
        print("  treatment window is absent. The JSON trajectories do carry the")
        print("  pressure arm's post-fork offers, but no JSON carries listing prices,")
        print("  and the thread->listing mapping needed to normalize them lives in")
        print("  the unrecoverable region. Joining through the baseline arm instead")
        print("  is invalid (ids are re-used across the fork) and fabricates a large")
        print("  spurious effect -- see this file's docstring.")
        print("\n  This OOD check cannot be completed with the data on hand. It needs")
        print("  an intact pressure-arm database or a re-run of the BazaarBench cell.")
        out = root / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"outcome": "ALL", "source": "bazaarbench",
                       "status": "blocked: pressure arm truncated before the fork",
                       "pressure_max_tick": max_tick_p, "fork_tick": FORK_TICK,
                       "prefork_rows_recovered": pre_p,
                       "prefork_rows_expected": pre_b}]).to_csv(out, index=False)
        print(f"\nwrote {out}")
        return

    # ---- outcome 1: BazaarBench's own market_premium ----------------------
    # This cell is the *no-swap* falsification arm: no agent is replaced, the
    # pressure prompt is applied to the marketplace. Every post-fork buyer is
    # therefore treated, and restricting to the 20 agents whose trajectories are
    # logged would discard two thirds of the data for no design reason. Both
    # populations are reported; the all-buyer one is primary.
    treated = set(json.loads((bz / PRESSURE_JSON).read_text())["treated_agents"])
    rows = []
    frames = []
    for arm, tb in [("baseline", tb_base), ("pressure", tb_press)]:
        t = tb["tx"][tb["tx"].accept_tick > FORK_TICK].copy()
        t["arm"] = arm
        t["agent_id"] = t.buyer_agent_id
        t["price_usd"] = t.final_price_cents / 100.0
        t["premium_usd"] = t.market_premium / 100.0
        t["logged_agent"] = t.buyer_agent_id.isin(treated)
        frames.append(t)
    tx = pd.concat(frames, ignore_index=True)

    for pop, sub in [("all post-fork buyers", tx),
                     (f"the {len(treated)} logged agents", tx[tx.logged_agent])]:
        print(f"\n=== post-fork transactions, {pop} ===")
        for arm, g in sub.groupby("arm"):
            print(f"  {arm:10s} n={len(g):4d}  agents={g.agent_id.nunique():3d}  "
                  f"mean premium ${g.premium_usd.mean():+.2f}  "
                  f"mean paid ${g.price_usd.mean():.2f}")
        print("  pressure - baseline (unit = buyer agent):")
        for metric, label in [("premium_usd", "market premium (USD)"),
                              ("price_usd", "price paid (USD)")]:
            r = agent_paired(sub, metric, args.reps)
            rows.append({"outcome": metric, "population": pop,
                         "source": "transaction_utility", **r})
            print(f"    {label:24s} {r['diff']:+8.3f} "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p_value']:.4f} "
                  f"n_agents={r['n_agents']}")

    # ---- outcome 2: offers, joined per-arm --------------------------------
    off = pd.concat([extract_offers(bz / BASE_JSON, tb_base, "baseline"),
                     extract_offers(bz / PRESSURE_JSON, tb_press, "pressure")],
                    ignore_index=True)
    print("\n=== offers by treated agents (each arm joined through its own tables) ===")
    for arm, g in off.groupby("arm"):
        print(f"  {arm:10s} offers={len(g):4d}  joined={g.joined.mean():.0%}  "
              f"post-fork threads={g.thread_post_fork.mean():.0%}  "
              f"mean offer/list={g.offer_ratio.mean():.3f}")
    ok = off[off.joined & off.offer_ratio.notna()]
    for metric, label in [("offer_ratio", "offer / list price"),
                          ("offer_usd", "offer (USD)")]:
        r = agent_paired(ok, metric, args.reps)
        rows.append({"outcome": metric, "source": "json_trajectories", **r})
        print(f"    {label:24s} {r['diff']:+8.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p_value']:.4f} "
              f"n_agents={r['n_agents']}")

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    ok.to_csv(out.with_name("ood_bazaarbench_offers.csv"), index=False)
    tx.to_csv(out.with_name("ood_bazaarbench_transactions.csv"), index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
