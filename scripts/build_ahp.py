"""Build the AmazonHistoryPrice task file and check it is well posed.

    python scripts/build_ahp.py

Same diagnostics as the other builders: revenue neutrality across renderings and
the genuinely-bad-deal fraction, which must have variance for HAR to mean
anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import ATTACKS, assert_revenue_neutral, render
from src.dataset import is_harmful, make_splits, true_utility, write_tasks
from src.dataset_ahp import build_task, load_products


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/AmazonHistoryPrice")
    ap.add_argument("--output", default="data/tasks_ahp.jsonl")
    ap.add_argument("--splits", default="data/splits_ahp.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    recs = load_products(str(root / args.raw))
    print(f"usable products (ask and historical average present): {len(recs)}")
    cats = {}
    for r in recs:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print("  categories:", dict(sorted(cats.items(), key=lambda kv: -kv[1])))

    rng = np.random.default_rng(args.seed)
    tasks = [build_task(r, i, rng) for i, r in enumerate(recs)]

    rng2 = np.random.default_rng(args.seed + 1)
    bad, surplus = [], []
    for t in tasks:
        sigs = {a: render(t, a, rng2) for a in ATTACKS}
        assert_revenue_neutral(t, sigs)
        bad.append(is_harmful(sigs["clean"], t.oracle))
        surplus.append(true_utility(sigs["clean"], t.oracle)
                       - float(t.oracle["reservation_utility"]))
    frac = float(np.mean(bad))
    print(f"  revenue neutrality: OK on {len(tasks)} tasks x {len(ATTACKS)} renderings")
    print(f"  genuinely-bad-deal fraction: {frac:.3f}")
    s = np.asarray(surplus)
    print(f"  U_true - U_res: mean {s.mean():+.2f}  sd {s.std():.2f}")
    ratios = np.array([t.oracle["market_ratio"] for t in tasks])
    print(f"  ask / historical average: median {np.median(ratios):.3f} "
          f"[{np.quantile(ratios,.05):.2f}, {np.quantile(ratios,.95):.2f}]")
    if not (0.2 < frac < 0.8):
        print("  WARNING: HAR will be near-degenerate")

    write_tasks(tasks, str(root / args.output))
    splits = make_splits(tasks, seed=args.seed)
    (root / args.splits).write_text(json.dumps(splits, indent=2))
    print(f"wrote {root / args.output}")
    for k, v in splits.items():
        print(f"  {k:12s} {len(v)}")


if __name__ == "__main__":
    main()
