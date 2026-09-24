"""Build the benchmark task file from the open CraigslistBargain corpus.

    python scripts/build_dataset.py --parquet data/cb_train.parquet \
        --output data/tasks.jsonl --splits data/splits.json --seed 42

Writes one task per real listing plus a task-level, category-stratified split.
Prints the diagnostics that decide whether the benchmark is well posed at all:
the fraction of genuinely bad deals (HAR must have variance) and the
revenue-neutrality check on every attack rendering.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import ATTACKS, assert_revenue_neutral, render
from src.dataset import (build_task, is_harmful, load_dialogues, make_splits,
                         market_ratios, true_utility, write_tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/cb_train.parquet")
    ap.add_argument("--output", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tasks", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    parquet = root / args.parquet if not Path(args.parquet).is_absolute() else Path(args.parquet)

    print(f"reading {parquet}")
    dial = load_dialogues(str(parquet))
    print(f"  {len(dial)} distinct dialogues")

    ratios = market_ratios(dial)
    print("  market ratio (median realized agreed/listing) by category:")
    for k, v in sorted(ratios.items()):
        print(f"    {k:12s} {v:.4f}")

    dial = dial[dial.listing_price > 0].reset_index(drop=True)
    if args.max_tasks:
        dial = dial.iloc[: args.max_tasks].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    tasks = []
    for i, rec in enumerate(dial.to_dict("records")):
        if rec["category"] not in ratios:
            continue
        tasks.append(build_task(rec, i, ratios[rec["category"]], rng))
    print(f"  built {len(tasks)} tasks")

    # ---- well-posedness diagnostics --------------------------------------
    rng_r = np.random.default_rng(args.seed + 1)
    harmful_clean, utils = [], []
    for t in tasks:
        sigs = {a: render(t, a, rng_r) for a in ATTACKS}
        assert_revenue_neutral(t, sigs)
        harmful_clean.append(is_harmful(sigs["clean"], t.oracle))
        utils.append(true_utility(sigs["clean"], t.oracle)
                     - float(t.oracle["reservation_utility"]))

    frac_bad = float(np.mean(harmful_clean))
    print(f"  revenue neutrality: OK on {len(tasks)} tasks x {len(ATTACKS)} renderings")
    print(f"  genuinely-bad-deal fraction: {frac_bad:.3f}")
    u = np.asarray(utils)
    print(f"  U_true - U_res: mean {u.mean():+.2f}  sd {u.std():.2f} "
          f"[{np.quantile(u, .05):+.2f}, {np.quantile(u, .95):+.2f}]")
    if not (0.2 < frac_bad < 0.8):
        print("  WARNING: HAR will be near-degenerate; rebalance the latent state")

    out = root / args.output
    write_tasks(tasks, str(out))
    splits = make_splits(tasks, seed=args.seed)
    (root / args.splits).write_text(json.dumps(splits, indent=2))
    print(f"wrote {out}")
    for k, v in splits.items():
        print(f"  {k:12s} {len(v)}")


if __name__ == "__main__":
    main()
