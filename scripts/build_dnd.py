"""Build the Deal or No Deal task file and check it is well posed.

    python scripts/build_dnd.py --raw data/dnd_train.txt

Prints the same diagnostics as the Craigslist builder: the genuinely-bad-deal
fraction (harmful acceptance needs variance to be measurable at all) and the
revenue-neutrality check. Here neutrality holds by construction, because the
agreed allocation is fixed and only its description changes -- so the check is
an assertion that the code does what the design says, not a calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import make_splits, true_utility, write_tasks
from src.dataset_dnd import DND_ATTACKS, build_task, load_dialogues, render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/dnd_train.txt")
    ap.add_argument("--output", default="data/tasks_dnd.jsonl")
    ap.add_argument("--splits", default="data/splits_dnd.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tasks", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw = root / args.raw
    print(f"reading {raw}")
    recs = load_dialogues(str(raw))
    print(f"  {len(recs)} agreed, de-duplicated dialogues")
    if args.max_tasks:
        recs = recs[: args.max_tasks]

    tasks = [build_task(r, i) for i, r in enumerate(recs)]
    print(f"  built {len(tasks)} tasks")

    rng = np.random.default_rng(args.seed)
    bad, surplus, spread = [], [], []
    for t in tasks:
        sigs = {a: render(t, a, rng) for a in DND_ATTACKS}
        us = [true_utility(s, t.oracle) for s in sigs.values()]
        if max(us) - min(us) > 1e-9:
            raise AssertionError(f"{t.task_id}: true utility varies across renderings")
        u = us[0]
        res = float(t.oracle["reservation_utility"])
        bad.append(u < res)
        surplus.append(u - res)
        spread.append(sigs["combined"]["claimed_quality"]
                      - sigs["clean"]["claimed_quality"])

    frac = float(np.mean(bad))
    print(f"  revenue neutrality: OK on {len(tasks)} tasks x {len(DND_ATTACKS)} renderings")
    print(f"  genuinely-bad-deal fraction: {frac:.3f}")
    s = np.asarray(surplus)
    print(f"  U_true - U_res: mean {s.mean():+.2f}  sd {s.std():.2f} "
          f"[{np.quantile(s,.05):+.2f}, {np.quantile(s,.95):+.2f}]")
    sp = np.asarray(spread)
    print(f"  claim inflation under `combined`: mean {sp.mean():+.3f} of the pie, "
          f"nonzero on {float((sp>1e-9).mean()):.1%} of tasks")
    if not (0.2 < frac < 0.8):
        print("  WARNING: HAR will be near-degenerate on this environment")
    if float((sp > 1e-9).mean()) < 0.5:
        print("  WARNING: the attack cannot inflate the claim on most tasks")

    write_tasks(tasks, str(root / args.output))
    splits = make_splits(tasks, seed=args.seed)
    (root / args.splits).write_text(json.dumps(splits, indent=2))
    print(f"wrote {root / args.output}")
    for k, v in splits.items():
        print(f"  {k:12s} {len(v)}")


if __name__ == "__main__":
    main()
