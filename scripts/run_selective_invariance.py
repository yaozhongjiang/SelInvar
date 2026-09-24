"""Both halves of selective invariance, measured on the same arms.

    python scripts/run_selective_invariance.py

The attack half is already reported; this run adds the genuine-improvement half
and prints them side by side, because either number alone is uninformative. A
rule that rejects everything is perfectly invariant to attacks, and a rule that
accepts everything tracks improvements perfectly.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buyers import GateConfig, SCRIPTED
from src.dataset import read_tasks
from src.experiment import run
from src.improvements import IMPROVEMENTS, render as render_improvement

ARMS = ["oracle", "claimed_eu", "conservative", "dro_fixed", "robust_gate"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/raw/main_improvements.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc = json.loads((root / args.gate_config).read_text())
    ad, fx = gc["adaptive"], gc["fixed"]
    cfg = GateConfig(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                     lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                     fixed_radius=fx["radius"])
    keep = set(json.loads((root / args.splits).read_text())[args.split])
    tasks = [t for t in read_tasks(str(root / args.tasks)) if t.task_id in keep]
    print(f"{args.split} tasks: {len(tasks)}")

    d = run(tasks, ARMS, ["clean"] + list(IMPROVEMENTS), cfg,
            render_fn=render_improvement, seed=args.seed,
            experiment_id="improvements", progress=False)
    d["environment"] = "CraigslistBargain"
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_json(out, orient="records", lines=True)
    print(f"wrote {len(d)} rows\n")

    p = d.pivot_table(index="task_id", columns=["buyer_id", "attack_id"],
                      values="accepted", aggfunc="first")
    print(f"{'arm':<16}" + "".join(f"{k.replace('genuine_discount_',''):>12}"
                                    for k in IMPROVEMENTS)
          + f"{'clean acc.':>12}")
    for arm in ARMS:
        base = p[(arm, "clean")]
        cells = [(p[(arm, k)] - base).mean() for k in IMPROVEMENTS]
        print(f"{arm:<16}" + "".join(f"{c:>+12.4f}" for c in cells)
              + f"{base.mean():>12.4f}")


if __name__ == "__main__":
    main()
