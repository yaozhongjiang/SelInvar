"""Measure the failing side of Proposition 1's precondition.

    python scripts/run_boundary.py

`nested_addition` keeps the completion sets nested and is the control;
`incomparable_swap` breaks the inclusion while holding the seller's true
expected settlement fixed. Closure is exact on the first and must not be on the
second, which is what makes Proposition 1 a characterisation rather than a
sufficient condition.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boundary import BOUNDARY_ATTACKS, expected_settlement, render, seed_clean
from src.buyers import GateConfig, SCRIPTED
from src.dataset import read_tasks
from src.experiment import run

ARMS = ["oracle", "claimed_eu", "conservative", "robust_gate"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/raw/ablations_boundary.jsonl")
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

    # Revenue neutrality, checked paired on the same draw before anything is
    # measured: the swap must leave the true expected settlement alone, or the
    # experiment is measuring a better deal rather than a broken precondition.
    import numpy as _np
    worst = 0.0
    for t in tasks[:800]:
        sd = abs(hash(t.task_id)) % 2**31
        base = expected_settlement(seed_clean(t, _np.random.default_rng(sd)))
        got = expected_settlement(render(t, "incomparable_swap",
                                         _np.random.default_rng(sd)))
        worst = max(worst, abs(got - base))
    print(f"incomparable swap is revenue-neutral to {worst:.2e}")
    assert worst < 1e-6, "swap changed the true settlement; not a valid control"

    d = run(tasks, ARMS, list(BOUNDARY_ATTACKS), cfg, render_fn=render,
            seed=args.seed, experiment_id="boundary", progress=False)
    d["environment"] = "CraigslistBargain"
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_json(out, orient="records", lines=True)
    print(f"wrote {len(d)} rows\n")

    g = d.groupby(["buyer_id", "attack_id"]).harmful_accept.mean()
    print(f"{'arm':<16}{'nested (control)':>19}{'incomparable':>16}")
    for arm in ARMS:
        c = g.get((arm, "clean"), float("nan"))
        print(f"{arm:<16}{g.get((arm,'nested_addition'),float('nan'))-c:>+19.4f}"
              f"{g.get((arm,'incomparable_swap'),float('nan'))-c:>+16.4f}")


if __name__ == "__main__":
    main()
