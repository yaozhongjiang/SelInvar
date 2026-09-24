"""Fit the gate's operating point on the calibration split only.

Both robust arms are fitted with the same objective and the same data, so the
comparison between them is not an artifact of one being hand-tuned:

    fixed radius     1 free parameter   (eps)
    adaptive radius  4 free parameters  (eps0, lambda_x, lambda_phi, lambda_s)

The adaptive arm is strictly more expressive -- it contains the fixed arm at
lambda = 0 -- so on the *fitting* split it cannot do worse. The comparison that
matters is therefore (a) held-out, and (b) at matched valid-acceptance, since
any gate can buy safety by rejecting more.

Objective is mean normalized regret against the oracle, which charges a policy
both for accepting a bad deal and for walking away from a good one. Optimizing
harmful acceptance alone selects the degenerate reject-everything gate.

Test tasks are never read here. Chosen parameters are frozen into
outputs/gate_config.json before any test evaluation.

    python scripts/calibrate_gate.py --n-random 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import ATTACKS
from src.dataset import read_tasks
from src.precompute import evaluate_config, precompute


def frontier(pre, coef: Dict[str, float], scales) -> pd.DataFrame:
    rows = []
    for s in scales:
        if "radius" in coef:
            m = evaluate_config(pre, 0, 0, 0, 0, fixed_radius=coef["radius"] * s)
        else:
            m = evaluate_config(pre, coef["eps0"] * s, coef["lambda_x"] * s,
                                coef["lambda_phi"] * s, coef["lambda_s"] * s)
        m["scale"] = s
        rows.append(m)
    return pd.DataFrame(rows)


def har_at_valid(f: pd.DataFrame, target: float) -> float:
    g = f.sort_values("valid")
    return float(np.interp(target, g.valid.values, g.HAR.values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--n-random", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--output", default="outputs/gate_config.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    splits = json.loads((root / args.splits).read_text())
    keep = set(splits[args.split])
    tasks = [t for t in read_tasks(str(root / args.tasks)) if t.task_id in keep]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    print(f"{args.split} tasks: {len(tasks)}")

    pre = precompute(tasks, list(ATTACKS))
    print(f"precomputed {len(pre['table'])} renderings")

    # ---- fixed radius: one parameter, swept exhaustively -----------------
    grid = np.round(np.arange(0.0, 1.401, 0.005), 4)
    fx = pd.DataFrame([{**evaluate_config(pre, 0, 0, 0, 0, fixed_radius=float(r)),
                        "radius_param": float(r)} for r in grid])
    best_fixed = fx.loc[fx.regret.idxmin()]
    print(f"best fixed  eps={best_fixed.radius_param:.3f}  regret {best_fixed.regret:.5f}"
          f"  HAR {best_fixed.HAR:.4f}  valid {best_fixed.valid:.4f}")

    # ---- adaptive radius: seeded search + coordinate refinement ----------
    # The adaptive family CONTAINS the fixed one (lambda = 0), so on the fitting
    # split it can never be genuinely worse. A uniform random search over the
    # 4-D box finds the fixed optimum with probability ~2e-6 per draw, so it
    # reports a spurious "adaptive loses". The fixed optimum is therefore seeded
    # explicitly and refined, which makes the nesting property hold by
    # construction and leaves only the held-out comparison to be informative.
    KEYS = ("eps0", "lambda_x", "lambda_phi", "lambda_s")

    def ev(c: Dict[str, float]) -> Dict[str, Any]:
        m = evaluate_config(pre, c["eps0"], c["lambda_x"], c["lambda_phi"],
                            c["lambda_s"])
        return {**m, **c}

    seed_cfg = {"eps0": float(best_fixed.radius_param), "lambda_x": 0.0,
                "lambda_phi": 0.0, "lambda_s": 0.0}
    best = ev(seed_cfg)

    rng = np.random.default_rng(args.seed)
    for _ in range(args.n_random):
        # Log-uniform, with an atom at zero on each coefficient: the optimum is
        # plausibly sparse and uniform sampling never proposes exact zeros.
        c = {"eps0": float(10 ** rng.uniform(-3, np.log10(0.8)))}
        for k in KEYS[1:]:
            c[k] = 0.0 if rng.random() < 0.3 else float(10 ** rng.uniform(-3, np.log10(1.5)))
        m = ev(c)
        if m["regret"] < best["regret"]:
            best = m

    # Coordinate descent on a shrinking grid around the incumbent.
    step = {"eps0": 0.08, "lambda_x": 0.30, "lambda_phi": 0.30, "lambda_s": 0.30}
    for _ in range(60):
        improved = False
        for k in KEYS:
            for mult in (-1.0, -0.5, 0.5, 1.0):
                c = {kk: best[kk] for kk in KEYS}
                c[k] = max(0.0, c[k] + mult * step[k])
                m = ev(c)
                if m["regret"] < best["regret"] - 1e-12:
                    best = m
                    improved = True
        if not improved:
            step = {k: v * 0.5 for k, v in step.items()}
            if max(step.values()) < 1e-4:
                break
    print(f"best adapt  regret {best['regret']:.5f}  HAR {best['HAR']:.4f}"
          f"  valid {best['valid']:.4f}  radius {best['radius']:.3f}")
    print(f"            eps0 {best['eps0']:.3f}  l_x {best['lambda_x']:.3f}"
          f"  l_phi {best['lambda_phi']:.3f}  l_s {best['lambda_s']:.3f}")

    # ---- frontier comparison at matched valid-acceptance -----------------
    scales = list(np.round(np.arange(0.0, 3.01, 0.1), 2))
    f_ad = frontier(pre, best, scales)
    f_fx = frontier(pre, {"radius": float(best_fixed.radius_param)}, scales)

    lo = max(f_ad.valid.min(), f_fx.valid.min())
    hi = min(f_ad.valid.max(), f_fx.valid.max())
    targets = np.linspace(lo, hi, 9)
    print(f"\nHAR at matched valid-acceptance ({args.split}):")
    print(f"  {'valid':>8} {'adaptive':>10} {'fixed':>10}   winner")
    wins = ties = 0
    for tv in targets:
        a, b = har_at_valid(f_ad, tv), har_at_valid(f_fx, tv)
        if abs(a - b) < 1e-9:
            w = "tie"
            ties += 1
        elif a < b:
            w = "adaptive"
            wins += 1
        else:
            w = "fixed"
        print(f"  {tv:8.4f} {a:10.4f} {b:10.4f}   {w}")
    print(f"  adaptive strictly better at {wins}/{len(targets)} matched points "
          f"({ties} ties)")

    out = {
        "split": args.split, "n_tasks": len(tasks),
        "objective": "mean normalized regret vs oracle",
        "adaptive": {k: best[k] for k in ["eps0", "lambda_x", "lambda_phi",
                                          "lambda_s", "regret", "HAR", "valid",
                                          "missed", "radius"]},
        "fixed": {"radius": float(best_fixed.radius_param),
                  "regret": float(best_fixed.regret), "HAR": float(best_fixed.HAR),
                  "valid": float(best_fixed.valid), "missed": float(best_fixed.missed)},
        "frontier_adaptive": f_ad.to_dict("records"),
        "frontier_fixed": f_fx.to_dict("records"),
        "matched_valid_targets": [float(x) for x in targets],
        "adaptive_wins": int(wins), "matched_points": int(len(targets)),
    }
    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
