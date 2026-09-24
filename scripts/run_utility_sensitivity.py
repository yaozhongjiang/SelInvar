"""Does the exact channel closure depend on the buyer's utility being affine?

    python scripts/run_utility_sensitivity.py

WHY THIS EXISTS. Every environment in the study prices quality with the same
affine map, and the oracle is that same map evaluated at the true state rather
than the claimed one. That is the right construction for measuring decision
distortion---hold the utility model fixed, vary only the state---but it invites
the objection that the three environments are one measuring instrument, and that
the exact zeros are an artifact of linearity.

The criterion answers that objection before the run. Proposition~1's argument
needs the acceptance rule to be monotone in the closed value, and nothing more;
it never uses the shape of the quality-to-value map. The contract and tail
channels act on the price side, which the map does not touch at all. So the
prediction is sharp:

  * contract complexity and tail incentives stay at EXACTLY 0.0000 under closure
    for every curvature, and
  * the latent channels' magnitudes move, because the same displacement in
    claimed quality buys a different amount of value once the map is curved.

A curvature that broke the first half would mean the closure result is a
property of the affine model rather than of the operator, which is exactly what
a reader should want ruled out.

WHAT IS VARIED. `value_of(L, q) = L * (LO + (HI-LO) * q**gamma)` with gamma in
{0.5, 1.0, 2.0}: diminishing, affine, and increasing returns to quality. The
curve is applied to the oracle's true value, the buyer's valuation, and the
reservation utility together, so each gamma is a coherent utility model rather
than a mismatch between buyer and oracle. Latent states are regenerated from the
same seed, so the three runs share their true qualities and differ only in how
those qualities are priced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset as ds
from src.attacks import ATTACKS
from src.buyers import GateConfig, SCRIPTED
from src.experiment import run

CURVES = (0.5, 1.0, 2.0)
CLOSED = ("contract_complexity", "tail_incentive")


def build_split(root: Path, corpus: str, splits_path: str, split: str,
                seed: int):
    """Rebuild tasks from the corpus under the curve currently set.

    Reading data/tasks.jsonl would defeat the study: that file was written once
    under the affine map, so its true values, preference endpoints, and
    reservation utilities are all baked in. Setting the curve would then change
    only the buyer's side and leave the oracle affine -- a buyer/oracle mismatch
    measuring a specification error rather than a curvature effect. Rebuilding
    from the corpus with the original seed regenerates the same latent qualities
    and re-prices them, which is what the comparison needs.
    """
    dial = ds.load_dialogues(str(root / corpus))
    ratios = ds.market_ratios(dial)
    rng = np.random.default_rng(seed)
    tasks = []
    for i, rec in enumerate(dial.to_dict("records")):
        if rec["category"] not in ratios:            # same filter as the builder
            continue
        tasks.append(ds.build_task(rec, i, ratios[rec["category"]], rng))
    keep = set(json.loads((root / splits_path).read_text())[split])
    return [t for t in tasks if t.task_id in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/cb_train.parquet")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/raw/_utility_sensitivity.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc = json.loads((root / args.gate_config).read_text())
    ad, fx = gc["adaptive"], gc["fixed"]
    cfg = GateConfig(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                     lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                     fixed_radius=fx["radius"])

    frames = []
    for gamma in CURVES:
        ds.QUALITY_CURVE = gamma
        tasks = build_split(root, args.corpus, args.splits, args.split,
                            args.seed)
        print(f"gamma={gamma}: {len(tasks)} tasks")
        d = run(tasks, list(SCRIPTED.keys()), list(ATTACKS), cfg,
                seed=args.seed, experiment_id=f"utility_gamma{gamma}",
                progress=False)
        d["quality_curve"] = gamma
        d["environment"] = "CraigslistBargainCurved"
        frames.append(d)
    ds.QUALITY_CURVE = 1.0                      # never leave it set

    out = pd.concat(frames, ignore_index=True)
    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(p, orient="records", lines=True)
    print(f"wrote {len(out)} rows")

    # ---- the prediction, checked here so a failure cannot go unnoticed -----
    g = out.groupby(["quality_curve", "buyer_id", "attack_id"]).harmful_accept.mean()
    print(f"\n{'curve':>6}  {'arm':<18}" + "".join(f"{a[:12]:>14}" for a in CLOSED)
          + f"{'state':>10}{'disclosure':>12}{'clean':>8}")
    for gamma in CURVES:
        for arm in ("claimed_eu", "conservative", "robust_gate"):
            clean = g.get((gamma, arm, "clean"), float("nan"))
            cells = [g.get((gamma, arm, a), float("nan")) - clean for a in CLOSED]
            st = g.get((gamma, arm, "state_misrepresentation"), float("nan")) - clean
            di = g.get((gamma, arm, "selective_disclosure"), float("nan")) - clean
            print(f"{gamma:>6}  {arm:<18}" + "".join(f"{c:>+14.4f}" for c in cells)
                  + f"{st:>+10.4f}{di:>+12.4f}{clean:>8.4f}")


if __name__ == "__main__":
    main()
