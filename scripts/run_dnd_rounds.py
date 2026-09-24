"""Multi-round Deal or No Deal: the cross-round consistency experiment.

    python scripts/run_dnd_rounds.py

Two runs are produced. The first is the matched attack pair -- an inflation that
breaks the public-count identity against one that respects it -- and tests the
prediction that only the first is exactly closable. The second sweeps how many
items the seller commits to, tracing the price of proofness as commitments
accumulate; every rendering in that sweep is honest, so anything the closing
buyer loses there is price rather than protection.

Task ids and the split come from the single-round Deal or No Deal split file, so
the tasks are the same dialogues under a different encoding and the comparison
with the single-round result is paired rather than merely parallel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buyers import GateConfig
from src.dataset_dnd import load_dialogues
from src.dataset_dnd_rounds import (COMMITMENT_LEVELS, DND_ROUND_ATTACKS,
                                    build_task, render, render_partial)
from src.experiment import run

ARMS = ["oracle", "claimed_eu", "conservative", "robust_gate",
        "consistency_closed"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="data/dnd_train.txt")
    ap.add_argument("--splits", default="data/splits_dnd.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc = json.loads((root / args.gate_config).read_text())
    ad, fx = gc["adaptive"], gc["fixed"]
    cfg = GateConfig(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                     lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                     fixed_radius=fx["radius"])

    tasks = [build_task(r, i)
             for i, r in enumerate(load_dialogues(str(root / args.dialogues)))]
    # The single-round split is keyed on `dnd-xxxxx`; this build uses `dndr-`
    # for the same index, so the same dialogues land in the same split.
    keep = {t.replace("dnd-", "") for t in
            json.loads((root / args.splits).read_text())[args.split]}
    tasks = [t for t in tasks if t.task_id.replace("dndr-", "") in keep]
    print(f"{args.split} tasks: {len(tasks)}")

    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("matched attack pair ...")
    d = run(tasks, ARMS, list(DND_ROUND_ATTACKS), cfg, render_fn=render,
            seed=args.seed, experiment_id="dnd_rounds")
    d["environment"] = "DealOrNoDealRounds"
    d.to_json(out / "main_dnd_rounds.jsonl", orient="records", lines=True)
    print(f"  wrote {len(d)} rows")

    print("commitment sweep ...")
    frames = []
    for k in COMMITMENT_LEVELS:
        rf = lambda task, attack_id, rng, informative_evidence=True, _k=k: \
            render_partial(task, _k, rng)
        f = run(tasks, ARMS, ["clean"], cfg, render_fn=rf, seed=args.seed,
                experiment_id=f"dnd_rounds_commit{k}", progress=False)
        f["environment"] = "DealOrNoDealRounds"
        f["attack_id"] = f"commit_{k}"       # honest; k is the commitment level
        frames.append(f)
    c = pd.concat(frames, ignore_index=True)
    c.to_json(out / "main_dnd_rounds_commit.jsonl", orient="records", lines=True)
    print(f"  wrote {len(c)} rows")
    print("done")


if __name__ == "__main__":
    main()
