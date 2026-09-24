"""RQ4: a seller that optimizes its rendering against the observed buyer policy.

PROTOCOL. The attacker is fitted on the validation split and evaluated on test.
It never sees a test task while choosing its policy, so the reported numbers are
transfer, not selection on the evaluation set. Two attackers are reported and
they answer different questions:

    transfer     a *policy*: for each buyer arm, the attack family with the
                 highest validation harmful-acceptance, conditioned on the
                 evidence level the seller can observe about itself. This is
                 deployable and is the number that matters.

    per-task     an upper bound: the best attack chosen per test task with
                 full knowledge of the outcome. Not deployable and labelled as
                 such -- it exists to bound how much a stronger attacker could
                 gain, since a weak attacker makes any defense look good.

Both are constrained to the same revenue-neutral, capacity-bounded attack set as
the main experiment: the seller may re-render, never re-price.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import ATTACKS
from src.buyers import GateConfig, SCRIPTED
from src.dataset import read_tasks
from src.experiment import run
from src.stats import bootstrap_mean, paired_diff

EVIDENCE_LEVELS = ["platform_verified", "third_party_report", "seller_photos",
                   "seller_claim_only"]


def evidence_of(task) -> str:
    return task.clean_signal["quality_evidence"]


def fit_policy(val: pd.DataFrame, tasks) -> pd.DataFrame:
    """Per (buyer, evidence level), the validation-best attack family."""
    ev = {t.task_id: evidence_of(t) for t in tasks}
    d = val.copy()
    d["evidence"] = d.task_id.map(ev)
    d = d[d.attack_id != "clean"]
    g = (d.groupby(["buyer_id", "evidence", "attack_id"])
           .harmful_accept.mean().reset_index())
    best = g.sort_values("harmful_accept", ascending=False) \
            .groupby(["buyer_id", "evidence"], as_index=False).first()
    return best.rename(columns={"harmful_accept": "val_har"})


def apply_policy(test: pd.DataFrame, policy: pd.DataFrame, tasks) -> pd.DataFrame:
    ev = {t.task_id: evidence_of(t) for t in tasks}
    d = test.copy()
    d["evidence"] = d.task_id.map(ev)
    sel = policy.set_index(["buyer_id", "evidence"]).attack_id.to_dict()
    d["chosen"] = [sel.get((b, e)) for b, e in zip(d.buyer_id, d.evidence)]
    return d[d.attack_id == d.chosen].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/processed/adaptive_attacker.csv")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc = json.loads((root / args.gate_config).read_text())
    ad, fx = gc["adaptive"], gc["fixed"]
    cfg = GateConfig(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                     lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                     fixed_radius=fx["radius"])

    splits = json.loads((root / args.splits).read_text())
    all_tasks = read_tasks(str(root / args.tasks))
    by_id = {t.task_id: t for t in all_tasks}
    val_tasks = [by_id[i] for i in splits["validation"]]
    test_tasks = [by_id[i] for i in splits["test"]]
    arms = list(SCRIPTED.keys())

    print(f"fitting attacker on {len(val_tasks)} validation tasks ...")
    val = run(val_tasks, arms, list(ATTACKS), cfg, seed=args.seed,
              experiment_id="attacker_fit", progress=False)
    policy = fit_policy(val, val_tasks)
    print(policy.to_string(index=False))

    print(f"\nevaluating on {len(test_tasks)} test tasks ...")
    test = run(test_tasks, arms, list(ATTACKS), cfg, seed=args.seed,
               experiment_id="attacker_eval", progress=False)

    transfer = apply_policy(test, policy, test_tasks)
    # Per-task upper bound: strongest attack chosen with outcome knowledge.
    per_task = (test[test.attack_id != "clean"]
                .sort_values("harmful_accept", ascending=False)
                .groupby(["buyer_id", "task_id"], as_index=False).first())

    rows = []
    for arm in arms:
        base = test[(test.buyer_id == arm) & (test.attack_id == "combined")]
        tr = transfer[transfer.buyer_id == arm]
        pt = per_task[per_task.buyer_id == arm]
        clean = test[(test.buyer_id == arm) & (test.attack_id == "clean")]
        rec = {"buyer_id": arm,
               "chosen_attacks": ",".join(sorted(tr.attack_id.unique())),
               "HAR_clean": bootstrap_mean(clean, "harmful_accept", args.reps)["mean"],
               "HAR_static_combined": bootstrap_mean(base, "harmful_accept", args.reps)["mean"]}
        for name, frame in [("transfer", tr), ("per_task_bound", pt)]:
            s = bootstrap_mean(frame, "harmful_accept", args.reps, args.seed)
            rec[f"HAR_{name}"] = s["mean"]
            rec[f"HAR_{name}_lo"], rec[f"HAR_{name}_hi"] = s["ci_low"], s["ci_high"]
        diff = paired_diff(tr, base, "harmful_accept", args.reps, args.seed)
        rec["transfer_minus_combined"] = diff["diff"]
        rec["p_value"] = diff["p_value"]
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("HAR_per_task_bound")
    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print("\n=== adaptive seller (fitted on validation, evaluated on test) ===")
    print(out[["buyer_id", "HAR_clean", "HAR_static_combined", "HAR_transfer",
               "HAR_per_task_bound", "chosen_attacks"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
