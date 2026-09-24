"""Mechanism ablations, negative controls, and the evidence boundary condition.

Each variant removes exactly one design property from the frozen, calibrated
gate, so the comparison isolates mechanisms rather than tuning.

TWO THINGS THIS SCRIPT REPORTS THAT ARE EASY TO MISREAD.

`- urgency decoupling` is inert for every scripted arm, and that is a property
of the arms, not a null result. A scripted policy never reads the urgency field
in the first place, so dU/d(tau_clock) = 0 holds by construction and removing
the operator changes nothing. The evidence that the property is *needed* lives
in the LLM arms, which do respond to urgency; it is reported there.

`evidence uninformative` is the boundary condition for the whole adaptive-radius
claim. When verification metadata is drawn independently of whether the seller
manipulates, u_x and u_s carry no information about manipulation and no radius
that conditions on them can beat a fixed radius. Reporting the main result
without this row would overstate the mechanism's generality.
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
from src.buyers import GateConfig
from src.dataset import read_tasks
from src.dataset_dnd import DND_ATTACKS
from src.dataset_dnd import render as render_dnd
from src.experiment import run

ENVIRONMENTS = {
    "CraigslistBargain": {"tasks": "data/tasks.jsonl", "splits": "data/splits.json",
                          "attacks": list(ATTACKS), "render": None},
    # On Deal or No Deal every uncertainty score is constant across messages, so
    # a variant that changes a lambda changes only the radius *magnitude*, never
    # where uncertainty is placed. Rows that coincide there are evidence that the
    # mechanism has no channel, not that the ablation failed to apply.
    "DealOrNoDeal": {"tasks": "data/tasks_dnd.jsonl", "splits": "data/splits_dnd.json",
                     "attacks": list(DND_ATTACKS), "render": render_dnd},
    # Has prices and contracts, so every attack family applies and the
    # Craigslist renderer is reused unchanged.
    "AmazonHistoryPrice": {"tasks": "data/tasks_ahp.jsonl",
                           "splits": "data/splits_ahp.json",
                           "attacks": list(ATTACKS), "render": None},
}


def summarize(d: pd.DataFrame, name: str, env: str = "CraigslistBargain") -> dict:
    return {
        "environment": env,
        "variant": name,
        "n_tasks": int(d.task_id.nunique()),
        "HAR": float(d.harmful_accept.mean()),
        "valid_accept": float(d.valid_accept.mean()),
        "missed_opportunity": float(d.missed_opportunity.mean()),
        "regret": float(d.regret_norm.mean()),
        "accept_rate": float(d.accepted.mean()),
        "mean_radius": float(d.radius.mean()) if d.radius.notna().any() else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment", default="CraigslistBargain",
                    choices=sorted(ENVIRONMENTS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/raw/ablations.jsonl")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc = json.loads((root / args.gate_config).read_text())
    ad, fx = gc["adaptive"], gc["fixed"]

    def base(**kw) -> GateConfig:
        params = dict(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                      lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                      fixed_radius=fx["radius"])
        params.update(kw)
        return GateConfig(**params)

    envcfg = ENVIRONMENTS[args.environment]
    splits = json.loads((root / envcfg["splits"]).read_text())
    keep = set(splits[args.split])
    tasks = [t for t in read_tasks(str(root / envcfg["tasks"])) if t.task_id in keep]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    attacks, render_fn = envcfg["attacks"], envcfg["render"]
    print(f"{args.environment} {args.split} tasks: {len(tasks)} "
          f"({len(attacks)} attack families)")

    variants = [
        ("full method", base(), {}),
        ("- state uncertainty", base(use_state_uncertainty=False), {}),
        ("- contract canonicalization", base(use_contract_canonicalization=False), {}),
        ("- urgency decoupling", base(use_urgency_decoupling=False), {}),
        ("fixed epsilon", base(use_adaptive_radius=False), {}),
        ("evidence uninformative", base(), {"informative_evidence": False}),
    ]

    rows, frames = [], []
    for name, cfg, kw in variants:
        d = run(tasks, ["robust_gate"], attacks, cfg, seed=args.seed,
                experiment_id=f"ablation::{name}", progress=False,
                render_fn=render_fn, **kw)
        d["variant"] = name
        frames.append(d)
        rows.append(summarize(d, name, args.environment))
        print(f"  {name:32s} HAR {rows[-1]['HAR']:.4f}  valid {rows[-1]['valid_accept']:.4f}"
              f"  regret {rows[-1]['regret']:.5f}  eps {rows[-1]['mean_radius']:.3f}")

    # ---- negative control: same mean radius, assigned at random -----------
    # If the gain came from generic conservatism rather than from *where* the
    # uncertainty is placed, this control would match the full method.
    rng = np.random.default_rng(args.seed)
    full = frames[0]
    mean_eps = float(full.radius.mean())
    ctrl_cfg = base(use_adaptive_radius=False, fixed_radius=mean_eps)
    d = run(tasks, ["robust_gate"], attacks, ctrl_cfg, seed=args.seed,
            experiment_id="ablation::random-matched", progress=False,
            render_fn=render_fn)
    d["variant"] = "matched-mean fixed radius"
    frames.append(d)
    rows.append(summarize(d, "matched-mean fixed radius", args.environment))
    print(f"  {'matched-mean fixed radius':32s} HAR {rows[-1]['HAR']:.4f}"
          f"  valid {rows[-1]['valid_accept']:.4f}  regret {rows[-1]['regret']:.5f}")

    tag = "" if args.environment == "CraigslistBargain" else f"_{args.environment}"
    out = root / args.output.replace(".jsonl", f"{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_json(out, orient="records", lines=True)
    summ = pd.DataFrame(rows)
    proc = out.parent.parent / "processed"
    summ.to_csv(proc / f"ablations{tag}.csv", index=False)
    parts = [pd.read_csv(f) for f in sorted(proc.glob("ablations_*.csv"))]
    pd.concat([summ] + parts, ignore_index=True).drop_duplicates(
        ["environment", "variant"], keep="first").to_csv(
            proc / "ablations.csv", index=False)
    print(f"\nwrote {out}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
