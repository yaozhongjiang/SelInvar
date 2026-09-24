"""Main test-split evaluation with the frozen, calibration-fitted gate.

    python scripts/run_main.py --llm-tasks 400 --workers 1

Scripted arms run on the entire test split. The LLM arms are the expensive ones
(~2 s per decision against a local model, and the server serializes requests),
so they run on a category-stratified subsample of the *same* test tasks, drawn
with a fixed seed and reported with its own n.

Gate coefficients come from outputs/gate_config.json and are never re-fitted
here. If that file is missing the run aborts rather than silently falling back
to defaults.
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
from src.dataset_dnd import DND_ATTACKS
from src.dataset_dnd import render as render_dnd
from src.llm import LLMConfig

LLM_ARMS = ["llm_direct", "llm_safety"]


def stratified_sample(tasks, n: int, seed: int = 123):
    """Category-proportional subsample, deterministic in `seed`."""
    if n <= 0 or n >= len(tasks):
        return tasks
    rng = np.random.default_rng(seed)
    by: dict = {}
    for t in tasks:
        by.setdefault(t.category, []).append(t)
    out = []
    for cat, group in sorted(by.items()):
        group = sorted(group, key=lambda t: t.task_id)
        k = max(1, int(round(n * len(group) / len(tasks))))
        idx = rng.choice(len(group), size=min(k, len(group)), replace=False)
        out += [group[i] for i in sorted(idx)]
    return sorted(out, key=lambda t: t.task_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/raw")
    ap.add_argument("--llm-tasks", type=int, default=400)
    ap.add_argument("--llm-model", default="qwen3.6:35b-a3b")
    ap.add_argument("--llm-repeats", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=250,
                    help="completion budget; raise it for reasoning families "
                         "that would otherwise return an empty answer")
    ap.add_argument("--think", default="false",
                    help='reasoning mode: "false" to disable, or an effort level '
                         '("low"/"medium"/"high") for families that ignore the flag')
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scripted-tasks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="",
                    help="comma-separated subset of LLM arms; empty runs all. A "
                         "decoding-matched comparison only needs the direct arm.")
    ap.add_argument("--attacks", default="",
                    help="comma-separated subset of attack families for the LLM "
                         "arms; empty runs all. A replication that only needs one "
                         "contrast should not pay for six.")
    ap.add_argument("--reasoning-api", action="store_true",
                    help="use max_completion_tokens and send no sampling "
                         "parameters, which reasoning families require. An arm "
                         "run this way is not decoding-matched to the others "
                         "and must not enter the matched-pair comparison.")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-scripted", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for the LLM output file and buyer ids; use it "
                         "for a second model so the runs do not overwrite each "
                         "other or collide on buyer_id during analysis")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc_path = root / args.gate_config
    if not gc_path.exists():
        raise SystemExit(f"missing {gc_path}; run scripts/calibrate_gate.py first")
    gc = json.loads(gc_path.read_text())
    ad, fx = gc["adaptive"], gc["fixed"]
    cfg = GateConfig(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                     lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                     fixed_radius=fx["radius"])
    print(f"frozen gate: eps0={cfg.eps0:.4f} l_x={cfg.lambda_x:.4f} "
          f"l_phi={cfg.lambda_phi:.4f} l_s={cfg.lambda_s:.4f} "
          f"| fixed eps={cfg.fixed_radius:.4f}  (fitted on {gc['split']})")

    # Each environment supplies its own renderer and attack set. Dispatching on
    # the task file is what was missing: running Deal or No Deal through the
    # Craigslist renderer silently made three of its attack families no-ops,
    # because that renderer strips `known_defect` entries and DND's disclosures
    # are item-type exclusions.
    is_dnd = "dnd" in Path(args.tasks).stem
    render_fn = render_dnd if is_dnd else None
    attack_set = list(DND_ATTACKS) if is_dnd else list(ATTACKS)
    if is_dnd:
        print(f"  Deal or No Deal renderer, {len(attack_set)} attack families")

    splits = json.loads((root / args.splits).read_text())
    keep = set(splits[args.split])
    tasks = [t for t in read_tasks(str(root / args.tasks)) if t.task_id in keep]
    if args.scripted_tasks:
        tasks = tasks[: args.scripted_tasks]
    print(f"{args.split} tasks: {len(tasks)}")

    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    # ---- scripted arms, full split ---------------------------------------
    if not args.skip_scripted:
        print("running scripted arms ...")
        d_scripted = run(tasks, list(SCRIPTED.keys()), attack_set, cfg,
                         render_fn=render_fn, seed=args.seed,
                         experiment_id=f"main_scripted{args.tag}")
        # The tag must reach the scripted filename too. It did not, and running a
        # second environment's scripted arms silently overwrote the primary
        # benchmark's results file.
        d_scripted.to_json(out / f"main_scripted{args.tag}.jsonl",
                           orient="records", lines=True)
        print(f"  wrote {len(d_scripted)} rows")

    # ---- LLM arms, stratified subsample ----------------------------------
    if not args.skip_llm:
        sub = stratified_sample(tasks, args.llm_tasks, seed=args.seed + 1)
        print(f"running LLM arms on {len(sub)} tasks x {args.llm_repeats} repeats ...")
        atk = [a.strip() for a in args.attacks.split(",") if a.strip()] or list(ATTACKS)
        unknown = [a for a in atk if a not in ATTACKS]
        if unknown:
            raise SystemExit(f"unknown attack families: {unknown}")
        if "clean" not in atk:
            # The honest rendering is what the discrimination diagnostic is
            # computed from, so a subset without it cannot be interpreted.
            raise SystemExit("--attacks must include 'clean'")
        print(f"  attack families: {atk}")
        arms = [a.strip() for a in args.arms.split(",") if a.strip()] or LLM_ARMS
        unknown_arms = [a for a in arms if a not in LLM_ARMS]
        if unknown_arms:
            raise SystemExit(f"unknown arms: {unknown_arms}; known are {LLM_ARMS}")
        print(f"  arms: {arms}")
        d_llm = run(sub, arms, atk, cfg, render_fn=render_fn,
                    repeats=args.llm_repeats, seed=args.seed,
                    llm_cfg=LLMConfig(model=args.llm_model, max_tokens=args.max_tokens,
                                      reasoning_api=args.reasoning_api,
                                      think=(False if args.think == "false" else args.think)),
                    cache=str(out.parent / "llm_cache.sqlite"),
                    experiment_id=f"main_llm{args.tag}", workers=args.workers)
        if args.tag:
            # Keep the arms distinguishable once several models are pooled for
            # analysis; buyer_id alone would collide across models.
            d_llm["buyer_id"] = d_llm["buyer_id"] + args.tag
        d_llm.to_json(out / f"main_llm{args.tag}.jsonl", orient="records", lines=True)
        bad = d_llm[d_llm.status != "ok"]
        print(f"  wrote {len(d_llm)} rows; non-ok status: {len(bad)} "
              f"{bad.status.value_counts().to_dict() if len(bad) else ''}")

    print("done")


if __name__ == "__main__":
    main()
