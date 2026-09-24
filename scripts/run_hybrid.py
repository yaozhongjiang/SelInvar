"""Hybrid arm: the LLM appraises the item, the robust gate decides.

    python scripts/run_hybrid.py --tasks-n 400

This is the arm in which urgency decoupling is not vacuous. The scripted
policies never read the urgency channel, so `dU/d(tau_clock)=0` holds for them
by construction and the ablation that removes the operator changes nothing. Here
the model *does* read the offer, so stripping urgency before it appraises is a
real intervention, and it is run both ways:

    llm_gate                          urgency stripped before appraisal
    llm_gate_urgency_coupled          urgency visible to the appraiser

The decoding seed is derived from the prompt, so with the operator on the
urgency rendering is byte-identical to clean and the invariance is exact rather
than approximate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import ATTACKS
from src.buyers import GateConfig
from src.dataset import read_tasks
from src.experiment import run
from src.llm import LLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_main import stratified_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--output", default="outputs/raw/main_hybrid.jsonl")
    ap.add_argument("--tasks-n", type=int, default=400)
    ap.add_argument("--model", default="qwen3.6:35b-a3b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc = json.loads((root / args.gate_config).read_text())
    ad, fx = gc["adaptive"], gc["fixed"]

    def cfg(**kw) -> GateConfig:
        p = dict(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                 lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                 fixed_radius=fx["radius"])
        p.update(kw)
        return GateConfig(**p)

    splits = json.loads((root / args.splits).read_text())
    keep = set(splits[args.split])
    tasks = [t for t in read_tasks(str(root / args.tasks)) if t.task_id in keep]
    # Same subsample and seed as run_main.py, so the hybrid arm is paired with
    # the direct LLM arms on the very same tasks.
    sub = stratified_sample(tasks, args.tasks_n, seed=args.seed + 1)
    print(f"hybrid arm on {len(sub)} test tasks, model {args.model}")

    llm = LLMConfig(model=args.model, max_tokens=150)
    cache = str(root / "outputs" / "llm_cache.sqlite")
    frames = []
    for label, decouple in [("llm_gate", True), ("llm_gate_urgency_coupled", False)]:
        print(f"  {label} (urgency decoupling {'on' if decouple else 'off'}) ...")
        d = run(sub, ["llm_gate"], list(ATTACKS), cfg(use_urgency_decoupling=decouple),
                seed=args.seed, llm_cfg=llm, cache=cache,
                experiment_id=f"hybrid::{label}")
        d["buyer_id"] = label
        bad = d[d.status != "ok"]
        print(f"    {len(d)} rows; non-ok: {len(bad)} "
              f"{bad.status.value_counts().to_dict() if len(bad) else ''}")
        frames.append(d)

    out = pd.concat(frames, ignore_index=True)
    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(p, orient="records", lines=True)

    print("\n=== urgency invariance check ===")
    for label, g in out.groupby("buyer_id"):
        piv = g.pivot_table(index="task_id", columns="attack_id", values="accepted")
        same = (piv["clean"] == piv["temporal_pressure"]).mean()
        print(f"  {label:28s} clean==urgency on {same:6.1%} of tasks  "
              f"(accept {piv['clean'].mean():.3f} vs {piv['temporal_pressure'].mean():.3f})")

    print("\n=== hybrid arm summary ===")
    print(out.groupby("buyer_id").agg(
        HAR=("harmful_accept", "mean"), valid=("valid_accept", "mean"),
        missed=("missed_opportunity", "mean"), regret=("regret_norm", "mean"),
        accept=("accepted", "mean")).round(4).to_string())
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
