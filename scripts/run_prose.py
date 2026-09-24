"""Does the measured manipulation survive when the offer is a seller's message?

    python scripts/run_prose.py --model qwen3.6:35b-a3b --tasks 40

WHY THIS EXISTS. Every model arm in the study reads a JSON record whose keys
name the channels. A reader can object that the effects, and the ease of closing
a channel by enumerating a completion set, are partly artifacts of handing the
model a labelled schema. `src/prose.py` renders the identical signal as a
seller's message; this runner evaluates the same tasks under both forms and
reports the paired attack effects side by side.

WHAT IS COMPARED. Same tasks, same attack renderings, same arms, same seeds:
`trial_seed` takes (task, attack, repeat) and never the prompt form, so a
difference between the two panels is attributable to the form. The two panels
are reported separately and never pooled -- they differ in prompt length and
therefore in decoding conditions, which is exactly the kind of mixing the rest
of the pipeline refuses.

WHAT THE RESULT MEANS. The prediction under test is that the effects persist:
the channels are economic, not lexical, so a buyer that misprices an unresolved
clause should misprice it in either form. A prose panel with every effect at
zero would mean the reported effects need the schema, and would belong in the
limitations rather than in the abstract. A prose panel with *larger* effects is
also informative and is reported as such.
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
from src.experiment import run
from src.llm import LLMConfig

ARMS = ["llm_direct"]


def stratified(tasks, n: int, seed: int = 123):
    """Category-proportional subsample, the same construction as run_main."""
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


def panel(d: pd.DataFrame, form: str) -> pd.DataFrame:
    """Per-channel effect against that form's own clean rendering."""
    rows = []
    for arm in sorted(d.buyer_id.unique()):
        sub = d[d.buyer_id == arm]
        p = sub.pivot_table(index="task_id", columns="attack_id",
                            values="harmful_accept", aggfunc="first")
        if "clean" not in p:
            continue
        for a in sorted(x for x in p.columns if x != "clean"):
            eff = (p[a] - p["clean"]).dropna()
            rows.append({"form": form, "arm": arm, "attack": a,
                         "n": int(len(eff)), "effect": float(eff.mean()),
                         "har": float(sub[sub.attack_id == a].harmful_accept.mean())})
        rows.append({"form": form, "arm": arm, "attack": "clean",
                     "n": int(p["clean"].notna().sum()),
                     "effect": 0.0,
                     "har": float(sub[sub.attack_id == "clean"].harmful_accept.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6:35b-a3b")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--task-file", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    ap.add_argument("--cache", default="outputs/llm_cache.sqlite")
    ap.add_argument("--output", default="outputs/raw/_prose_pilot.jsonl")
    ap.add_argument("--workers", type=int, default=1)
    # gpt-oss ignores think=false and spends its whole budget reasoning, which
    # returns an empty completion and scores as a parse failure on every call.
    # The level is part of that family's configuration, as in run_main.py.
    ap.add_argument("--think", default="false",
                    help='false, or an effort level such as "low"')
    # Reasoning families reject max_tokens and a non-default temperature, so
    # they need the other protocol. An arm run this way is not decoding-matched
    # to the others, which is why it is an explicit flag and is recorded.
    ap.add_argument("--reasoning-api", action="store_true")
    # A reasoning family bills and spends the completion budget on its private
    # chain, and returns an empty `content` when the budget runs out mid-thought.
    # At the 300-token default, 2,794 of 2,800 gpt-5-mini calls came back empty
    # and every effect read as exactly zero. The study's run of that family
    # needed a mean of 913 completion tokens, so the budget is a parameter here.
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gc = json.loads((root / args.gate_config).read_text())
    ad, fx = gc["adaptive"], gc["fixed"]
    cfg = GateConfig(eps0=ad["eps0"], lambda_x=ad["lambda_x"],
                     lambda_phi=ad["lambda_phi"], lambda_s=ad["lambda_s"],
                     fixed_radius=fx["radius"])

    tasks = read_tasks(str(root / args.task_file))
    keep = set(json.loads((root / args.splits).read_text())[args.split])
    tasks = stratified([t for t in tasks if t.task_id in keep], args.tasks)
    print(f"{len(tasks)} tasks x {len(ATTACKS)} renderings x {len(ARMS)} arm(s)"
          f" x 2 forms = {len(tasks)*len(ATTACKS)*len(ARMS)*2} calls")

    frames = []
    for form, prose in (("json", False), ("prose", True)):
        llm = LLMConfig(model=args.model, prose=prose,
                        think=(False if args.think == "false" else args.think),
                        reasoning_api=args.reasoning_api,
                        max_tokens=args.max_tokens)
        d = run(tasks, ARMS, list(ATTACKS), cfg, seed=args.seed, llm_cfg=llm,
                cache=str(root / args.cache), workers=args.workers,
                experiment_id=f"prose::{form}", progress=True)
        d["prompt_form"] = form
        frames.append(d)

    out = pd.concat(frames, ignore_index=True)
    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(p, orient="records", lines=True)
    print(f"\nwrote {len(out)} rows to {p}")

    bad = out[out.status != "ok"]
    if len(bad):
        print(f"  {len(bad)} non-ok calls: "
              f"{bad.status.value_counts().to_dict()}")

    summ = pd.concat([panel(out[out.prompt_form == f], f)
                      for f in ("json", "prose")], ignore_index=True)
    piv = summ.pivot_table(index=["arm", "attack"], columns="form",
                           values="effect")
    print("\nper-channel effect against that form's own clean rendering")
    print(piv.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\nclean harmful acceptance by form")
    print(summ[summ.attack == "clean"].to_string(index=False,
                                                 float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
