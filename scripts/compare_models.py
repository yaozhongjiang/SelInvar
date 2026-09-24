"""Cross-model check on the RQ1 null, and the hybrid arm's effect.

    python scripts/compare_models.py

Two questions this answers, both of which the single-model run left open:

1. Is "no attack raises harmful acceptance" a property of one model, or does it
   reproduce? The attack effects are recomputed per model arm on that arm's own
   tasks, paired within task, with Holm correction inside each arm's family.

2. Does routing the model's *appraisal* through the gate help? The hybrid arm is
   compared with the direct arm on the intersection of their tasks, so the
   comparison is paired rather than a difference of two subsamples.

A null attack effect is only interesting if the arm was responding to the offer
at all, so `analyze.py`'s discrimination diagnostic should be read beside this:
an arm that accepts good and bad deals at the same rate has no decision to
distort in the first place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import attack_effect, bootstrap_mean, holm_bonferroni, paired_diff


def load(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        # Same exclusion as analyze.py: crossdomain and improvements carry their
        # own schemas, and pooling them here put NaN into buyer_id and crashed
        # this script, leaving cross_model.csv silently stale.
        if f.name.startswith(("ablations", "_")) or f.name in (
                "crossdomain.jsonl", "main_improvements.jsonl",
                "ablations_boundary.jsonl"):
            continue
        frames.append(pd.read_json(f, lines=True))
    if not frames:
        raise SystemExit(f"no raw results in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="outputs/raw")
    ap.add_argument("--output", default="outputs/processed")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    d = load(root / args.raw)
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    model_arms = sorted(a for a in d.buyer_id.unique() if a.startswith("llm"))
    if not model_arms:
        raise SystemExit("no LLM arms present")

    # ---- 1. does the RQ1 null reproduce? ---------------------------------
    rows = []
    for arm in model_arms:
        g = d[d.buyer_id == arm]
        eff = attack_effect(d, arm, "harmful_accept", args.reps, args.seed)
        acc = attack_effect(d, arm, "accepted", args.reps, args.seed)
        clean = bootstrap_mean(g[g.attack_id == "clean"], "harmful_accept",
                               args.reps, args.seed)
        urg = acc[acc.attack_id == "temporal_pressure"].iloc[0]
        rows.append({
            "arm": arm, "model": g.model_id.iloc[0], "n_tasks": g.task_id.nunique(),
            "clean_HAR": clean["mean"], "clean_HAR_lo": clean["ci_low"],
            "clean_HAR_hi": clean["ci_high"],
            "max_attack_effect_HAR": float(eff["diff"].max()),
            "n_sig_HAR": int(eff.reject_holm.sum()),
            "n_sig_accept": int(acc.reject_holm.sum()),
            "n_sig_accept_negative": int((acc.reject_holm & (acc["diff"] < 0)).sum()),
            "urgency_effect_accept": float(urg["diff"]),
            "urgency_lo": float(urg.ci_low), "urgency_hi": float(urg.ci_high),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "cross_model.csv", index=False)

    print("=== RQ1 per arm: does any attack raise harmful acceptance? ===")
    print(summ[["arm", "model", "n_tasks", "clean_HAR", "max_attack_effect_HAR",
                "n_sig_HAR", "n_sig_accept", "n_sig_accept_negative"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== urgency effect on acceptance (paired vs own clean) ===")
    for _, r in summ.iterrows():
        print(f"  {r.arm:30s} {r.urgency_effect_accept:+.4f} "
              f"[{r.urgency_lo:+.4f}, {r.urgency_hi:+.4f}]")

    # ---- 2. hybrid vs everything else, paired on shared tasks ------------
    if "llm_gate" in model_arms:
        print("\n=== hybrid (LLM appraisal + gate) vs other arms, paired ===")
        rows = []
        others = [a for a in d.buyer_id.unique() if a != "llm_gate"]
        for other in sorted(others):
            for m in ["harmful_accept", "valid_accept", "regret_norm"]:
                r = paired_diff(d[d.buyer_id == "llm_gate"], d[d.buyer_id == other],
                                m, args.reps, args.seed)
                rows.append({"reference": "llm_gate", "other": other,
                             "metric": m, **r})
        cmp = pd.DataFrame(rows)
        cmp["reject_holm"] = holm_bonferroni(cmp.p_value.values)
        cmp.to_csv(out / "hybrid_vs_others.csv", index=False)
        for m in ["harmful_accept", "valid_accept", "regret_norm"]:
            print(f"  {m}")
            for _, r in cmp[cmp.metric == m].iterrows():
                sig = "*" if r.reject_holm else " "
                print(f"    vs {r.other:28s} {r['diff']:+.4f} "
                      f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] p={r.p_value:.4f} "
                      f"n={int(r.n_tasks)} {sig}")

    # ---- 3. urgency invariance of the hybrid arm -------------------------
    pair = [a for a in ["llm_gate", "llm_gate_urgency_coupled"] if a in model_arms]
    if len(pair) == 2:
        print("\n=== urgency decoupling, hybrid arm ===")
        for arm in pair:
            g = d[d.buyer_id == arm]
            piv = g.pivot_table(index="task_id", columns="attack_id", values="accepted")
            same = float((piv["clean"] == piv["temporal_pressure"]).mean())
            r = paired_diff(g[g.attack_id == "temporal_pressure"],
                            g[g.attack_id == "clean"], "accepted", args.reps, args.seed)
            print(f"  {arm:30s} identical to clean on {same:6.1%} of tasks; "
                  f"accept effect {r['diff']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
