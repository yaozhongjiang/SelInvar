"""Aggregate raw trials into the study's result tables, with task-level CIs.

    python scripts/analyze.py

Reads outputs/raw/*.jsonl and writes outputs/processed/:

    summary.csv            per (buyer, attack) means with bootstrap CIs
    arm_summary.csv        per-buyer headline metrics
    rq1_attack_effects.csv paired clean-vs-attack change, per arm
    rq2_arm_comparison.csv robust_gate minus each baseline, paired
    ablations.csv          written by run_ablations.py, summarized here

Every interval resamples task ids, never trial rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import arm_comparison, attack_effect, bootstrap_mean

METRICS = ["harmful_accept", "valid_accept", "missed_opportunity",
           "regret_norm", "accepted"]


STANDALONE = {"crossdomain.jsonl",            # see check_artifacts.STANDALONE
              "main_improvements.jsonl",      # own attack vocabulary
              "ablations_boundary.jsonl"}     # own attack vocabulary
ENV_BY_PREFIX = {"cb-": "CraigslistBargain", "dnd-": "DealOrNoDeal",
                 "ahp-": "AmazonHistoryPrice"}


def load_raw(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        # Artifacts that are not negotiation trial records carry their own
        # schema and have no environment column. They live under raw/ and are
        # registered in check_artifacts.STANDALONE; pooling them here aborted
        # this script and silently left the processed CSVs stale.
        if f.name.startswith(("ablations", "_")) or f.name in STANDALONE:
            continue
        frames.append(pd.read_json(f, lines=True))
    if not frames:
        raise SystemExit(f"no raw results in {raw_dir}")
    d = pd.concat(frames, ignore_index=True)
    # Older runs predate the environment column; recover it from the task id,
    # which is prefixed per environment.
    if "environment" not in d.columns:
        d["environment"] = None
    miss = d.environment.isna()
    if miss.any():
        d.loc[miss, "environment"] = d.loc[miss, "task_id"].str.extract(
            r"^(cb-|dnd-|ahp-)", expand=False).map(ENV_BY_PREFIX)
    unknown = d.environment.isna().sum()
    if unknown:
        raise SystemExit(f"{unknown} rows have no identifiable environment")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="outputs/raw")
    ap.add_argument("--output", default="outputs/processed")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir, out = root / args.raw, root / args.output
    out.mkdir(parents=True, exist_ok=True)

    d_all = load_raw(raw_dir)
    d = d_all[d_all.environment == "CraigslistBargain"].copy()
    print(f"loaded {len(d_all)} trials across environments "
          f"{dict(d_all.environment.value_counts())}")
    print(f"paired analyses use CraigslistBargain: {len(d)} trials, "
          f"{d.task_id.nunique()} tasks, arms {sorted(d.buyer_id.unique())}")

    bad = d[d.status != "ok"]
    if len(bad):
        print(f"non-ok trials: {len(bad)} {bad.status.value_counts().to_dict()}")
        # Kept in the frame and reported; a dropped failure is a silent bias.

    # ---- per (buyer, attack) with CIs ------------------------------------
    rows = []
    for (env, b, a), g in d_all.groupby(["environment", "buyer_id", "attack_id"]):
        rec = {"environment": env, "buyer_id": b, "attack_id": a,
               "n_tasks": g.task_id.nunique(), "n_trials": len(g)}
        for m in METRICS:
            s = bootstrap_mean(g, m, reps=args.reps, seed=args.seed)
            rec[m] = s["mean"]
            rec[f"{m}_lo"], rec[f"{m}_hi"] = s["ci_low"], s["ci_high"]
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values(["environment", "buyer_id", "attack_id"])
    summary.to_csv(out / "summary.csv", index=False)

    # ---- per-buyer headline ----------------------------------------------
    rows = []
    for (env, b), g in d_all.groupby(["environment", "buyer_id"]):
        rec = {"environment": env, "buyer_id": b,
               "n_tasks": g.task_id.nunique(), "n_trials": len(g),
               "n_attacks": g.attack_id.nunique(),
               "model_id": g.model_id.iloc[0]}
        for m in METRICS:
            s = bootstrap_mean(g, m, reps=args.reps, seed=args.seed)
            rec[m] = s["mean"]
            rec[f"{m}_lo"], rec[f"{m}_hi"] = s["ci_low"], s["ci_high"]
        rec["mean_latency_s"] = float(g.latency_s.mean()) if g.latency_s.notna().any() else np.nan
        rec["mean_token_out"] = float(g.token_out.mean()) if g.token_out.notna().any() else np.nan
        rows.append(rec)
    arms = pd.DataFrame(rows).sort_values(["environment", "regret_norm"])
    arms.to_csv(out / "arm_summary.csv", index=False)
    print("\n=== per-arm headline (task-level bootstrap CIs) ===")
    show = arms[["environment", "buyer_id", "n_tasks", "harmful_accept",
                 "harmful_accept_lo", "harmful_accept_hi", "valid_accept",
                 "missed_opportunity", "regret_norm"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- RQ1: does the attack move the decision? -------------------------
    rq1 = pd.concat([attack_effect(d, b, "accepted", args.reps, args.seed)
                     for b in sorted(d.buyer_id.unique())], ignore_index=True)
    rq1_har = pd.concat([attack_effect(d, b, "harmful_accept", args.reps, args.seed)
                         for b in sorted(d.buyer_id.unique())], ignore_index=True)
    rq1 = pd.concat([rq1, rq1_har], ignore_index=True)
    rq1.to_csv(out / "rq1_attack_effects.csv", index=False)

    print("\n=== RQ1: change in harmful acceptance vs the clean rendering ===")
    piv = rq1[rq1.metric == "harmful_accept"].pivot_table(
        index="buyer_id", columns="attack_id", values="diff")
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ---- is each arm responding to the economics at all? -----------------
    # A null attack effect means nothing if the arm ignores the message. On the
    # clean rendering an arm that reasons economically accepts good deals and
    # refuses bad ones; the gap between those two rates says how much decision
    # there is to distort in the first place.
    rows = []
    clean = d[d.attack_id == "clean"]
    for b, g in clean.groupby("buyer_id"):
        good, bad = g[g.is_good_deal == 1], g[g.is_good_deal == 0]
        corr = (np.corrcoef(g.accepted, g.surplus_norm)[0, 1]
                if g.accepted.nunique() > 1 else np.nan)
        varies = (d[d.buyer_id == b].groupby("task_id").accepted.nunique() > 1).mean()
        rows.append({"buyer_id": b, "corr_accept_surplus": corr,
                     "accept_given_good": good.accepted.mean(),
                     "accept_given_bad": bad.accepted.mean(),
                     "discrimination_gap": good.accepted.mean() - bad.accepted.mean(),
                     "frac_tasks_decision_varies_by_rendering": varies,
                     "n_tasks": g.task_id.nunique()})
    disc = pd.DataFrame(rows).sort_values("discrimination_gap", ascending=False)
    disc.to_csv(out / "discrimination.csv", index=False)
    print("\n=== does the arm respond to the economics? (clean renderings) ===")
    print(disc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- RQ2: ours vs each baseline, paired ------------------------------
    if "robust_gate" in set(d.buyer_id):
        parts = []
        for m in ["harmful_accept", "valid_accept", "regret_norm"]:
            parts.append(arm_comparison(d, "robust_gate", m, reps=args.reps,
                                        seed=args.seed))
        rq2 = pd.concat(parts, ignore_index=True)
        rq2.to_csv(out / "rq2_arm_comparison.csv", index=False)
        print("\n=== RQ2: robust_gate minus baseline (paired, task-level) ===")
        for m in ["harmful_accept", "valid_accept", "regret_norm"]:
            s = rq2[rq2.metric == m]
            print(f"  {m}")
            for _, r in s.iterrows():
                sig = "*" if r.reject_holm else " "
                print(f"    vs {r.other:20s} {r['diff']:+.4f} "
                      f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] p={r.p_value:.4f} {sig}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
