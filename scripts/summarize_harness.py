"""Processed summaries for the informed-seller study and the CaMeL harness comparison.

    python scripts/summarize_harness.py

Reads the per-run outputs of
  scripts/run_informed_seller.py   (seller who knows the closure rule)
  scripts/run_camel_full.sh, run_camel_isolated.py, run_gpt4o_travel.sh,
  run_camel_benign_task.sh              (CaMeL in the same AgentDojo harness)
and writes the files under outputs/processed/ that scripts/check_artifacts.py
recomputes the reported numbers from. The per-pair records are not
shipped, as with every other raw output.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.adj_defenses import ON_CHANNEL

REB = ROOT / "outputs/harness_runs"
PROC = ROOT / "outputs/processed"
MAX_ATTEMPTS = 10


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), c + h


def concealment():
    fr = []
    for s in (1, 3):
        d = pd.read_csv(REB / f"adaptive_seller_s{s:g}_summary.csv")
        d.insert(0, "scale", s)
        fr.append(d)
    pd.concat(fr).to_csv(PROC / "informed_seller.csv", index=False)


def camel():
    rows = []
    c = pd.read_csv(REB / "camel_comparison.csv")
    c.insert(0, "model", "gpt-4o-mini-2024-07-18")
    rows.append(c)
    g = pd.concat([pd.read_json(REB / "snapshot_gpt4o_travel.jsonl", lines=True),
                   pd.read_json(REB / "camel_gpt-4o_travel.jsonl", lines=True)])
    g = g[g.get("status", pd.Series("ok", index=g.index)).fillna("ok") == "ok"]
    out = []
    for arm, a in g.groupby("arm"):
        n = len(a)
        k, u = int(a.attack_success.astype(bool).sum()), int(a.utility.astype(bool).sum())
        lo, hi = wilson(k, n)
        ulo, uhi = wilson(u, n)
        out.append({"model": "gpt-4o-2024-08-06", "suite": "travel", "arm": arm, "scope": "on_channel",
                    "n": n, "ASR": k / n, "ASR_lo": lo, "ASR_hi": hi,
                    "utility": u / n, "util_lo": ulo, "util_hi": uhi})
    rows.append(pd.DataFrame(out))
    pd.concat(rows, ignore_index=True).to_csv(PROC / "camel_harness.csv", index=False)


def camel_runs():
    """Outcome counts: excluded crashes, exhausted repair attempts, benign utility."""
    rows = []
    for s in ("banking", "travel", "slack"):
        d = pd.read_json(REB / f"camel_gpt-4o-mini_{s}.jsonl", lines=True)
        st = d.get("status", pd.Series("ok", index=d.index)).fillna("ok")
        rows.append({"suite": s, "metric": "camel_pairs_not_finished",
                     "value": int(((d.arm == "camel") & (st != "ok")).sum())})
    oc = pd.read_csv(REB / "camel_outcomes.csv")
    for s, g in oc[(oc.arm == "camel") & (oc.outcome == "exhausted")].groupby("suite"):
        rows.append({"suite": s, "metric": "camel_exhausted", "value": int(g.n.sum())})
    for s, g in oc[oc.arm == "camel"].groupby("suite"):
        rows.append({"suite": s, "metric": "camel_pairs_finished", "value": int(g.n.sum())})
    b = pd.read_json(REB / "camel_gpt-4o-mini_travel_benign.jsonl", lines=True)
    b = b[b.arm == "camel"]
    rows.append({"suite": "travel", "metric": "camel_benign_successes", "value": int(b.utility.sum())})
    rows.append({"suite": "travel", "metric": "camel_benign_tasks", "value": len(b)})
    pd.DataFrame(rows).to_csv(PROC / "camel_harness_runs.csv", index=False)


def coverage():
    """Share of undefended gpt-4o-mini pairs, and of their successes, on closable goals.

    Two configurations: the dated snapshot every arm of the CaMeL comparison
    uses, and the undated alias of the original AgentDojo runs.
    """
    snap = pd.concat([pd.read_json(REB / f"snapshot_{s}.jsonl", lines=True).assign(suite=s)
                      for s in ("banking", "travel", "slack")])
    alias = pd.read_json(ROOT / "outputs/raw/agentdojo.jsonl", lines=True)
    alias = alias[alias.model_id == "gpt-4o-mini"]
    rows = []
    for config, r in (("snapshot", snap), ("alias", alias)):
        r = r[(r.arm == "undefended") & (r.vector.fillna("all") == "all")].copy()
        r["on"] = [ON_CHANNEL[s](t) for s, t in zip(r.suite, r.injection_task)]
        succ = r[r.attack_success.astype(bool)]
        rows.append({"config": config, "pairs": len(r), "pairs_on_channel": int(r.on.sum()),
                     "successes": len(succ), "successes_on_channel": int(succ.on.sum())})
    pd.DataFrame(rows).to_csv(PROC / "agentdojo_coverage.csv", index=False)


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    concealment()
    camel()
    camel_runs()
    coverage()
    for f in sorted(PROC.glob("*harness*.csv")):
        print("wrote", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
