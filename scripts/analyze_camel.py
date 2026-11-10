"""Snapshot-matched baselines vs CaMeL (gpt-4o-mini-2024-07-18).

Sources: outputs/harness_runs/snapshot_{suite}.jsonl (undefended, canonicalization) and
outputs/harness_runs/camel_gpt-4o-mini_{suite}.jsonl (camel, camel_secpol). Pairs that
did not finish (status != ok) are excluded and counted. CaMeL outcomes are read
from logs/: how many error-feedback turns the privileged LLM needed, and whether
a security policy denied a call.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.adj_defenses import ON_CHANNEL, TRACE_SCORED

SUITES = ("banking", "travel", "slack")
OUT = ROOT / "outputs/harness_runs"
MAX_ATTEMPTS = 10


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def camel_outcome(suite, arm, u, i):
    name = "gpt-4o-mini-2024-07-18+camel" + ("+secpol" if arm == "camel_secpol" else "")
    f = ROOT / "logs" / name / suite / u / "important_instructions" / f"{i}.json"
    if not f.exists():
        return "no_log", 0, False
    d = json.loads(f.read_text())
    text = json.dumps(d["messages"])
    n_err = text.count("Provide the new code with the error fixed")
    denied = "SecurityPolicyDeniedError" in text or "security policy" in text.lower()
    kind = "exhausted" if n_err >= MAX_ATTEMPTS else ("recovered" if n_err else "clean")
    return kind, n_err, denied


def load():
    fr = []
    for s in SUITES:
        for f in (OUT / f"snapshot_{s}.jsonl", OUT / f"camel_gpt-4o-mini_{s}.jsonl"):
            d = pd.read_json(f, lines=True)
            d["suite"] = s
            fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["status"] = d.get("status").fillna("ok") if "status" in d else "ok"
    d["on_channel"] = [ON_CHANNEL[s](t) for s, t in zip(d.suite, d.injection_task)]
    d["trace_scored"] = [(s, t) in TRACE_SCORED for s, t in zip(d.suite, d.injection_task)]
    return d


def main():
    d = load()
    excluded = d[d.status != "ok"].groupby(["suite", "arm", "status"]).size()
    ok = d[d.status == "ok"].copy()
    ok["attack_success"] = ok.attack_success.astype(bool)
    ok["utility"] = ok.utility.astype(bool)
    rows = []
    for (s, arm), g in ok.groupby(["suite", "arm"]):
        for scope, sub in (("all", g), ("on_channel", g[g.on_channel]), ("off_channel", g[~g.on_channel])):
            n = len(sub)
            k, u = int(sub.attack_success.sum()), int(sub.utility.sum())
            lo, hi = wilson(k, n)
            ulo, uhi = wilson(u, n)
            rows.append({"suite": s, "arm": arm, "scope": scope, "n": n,
                         "ASR": k / n if n else np.nan, "ASR_lo": lo, "ASR_hi": hi,
                         "utility": u / n if n else np.nan, "util_lo": ulo, "util_hi": uhi})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "camel_comparison.csv", index=False)

    cm = ok[ok.arm.str.startswith("camel")].copy()
    res = [camel_outcome(s, a, u, i) for s, a, u, i in zip(cm.suite, cm.arm, cm.user_task, cm.injection_task)]
    cm["outcome"], cm["n_err"], cm["policy_denied"] = zip(*res)
    oc = (cm.groupby(["suite", "arm", "outcome"])
            .agg(n=("utility", "size"), utility=("utility", "mean"), ASR=("attack_success", "mean"),
                 policy_denied=("policy_denied", "mean")).reset_index())
    oc.to_csv(OUT / "camel_outcomes.csv", index=False)

    # Paired on the same (user, injection) pairs: canonicalization vs camel_secpol.
    piv = ok.pivot_table(index=["suite", "user_task", "injection_task"], columns="arm",
                         values=["attack_success", "utility"], aggfunc="first")
    pair = []
    for s in SUITES:
        p = piv.loc[s].dropna()
        for a in ("camel", "camel_secpol"):
            pair.append({"suite": s, "vs": a, "n_paired": len(p),
                         "utility_canon_minus": (p["utility"]["canonicalization"].astype(float)
                                                 - p["utility"][a].astype(float)).mean(),
                         "ASR_canon_minus": (p["attack_success"]["canonicalization"].astype(float)
                                             - p["attack_success"][a].astype(float)).mean()})
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print("excluded pairs:\n", excluded.to_string() if len(excluded) else "none")
    order = {"undefended": 0, "canonicalization": 1, "camel": 2, "camel_secpol": 3}
    t = t.sort_values(["suite", "scope", "arm"], key=lambda c: c.map(order) if c.name == "arm" else c)
    print(t.round(3).to_string(index=False))
    print(oc.round(3).to_string(index=False))
    print(pd.DataFrame(pair).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
