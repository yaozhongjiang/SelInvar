"""Simulated contract channel on Deal or No Deal, whose oracle is a dataset fact.

    python scripts/run_dnd_contract.py

Answers whether exact contract closure depends on the constructed latent labels
the other two environments use. Deal or No Deal records both agents' true
per-item values, so nothing here is sampled.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.buyers import safe_price
from src.dataset import read_tasks
from src.dnd_contract import PIE, render_contract, true_utility

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/processed/dnd_contract.csv")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    splits = json.loads((root / "data/splits_dnd.json").read_text())
    by_id = {t.task_id: t for t in read_tasks(str(root / "data/tasks_dnd.jsonl"))}

    def gross(s):
        return float(s["claimed_quality"]) * PIE - sum(
            float(x.get("value_impact", 0.0)) for x in (s.get("disclosed_attributes") or []))
    naive = lambda s: gross(s) - sum(float(c["amount"]) * float(c["p_claimed"])
                                     for c in (s.get("conditions") or []))
    cons = lambda s: gross(s) - safe_price(s)

    rows, drift = [], []
    for att in ("clean", "contract_complexity", "tail_incentive"):
        hn = hc = n = 0
        for i in splits["test"]:
            t = by_id[i]
            rng = np.random.default_rng(0)
            s = render_contract(t, att, rng)
            u = true_utility(t, s)
            if att != "clean":
                drift.append(u - true_utility(t, render_contract(t, "clean", rng)))
            res = float(t.oracle["reservation_utility"])
            good = u >= res
            hn += int((naive(s) >= res) and not good)
            hc += int((cons(s) >= res) and not good)
            n += 1
        rows.append(dict(attack=att, naive_HAR=hn / n, conservative_HAR=hc / n, n=n))
    d = pd.DataFrame(rows)
    base = d[d.attack == "clean"].iloc[0]
    d["naive_effect"] = d.naive_HAR - base.naive_HAR
    d["conservative_effect"] = d.conservative_HAR - base.conservative_HAR
    p = root / a.output
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(p, index=False)
    print(f"revenue neutrality: max |drift| {max(abs(x) for x in drift):.2e}")
    print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {p}")

if __name__ == "__main__":
    main()
