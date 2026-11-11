"""Valuation dominance, checked completion by completion on the held-out tasks.

    python scripts/check_dominance.py

For a clean rendering m and a manipulated rendering m' of the same task, the
closure's completions are the binding patterns of the unresolved charges; a
benefit is credited only when it is verifiable and certain, as in
`buyers.safe_price`. With V(Q, w) = claimed value - realized price, valuation
dominance holds when every clean completion w has a manipulated completion w'
with V(Q(m'), w') <= V(Q(m), w). Ordinary inclusion is recorded beside it: the
claimed value is unchanged and every clean realized price is also a manipulated
one. The minimum over completions is asserted to equal the closed value the
deployed rule uses, so the check is about the operator that runs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import attacks, boundary
from src.buyers import _clauses, claimed_value, safe_price
from src.dataset import read_tasks

ACTIONS = (("contract_complexity", attacks.render, "clean"),
           ("tail_incentive", attacks.render, "clean"),
           ("nested_addition", boundary.render, "clean"),
           ("incomparable_swap", boundary.render, "clean"))


def seed(task_id: str, action: str) -> int:
    h = hashlib.blake2b(f"{task_id}|{action}|dominance".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 31 - 1)


def completions(signal, prefs):
    """(valuation, realized price) for every completion the closure ranges over."""
    head = float(signal["headline_price"])
    fixed, charges = 0.0, []
    for c in _clauses(signal):
        amt = float(c["amount"])
        if amt > 0:
            charges.append(amt)
        elif c.get("verifiable") and float(c["p_true"]) >= 1.0:
            fixed += amt
    value = claimed_value(signal, prefs)
    out = []
    for bits in itertools.product((0, 1), repeat=len(charges)):
        price = head + fixed + sum(b * a for b, a in zip(bits, charges))
        out.append((value - price, round(price, 6)))
    return value, out


def main():
    sp = json.loads((ROOT / "data/splits.json").read_text())["test"]
    by = {t.task_id: t for t in read_tasks(str(ROOT / "data/tasks.jsonl"))}
    tasks = [by[i] for i in sp]
    rows = []
    for action, render, clean_id in ACTIONS:
        dom = inc = 0
        for t in tasks:
            s = seed(t.task_id, action)
            clean = render(t, clean_id, np.random.default_rng(s))
            man = render(t, action, np.random.default_rng(s))
            prefs = t.buyer_preferences
            v0, c0 = completions(clean, prefs)
            v1, c1 = completions(man, prefs)
            for sig, comp, v in ((clean, c0, v0), (man, c1, v1)):
                assert abs(min(x for x, _ in comp) - (v - safe_price(sig))) < 1e-9
            dom += all(any(x1 <= x0 + 1e-9 for x1, _ in c1) for x0, _ in c0)
            inc += abs(v0 - v1) < 1e-12 and {p for _, p in c0} <= {p for _, p in c1}
        rows.append({"action": action, "tasks": len(tasks),
                     "dominance_holds": dom, "ordinary_inclusion_holds": inc})
        print(rows[-1], flush=True)
    out = ROOT / "outputs/processed/valuation_dominance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
