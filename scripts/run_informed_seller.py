"""A seller who knows the closure rule (scripted buyers).

Baseline: `boundary.seed_clean`, a clean rendering with two genuine conditions.
The TRUE contract is never changed, so no action alters the seller's true
expected settlement. Actions:

  B1 keep_inclusion  keep every clean clause, add clauses, cut headline to match
  B2a swap           replace clause(s) by one of equal expected cost, higher p
  B2b omit           leave clause(s) out of the rendering; they still bind

Defenses evaluated on the rendered signal:
  claimed_eu, closure (`conservative`), gate (`robust_gate`),
  closure/gate with omission budget kappa (kappa*L added to the closed price),
  closure with a trusted reference contract (by construction restores inclusion).

Two attackers: `best_response` picks per task any action in the family producing
harm (upper bound); `transfer` is the single action best on validation, applied
to test. Harm is judged against the true contract.
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.boundary import seed_clean, _condition
from src.buyers import (GateConfig, decide_claimed_eu, decide_conservative,
                        decide_robust_gate, claimed_value, safe_price, adaptive_radius,
                        robust_lower_bound, _clauses)
from src.dataset import read_tasks, true_utility

KAPPAS = (0.0, 0.02, 0.04, 0.07, 0.10, 0.15, 0.21)
CFG = None
SCALE = 1.0


def seed_of(task_id):
    return int.from_bytes(hashlib.blake2b(f"{task_id}|reb".encode(), digest_size=8).digest(), "big") % (2**31 - 1)


def actions():
    out = [("clean", "none", ())]
    for a, p in itertools.product((0.01, 0.03, 0.06, 0.10), (0.1, 0.3, 0.5, 0.9)):
        out.append((f"add_a{a}_p{p}", "B1_keep_inclusion", ("add", a, p)))
    for S in ((0,), (1,), (0, 1)):
        for pn in (1.6, 2.2, 0.8, 0.95, 1.0):
            out.append((f"swap_{S}_{pn}", "B2a_swap", ("swap", S, pn)))
        out.append((f"omit_{S}", "B2b_omit", ("omit", S)))
    return out


def apply(task, clean, act):
    s = copy.deepcopy(clean)
    if not act:
        return s
    L = float(task.listing_price)
    if act[0] == "add":
        a = act[1] * L
        s["headline_price"] = float(s["headline_price"]) - a * act[2]
        s["conditions"] = s["conditions"] + [_condition("handling_fee", a, act[2])]
    elif act[0] == "swap":
        _, S, pn = act
        gone = [s["conditions"][i] for i in S]
        kept = [c for i, c in enumerate(s["conditions"]) if i not in S]
        E = sum(c["amount"] * c["p_true"] for c in gone)
        # pn > 1 is a multiplier on the largest dropped p; pn <= 1 an absolute p.
        p = round(min(0.95, max(c["p_true"] for c in gone) * pn) if pn > 1 else pn, 6)
        s["conditions"] = kept + [_condition("expedite_fee", E / p, p)]
    elif act[0] == "omit":
        s["conditions"] = [c for i, c in enumerate(s["conditions"]) if i not in act[1]]
    return s


def gate_lb(task, sig):
    """robust_gate's statistic, returned so kappa is applied without re-solving."""
    _, info = decide_robust_gate(task, sig, CFG)
    return float(info["u_hat"])


def one(task):
    clean = seed_clean(task, np.random.default_rng(seed_of(task.task_id)))
    for c in clean["conditions"]:
        c["amount"] *= SCALE
    truth = true_utility(clean, task.oracle)
    res_true = float(task.oracle["reservation_utility"])
    bad = truth < res_true
    res = float(task.buyer_preferences["reservation_utility"])
    L = float(task.listing_price)
    rows = []
    for name, fam, act in actions():
        s = apply(task, clean, act)
        neutral = abs(true_utility(s, task.oracle) - truth) if act and act[0] != "omit" else 0.0
        closed = claimed_value(s, task.buyer_preferences) - safe_price(s)
        lb = gate_lb(task, s)
        r = {"task_id": task.task_id, "action": name, "family": fam, "bad": int(bad), "neutral_gap": neutral,
             "claimed_eu": int(decide_claimed_eu(task, s, CFG)[0] == "ACCEPT"),
             "reference": int(decide_conservative(task, clean, CFG)[0] == "ACCEPT")}
        for k in KAPPAS:
            r[f"closure_k{k}"] = int(closed - k * L >= res)
            r[f"gate_k{k}"] = int(lb - k * L >= res + CFG.delta)
        rows.append(r)
    return rows


def init(cfg, scale):
    global CFG, SCALE
    CFG, SCALE = cfg, scale


def boot(x, reps, seed=42):
    x = np.asarray(x, float)
    g = np.random.default_rng(seed)
    m = x[g.integers(0, len(x), (reps, len(x)))].mean(1)
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on the two clean clause amounts")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--outdir", default="outputs/harness_runs")
    a = ap.parse_args()
    gc = json.loads((ROOT / "outputs/gate_config.json").read_text())
    ad, fx = gc["adaptive"], gc["fixed"]
    cfg = GateConfig(eps0=ad["eps0"], lambda_x=ad["lambda_x"], lambda_phi=ad["lambda_phi"],
                     lambda_s=ad["lambda_s"], fixed_radius=fx["radius"])
    sp = json.loads((ROOT / "data/splits.json").read_text())
    by = {t.task_id: t for t in read_tasks(str(ROOT / "data/tasks.jsonl"))}
    frames = {}
    for split in ("validation", "test"):
        ts = [by[i] for i in sp[split]]
        if a.limit:
            ts = ts[:a.limit]
        with Pool(a.workers, init, (cfg, a.scale)) as p:
            res = p.map(one, ts, chunksize=8)
        frames[split] = pd.DataFrame([r for rs in res for r in rs])
        print(split, len(ts), "tasks", flush=True)
    te, va = frames["test"], frames["validation"]
    out = ROOT / a.outdir
    out.mkdir(parents=True, exist_ok=True)
    te.to_csv(out / f"adaptive_seller_s{a.scale:g}_test_raw.csv.gz", index=False)
    print("max neutrality gap", te.neutral_gap.max())
    arms = ["claimed_eu", "reference"] + [f"{b}_k{k}" for b in ("closure", "gate") for k in KAPPAS]
    rows = []
    clean = te[te.action == "clean"].set_index("task_id")
    n = len(clean)
    for arm in arms:
        ch = clean[arm] & clean.bad
        cv = clean[arm] & (1 - clean.bad)
        base = {"arm": arm, "n": n, "HAR_clean": ch.mean(), "valid_accept_clean": cv.mean()}
        for fam in ("B1_keep_inclusion", "B2a_swap", "B2b_omit"):
            sub = te[te.family == fam].copy()
            sub["harm"] = sub[arm] & sub.bad
            br = sub.groupby("task_id").harm.max().reindex(clean.index)
            v = va[va.family == fam].copy()
            v["harm"] = v[arm] & v.bad
            bestact = v.groupby("action").harm.mean().idxmax()
            tr = sub[sub.action == bestact].set_index("task_id").harm.reindex(clean.index)
            m, lo, hi = boot(br.values, a.reps)
            mt, lt, ht = boot(tr.values, a.reps)
            ref = arm.split("_k")[0] + "_k0.0" if "_k" in arm else arm
            flips = int(((sub[arm] == 1) & (sub.task_id.map(clean[ref]) == 0)).groupby(sub.task_id).max().sum())
            rows.append({**base, "family": fam, "HAR_best_response": m, "BR_lo": lo, "BR_hi": hi,
                         "HAR_transfer": mt, "TR_lo": lt, "TR_hi": ht, "transfer_action": bestact,
                         "tasks_flipped_to_accept": flips, "n_actions": sub.action.nunique()})
    res = pd.DataFrame(rows)
    res.to_csv(out / f"adaptive_seller_s{a.scale:g}_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    print(res.drop(columns=["BR_lo", "BR_hi", "TR_lo", "TR_hi", "n"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
