"""RQ6: does a calibrated robustness certificate survive a seller who reads it?

    python scripts/run_strategic_certificate.py

Reports, per environment and level, two certificates for the same radius family:

    mixture      the risk is averaged over the attack mixture the calibration
                 split contains -- the exchangeability assumption every
                 conformal calibration of an ambiguity set relies on.

    worst-case   the risk is maximized over the seller's action set per task
                 before aggregating, so it dominates every seller policy.

and evaluates both against three deployment conditions: the static mixture, a
best response fitted on calibration and transferred to test, and the per-task
worst case. The middle column is the deployable attack and the one that decides
whether the certificate is honest; the third is the bound's own target.

The predictor of the radius is fitted on calibration against the realized
distortion, which is an oracle quantity there and ordinary supervised learning,
not leakage: at test time only observables are read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.certify import best_response, certify, policy_mask, task_risk
from src.conformal_data import ENVIRONMENTS, FEATURES, build, decisions
from src.dataset import read_tasks

LEVELS = [0.15, 0.10, 0.05, 0.02]


def bootstrap(tasks: np.ndarray, all_tasks: np.ndarray, x: np.ndarray,
              reps: int, seed: int) -> np.ndarray:
    """Per-task means aligned to a common task index, for paired resampling."""
    s = pd.Series(np.asarray(x, float)).groupby(np.asarray(tasks)).mean()
    return np.nan_to_num(s.reindex(all_tasks).values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/processed/strategic_certificate.csv")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]

    from sklearn.ensemble import GradientBoostingRegressor

    rows: List[Dict[str, Any]] = []
    for env, (tp, sp, attacks, rfn) in ENVIRONMENTS.items():
        splits = json.loads((root / sp).read_text())
        by_id = {t.task_id: t for t in read_tasks(str(root / tp))}
        Ccal, Mcal = build(splits["calibration"], by_id, attacks, rfn)
        Cte, Mte = build(splits["test"], by_id, attacks, rfn)
        ev_cal = np.array([by_id[t].clean_signal["quality_evidence"] for t in Mcal.task])
        ev_te = np.array([by_id[t].clean_signal["quality_evidence"] for t in Mte.task])

        feats = [f for f in FEATURES if Mcal[f].nunique() > 1] or ["claim_q"]
        g = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=300)
        g.fit(Mcal[feats], Mcal.ideal)
        p_cal = np.clip(g.predict(Mcal[feats]), 0, None)
        p_te = np.clip(g.predict(Mte[feats]), 0, None)

        print(f"\n=== {env} ===  calibration {Mcal.task.nunique()} tasks, "
              f"test {Mte.task.nunique()} tasks, {len(attacks)} attack families")
        print(f"  {'level':>6s} {'certified on':>13s} {'offset':>7s} | "
              f"{'R|mixture':>10s} {'R|best-resp':>12s} {'R|worst':>8s} | {'valid|BR':>9s}")

        all_tasks = Mte.task.unique()
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, len(all_tasks), (args.reps, len(all_tasks)))

        for alpha in LEVELS:
            picked: Dict[str, Any] = {}
            for uniform, label in [(False, "mixture"), (True, "worst-case")]:
                t = certify(lambda o: decisions(Ccal, Mcal, p_cal + o)[0],
                            Mcal.task.values, alpha, uniform)
                if t is None:
                    print(f"  {alpha:6.2f} {label:>13s} {'--':>7s} | "
                          f"not certifiable at this calibration size")
                    rows.append(dict(environment=env, alpha=alpha, certified_on=label,
                                     certified=False))
                    continue

                harm, valid = decisions(Cte, Mte, p_te + t)
                r_mix = float(task_risk(Mte.task.values, harm, False).mean())
                r_wc = float(task_risk(Mte.task.values, harm, True).mean())

                pol = best_response(decisions(Ccal, Mcal, p_cal + t)[0],
                                    ev_cal, Mcal.attack.values)
                m = policy_mask(pol, ev_te, Mte.attack.values)
                Mb = Mte[m].reset_index(drop=True)
                h_b, v_b = decisions(Cte[m], Mb, (p_te + t)[m])
                r_br = float(task_risk(Mb.task.values, h_b, False).mean())

                mark = lambda x: "!" if x > alpha else " "
                print(f"  {alpha:6.2f} {label:>13s} {t:7.3f} | {r_mix:10.4f}{mark(r_mix)}"
                      f"{r_br:11.4f}{mark(r_br)}{r_wc:7.4f}{mark(r_wc)}| {v_b.mean():9.4f}")

                picked[label] = dict(
                    offset=t, r_mix=r_mix, r_br=r_br, r_wc=r_wc,
                    Rs=bootstrap(Mte.task.values, all_tasks, harm, args.reps, args.seed),
                    Rb=bootstrap(Mb.task.values, all_tasks, h_b, args.reps, args.seed),
                    V=bootstrap(Mb.task.values, all_tasks, v_b, args.reps, args.seed))
                rows.append(dict(environment=env, alpha=alpha, certified_on=label,
                                 certified=True, offset=t, risk_mixture=r_mix,
                                 risk_best_response=r_br, risk_worst_case=r_wc,
                                 valid_best_response=float(v_b.mean()),
                                 violated=bool(r_br > alpha)))

            if "mixture" in picked and "worst-case" in picked:
                a, b = picked["mixture"], picked["worst-case"]
                infl = np.array([b_.mean() / max(s_.mean(), 1e-12) for b_, s_ in
                                 ((a["Rb"][i], a["Rs"][i]) for i in idx)])
                dv = np.array([(b["V"][i] - a["V"][i]).mean() for i in idx])
                point = a["r_br"] / max(a["r_mix"], 1e-12)
                print(f"         inflation under best response {point:.2f}x "
                      f"[{np.quantile(infl, .025):.2f}, {np.quantile(infl, .975):.2f}]"
                      f"   price of validity {b['V'].mean() - a['V'].mean():+.4f} "
                      f"[{np.quantile(dv, .025):+.4f}, {np.quantile(dv, .975):+.4f}]")
                rows[-1].update(inflation=point,
                                inflation_lo=float(np.quantile(infl, .025)),
                                inflation_hi=float(np.quantile(infl, .975)),
                                price=float(b["V"].mean() - a["V"].mean()),
                                price_lo=float(np.quantile(dv, .025)),
                                price_hi=float(np.quantile(dv, .975)))

    out = pd.DataFrame(rows)
    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    c = out[out.certified == True]                                # noqa: E712
    print("\n=== certificate honesty, pooled over environments and levels ===")
    for label in ["mixture", "worst-case"]:
        s = c[c.certified_on == label]
        print(f"  certified on {label:11s}: {int(s.violated.sum())}/{len(s)} levels "
              f"violated under the fitted best response, mean valid acceptance "
              f"{s.valid_best_response.mean():.4f}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
