"""RQ7: gate the radius on predicted attackability instead of estimated distortion.

    python scripts/run_gated_radius.py

Reports, per environment, the scalar radius the previous sections certify and the
gated rule of Eq. (2), both certified by the same policy-uniform Learn-then-Test
procedure on the same calibration tasks, and both evaluated on held-out test
tasks under two deployment conditions: the static attack mixture, and a seller
best response fitted on calibration and transferred to test.

The oracle row flags exactly the attackable tasks and is not deployable; it is
the ceiling the gate is trying to reach.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import GradientBoostingClassifier
from src.certify import best_response, policy_mask, task_risk
from src.conformal_data import ENVIRONMENTS, FEATURES, GRID, build, decisions
from src.gated_radius import (OFFSET_GRID, TAU_GRID, apply_gate, certify_gate,
                              crossfit_scores, safe_radius)
from src.dataset import read_tasks


def design(M: pd.DataFrame) -> pd.DataFrame:
    g = M.groupby("task")
    X = g[[f for f in FEATURES if M[f].nunique() > 1]].mean()
    # the reservation gap is observable and the distortion target ignored it
    X["res_over_L"] = g.res.first() / g.L.first()
    X["n_msg"] = g.size()
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", default="outputs/processed/gated_radius.csv")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []

    for env, (tp, sp, attacks, rfn) in ENVIRONMENTS.items():
        splits = json.loads((root / sp).read_text())
        by_id = {t.task_id: t for t in read_tasks(str(root / tp))}
        Ccal, Mcal = build(splits["calibration"], by_id, attacks, rfn)
        Cte, Mte = build(splits["test"], by_id, attacks, rfn)

        rho_cal = safe_radius(lambda e: decisions(Ccal, Mcal, np.full(len(Mcal), e))[0],
                              Mcal.task.values, GRID)
        rho_te = safe_radius(lambda e: decisions(Cte, Mte, np.full(len(Mte), e))[0],
                             Mte.task.values, GRID)
        Xc, Xt = design(Mcal), design(Mte)
        z = np.array([rho_cal[t] > 1e-9 for t in Xc.index]).astype(int)
        mk = lambda: GradientBoostingClassifier(random_state=0, max_depth=3,
                                                n_estimators=300)
        g_cal = crossfit_scores(Xc, z, mk, folds=args.folds)
        g_te = mk().fit(Xc, z).predict_proba(Xt)[:, 1]

        def outcome(C, M, X, scores, tau, eps):
            e = pd.Series(apply_gate(scores, tau, eps), index=X.index)
            h, v = decisions(C, M, M.task.map(e).values)
            return task_risk(M.task.values, h, uniform=True).mean(), v.mean()

        risk_of = lambda tau, eps: outcome(Ccal, Mcal, Xc, g_cal, tau, eps)[0]
        util_of = lambda tau, eps: outcome(Ccal, Mcal, Xc, g_cal, tau, eps)[1]
        n = len(Xc)

        ev = np.array([by_id[t].clean_signal["quality_evidence"] for t in Mte.task])
        evc = np.array([by_id[t].clean_signal["quality_evidence"] for t in Mcal.task])
        print(f"\n=== {env} ===  {n} calibration tasks, "
              f"attackable {z.mean():.1%}; test attackable "
              f"{np.mean([rho_te[t] > 1e-9 for t in Xt.index]):.1%}")
        print(f"  {'rule':30s} {'tau':>5s} {'eps':>6s} {'R|static':>9s} "
              f"{'R|best-resp':>12s} {'valid':>7s}")

        for scalar_only, label in [(True, "scalar radius (Sec. 5.7)"),
                                   (False, "attackability-gated (ours)")]:
            pick = certify_gate(risk_of, util_of, n, args.alpha, args.delta,
                                taus=[] if scalar_only else TAU_GRID,
                                offsets=OFFSET_GRID, include_scalar=scalar_only)
            if pick is None:
                print(f"  {label:30s}   not certifiable at this sample size")
                rows.append(dict(environment=env, rule=label, certified=False))
                continue
            tau, eps = pick
            r_s, v_s = outcome(Cte, Mte, Xt, g_te, tau, eps)
            e_te = pd.Series(apply_gate(g_te, tau, eps), index=Xt.index)
            pol = best_response(decisions(Ccal, Mcal, Mcal.task.map(
                pd.Series(apply_gate(g_cal, tau, eps), index=Xc.index)).values)[0],
                evc, Mcal.attack.values)
            m = policy_mask(pol, ev, Mte.attack.values)
            Mb = Mte[m].reset_index(drop=True)
            h_b, _ = decisions(Cte[m], Mb, Mb.task.map(e_te).values)
            r_b = task_risk(Mb.task.values, h_b, uniform=False).mean()
            mark = lambda x: "!" if x > args.alpha else " "
            print(f"  {label:30s} {('--' if tau is None else f'{tau:.2f}'):>5s} "
                  f"{eps:6.3f} {r_s:9.4f}{mark(r_s)}{r_b:11.4f}{mark(r_b)}{v_s:7.4f}")
            rows.append(dict(environment=env, rule=label, certified=True, tau=tau,
                             eps_hi=eps, risk_static=r_s, risk_best_response=r_b,
                             valid=v_s))

        eo = pd.Series([0.6 if rho_te[t] > 1e-9 else 0.0 for t in Xt.index],
                       index=Xt.index)
        h, v = decisions(Cte, Mte, Mte.task.map(eo).values)
        r = task_risk(Mte.task.values, h, uniform=True).mean()
        print(f"  {'ORACLE gate (not deployable)':30s} {'--':>5s} {0.6:6.3f} "
              f"{r:9.4f} {'':11s}{v.mean():7.4f}")
        rows.append(dict(environment=env, rule="oracle gate", certified=True,
                         risk_static=r, valid=float(v.mean())))

    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
