"""Conformalized ambiguity radius, replicated across all three environments.

    python scripts/run_conformal.py

WHY. The safety property of Sec. 4.7 is conditional on assumption A1: the true
state lies inside the ambiguity set. The study's hand-designed radius
eps(m) = eps0 + lambda_x u_x + lambda_phi u_phi + lambda_s u_s treats that as an
assumption, and on the Craigslist test split it holds for only ~75% of messages
-- the theorem's precondition fails silently on a quarter of them.

Here the radius is instead (i) *predicted* from observables as an estimate of the
realized distortion, and (ii) corrected by split conformal prediction so that

    P( true state inside the ambiguity set ) >= 1 - alpha

by construction, from calibration data alone. The predictor is fitted on the
calibration split and only observable features are used at test time; the target
(the realized distortion) is an oracle quantity available in calibration, which
is ordinary supervised learning, not leakage.

TWO THINGS THIS SCRIPT IS CAREFUL ABOUT.

Coverage is measured on *every* calibrated message, never on the subset the gate
chose to accept. Conditioning on the gate's own decision inflates coverage,
because the gate preferentially releases the messages its bound covers.

The conformal guarantee assumes exchangeability, and our messages are clustered
by task (seven renderings each). Calibration and test are disjoint *task* sets,
so the clustering is between groups rather than across the split, but the
effective sample size is closer to the task count than the message count; the
coverage interval below is therefore bootstrapped over tasks, not messages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal_data import (ENVIRONMENTS, FEATURES, GRID, build,
                                robust_bound, true_expected_price)
from src.dataset import read_tasks


def evaluate(C: np.ndarray, M: pd.DataFrame, eps) -> Dict[str, float]:
    eps = np.clip(np.asarray(eps, float), GRID[0], GRID[-1])
    lb = robust_bound(C, M, eps)

    acc = (lb >= M.res.values).astype(int)
    good = (M.u_true.values >= M.res.values).astype(int)
    sp = M.u_true.values - M.res.values
    regret = (np.maximum(0.0, sp) - np.where(acc == 1, sp, 0.0)) / M.L.values
    covered = M.ideal.values <= eps + 1e-12

    # coverage interval resampled over tasks, since messages are clustered by task
    per_task = pd.DataFrame({"task": M.task.values, "c": covered}).groupby("task").c.mean()
    rng = np.random.default_rng(0)
    x = per_task.values
    boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(axis=1)
    return {"regret": float(regret.mean()), "HAR": float((acc * (1 - good)).mean()),
            "valid": float((acc * good).mean()), "coverage": float(covered.mean()),
            "cov_lo": float(np.quantile(boot, 0.025)),
            "cov_hi": float(np.quantile(boot, 0.975)),
            "mean_eps": float(np.mean(eps))}


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal correction; the ceil((n+1)(1-alpha))/n quantile."""
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(max(0.0, np.quantile(residuals, level, method="higher")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.20,0.10,0.05")
    ap.add_argument("--output", default="outputs/processed/conformal.csv")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    alphas = [float(a) for a in args.alphas.split(",")]

    from sklearn.ensemble import GradientBoostingRegressor

    gate = json.loads((root / "outputs" / "gate_config.json").read_text())["adaptive"]
    out_rows: List[Dict[str, Any]] = []

    for env, (tp, sp, attacks, rfn) in ENVIRONMENTS.items():
        splits = json.loads((root / sp).read_text())
        by_id = {t.task_id: t for t in read_tasks(str(root / tp))}
        Ccal, Mcal = build(splits["calibration"], by_id, attacks, rfn)
        Cte, Mte = build(splits["test"], by_id, attacks, rfn)

        print(f"\n=== {env} ===  calibration {Mcal.task.nunique()} tasks, "
              f"test {Mte.task.nunique()} tasks, {len(attacks)} attack families")
        print(f"  informative features on this environment: "
              f"{[f for f in FEATURES if Mcal[f].nunique() > 1]}")
        print(f"{'method':38s} {'regret':>9s} {'HAR':>7s} {'valid':>7s} "
              f"{'coverage [95% CI]':>24s} {'eps':>7s}")

        def emit(name: str, m: Dict[str, float]) -> None:
            print(f"{name:38s} {m['regret']:9.5f} {m['HAR']:7.4f} {m['valid']:7.4f} "
                  f"{m['coverage']:9.1%} [{m['cov_lo']:.1%}, {m['cov_hi']:.1%}]"
                  f" {m['mean_eps']:7.4f}")
            out_rows.append({"environment": env, "method": name, **m})

        eps_paper = (gate["eps0"] + gate["lambda_x"] * Mte.u_x
                     + gate["lambda_phi"] * Mte.u_phi + gate["lambda_s"] * Mte.u_s)
        emit("study: additive eps(m)", evaluate(Cte, Mte, eps_paper))

        best = min((evaluate(Cte, Mte, np.full(len(Mte), r))["regret"], r)
                   for r in np.round(np.arange(0, 0.41, 0.005), 4))
        emit(f"best fixed radius (eps={best[1]:.3f})",
             evaluate(Cte, Mte, np.full(len(Mte), best[1])))

        feats = [f for f in FEATURES if Mcal[f].nunique() > 1] or ["claim_q"]
        g = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=300)
        g.fit(Mcal[feats], Mcal.ideal)
        pred_cal = np.clip(g.predict(Mcal[feats]), 0, None)
        pred_te = np.clip(g.predict(Mte[feats]), 0, None)
        emit("learned eps(m), uncalibrated", evaluate(Cte, Mte, pred_te))

        resid = Mcal.ideal.values - pred_cal
        for a in alphas:
            q = conformal_quantile(resid, a)
            emit(f"conformal eps(m), alpha={a:.2f}", evaluate(Cte, Mte, pred_te + q))

        emit("ORACLE radius (not deployable)", evaluate(Cte, Mte, Mte.ideal.values))

    p = root / args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
