"""Render the study's figures from processed results.

Figures are only drawn from data that exists; a missing input is skipped with a
message rather than filled with a placeholder that could be mistaken for a
result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

ARM_LABEL = {
    "claimed_eu": "Claimed-dist. EU", "conservative": "Conservative price",
    "verify_then_decide": "Verify-then-decide", "dro_fixed": "Fixed-$\\epsilon$ DRO",
    "robust_gate": "Adaptive gate (ours)", "oracle": "Oracle",
    "llm_direct": "Direct LLM (A)", "llm_safety": "LLM+safety (A)",
    "llm_direct_gemma4": "Direct LLM (gemma4)", "llm_safety_gemma4": "LLM+safety (gemma4)",
    "llm_direct_gptoss": "Direct LLM (gpt-oss)", "llm_safety_gptoss": "LLM+safety (gpt-oss)",
    "llm_gate": "LLM appraisal + gate", "llm_gate_urgency_coupled": "  $-$ urgency decoupl.",
}
ATTACK_LABEL = {
    "clean": "clean", "state_misrepresentation": "state\nmisrep.",
    "selective_disclosure": "selective\ndisclosure", "temporal_pressure": "urgency",
    "contract_complexity": "contract\ncomplexity", "tail_incentive": "tail\nincentive",
    "combined": "combined",
}
ATTACK_ORDER = ["clean", "state_misrepresentation", "selective_disclosure",
                "temporal_pressure", "contract_complexity", "tail_incentive",
                "combined"]


def fig_frontier(gc: dict, out: Path) -> None:
    """Safety--utility frontier: adaptive vs fixed radius, on calibration."""
    ad = pd.DataFrame(gc["frontier_adaptive"]).sort_values("valid")
    fx = pd.DataFrame(gc["frontier_fixed"]).sort_values("valid")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(fx.valid, fx.HAR, "o-", ms=3, lw=1.2, label="fixed $\\epsilon$",
            color="#888888")
    ax.plot(ad.valid, ad.HAR, "s-", ms=3, lw=1.2, label="adaptive $\\epsilon(m)$",
            color="#1f77b4")
    ax.set_xlabel("Valid acceptance $\\uparrow$")
    ax.set_ylabel("Harmful acceptance $\\downarrow$")
    ax.set_title("Safety--utility frontier (calibration)")
    ax.legend(frameon=False)
    fig.savefig(out / "fig_frontier.pdf")
    plt.close(fig)


def fig_attack(summary: pd.DataFrame, out: Path) -> None:
    # Never pool environments: they share arm names but differ in
    # reservation, attack set and scale.
    if "environment" in summary.columns:
        summary = summary[summary.environment == "CraigslistBargain"]
    piv = summary.pivot_table(index="attack_id", columns="buyer_id",
                              values="harmful_accept")
    order = [a for a in ATTACK_ORDER if a in piv.index]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    x = np.arange(len(order))
    for b in [c for c in ["claimed_eu", "dro_fixed", "conservative",
                          "robust_gate", "verify_then_decide", "oracle"]
              if c in piv.columns]:
        ax.plot(x, piv[b].values, "o-", ms=3, lw=1.2, label=ARM_LABEL.get(b, b))
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABEL[a] for a in order])
    ax.set_ylabel("Harmful acceptance $\\downarrow$")
    ax.set_title("Harmful acceptance by attack family (revenue-neutral attacks)")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out / "fig_attack_har.pdf")
    plt.close(fig)


def fig_ablation(abl: pd.DataFrame, out: Path) -> None:
    d = abl.sort_values("regret")
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    y = np.arange(len(d))
    ax.barh(y, d.HAR.values, color="#c44e52", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d.variant.values)
    ax.invert_yaxis()
    ax.set_xlabel("Harmful acceptance $\\downarrow$")
    ax.set_title("Mechanism ablations")
    for i, (h, r) in enumerate(zip(d.HAR.values, d.regret.values)):
        ax.text(h + 0.001, i, f"regret {r:.4f}", va="center", fontsize=6)
    fig.savefig(out / "fig_ablation.pdf")
    plt.close(fig)


def fig_llm(summary: pd.DataFrame, out: Path) -> None:
    """Acceptance shift under each attack for the LLM arms.

    Urgency is the interesting column: it is economically inert, so any lift is
    pure framing susceptibility and is exactly what urgency decoupling removes.
    """
    # Never pool environments: they share arm names but differ in
    # reservation, attack set and scale.
    if "environment" in summary.columns:
        summary = summary[summary.environment == "CraigslistBargain"]
    arms = [b for b in ["llm_direct", "llm_direct_gemma4", "llm_direct_gptoss",
                        "llm_gate"]
            if b in set(summary.buyer_id)]
    if not arms:
        return
    piv = summary[summary.buyer_id.isin(arms)].pivot_table(
        index="attack_id", columns="buyer_id", values="accepted")
    order = [a for a in ATTACK_ORDER if a in piv.index]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(order))
    w = 0.8 / len(arms)
    for i, b in enumerate(arms):
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, piv[b].values, w,
               label=ARM_LABEL.get(b, b))
    if "clean" in piv.index:
        for i, b in enumerate(arms):
            ax.axhline(piv.loc["clean", b], ls="--", lw=0.7,
                       color=f"C{i}", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABEL[a] for a in order])
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Model-based buyer acceptance by attack (dashed = own clean baseline)")
    ax.legend(frameon=False, ncol=3, fontsize=6)
    fig.savefig(out / "fig_llm_attack.pdf")
    plt.close(fig)


def fig_competence(disc: pd.DataFrame, rq1: pd.DataFrame, out: Path) -> None:
    """Manipulability against economic competence.

    The single most informative cross-model plot: how far an attack can move an
    arm's harmful acceptance, versus how well that arm tells a good deal from a
    bad one in the first place. A buyer that cannot price the honest offer
    cannot be steered by a dishonest one.
    """
    eff = (rq1[rq1.metric == "harmful_accept"]
           .groupby("buyer_id")["diff"].max().rename("max_effect"))
    d = disc.set_index("buyer_id").join(eff).dropna(subset=["max_effect"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for b, r in d.iterrows():
        is_llm = b.startswith("llm")
        ax.scatter(r.discrimination_gap, r.max_effect, s=34,
                   color="#c44e52" if is_llm else "#4c72b0",
                   marker="o" if is_llm else "s", zorder=3)
        ax.annotate(ARM_LABEL.get(b, b), (r.discrimination_gap, r.max_effect),
                    fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Economic competence\n(accept | good deal $-$ accept | bad deal, clean offers)")
    ax.set_ylabel("Manipulability\n(largest increase in harmful acceptance)")
    ax.set_title("A buyer must be competent to be worth manipulating")
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(out / "fig_competence.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="outputs/processed")
    ap.add_argument("--output", default="outputs/figures")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    proc, out = root / args.processed, root / args.output
    out.mkdir(parents=True, exist_ok=True)

    gp = root / args.gate_config
    if gp.exists():
        fig_frontier(json.loads(gp.read_text()), out)
        print("  fig_frontier.pdf")
    if (proc / "summary.csv").exists():
        s = pd.read_csv(proc / "summary.csv")
        fig_attack(s, out)
        print("  fig_attack_har.pdf")
        fig_llm(s, out)
    if (proc / "ablations.csv").exists():
        fig_ablation(pd.read_csv(proc / "ablations.csv"), out)
        print("  fig_ablation.pdf")
    if (proc / "discrimination.csv").exists() and (proc / "rq1_attack_effects.csv").exists():
        fig_competence(pd.read_csv(proc / "discrimination.csv"),
                       pd.read_csv(proc / "rq1_attack_effects.csv"), out)
        print("  fig_competence.pdf")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
