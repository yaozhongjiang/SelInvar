"""Render LaTeX tables from processed results.

Tables are written to outputs/tables/ with the label names the write-up
already uses, so `main.tex` can \\input them in place of its \\TBD blocks.

A table is emitted only when its inputs exist. Missing inputs produce a visible
"not yet run" row rather than a plausible-looking placeholder number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARM_LABEL = {
    "claimed_eu": "Claimed-distribution EU",
    "conservative": "Conservative price rule",
    "verify_then_decide": "Verify-then-decide",
    "dro_fixed": "Fixed-$\\epsilon$ DRO",
    "robust_gate": "Adaptive robust gate (ours)",
    "oracle": "Oracle (hidden state)",
    "llm_direct": "Direct LLM buyer",
    "llm_safety": "LLM + safety prompt",
    "llm_gate": "LLM appraisal + gate (ours, hybrid)",
    "llm_gate_urgency_coupled": "\\quad $-$ urgency decoupling",
    "llm_direct_gemma4": "Direct LLM buyer (gemma4)",
    "llm_safety_gemma4": "LLM + safety prompt (gemma4)",
    "llm_direct_gptoss": "Direct LLM buyer (gpt-oss)",
    "llm_safety_gptoss": "LLM + safety prompt (gpt-oss)",
    "llm_direct_deepseek": "Direct LLM buyer (deepseek-v4)",
    "llm_safety_deepseek": "LLM + safety prompt (deepseek-v4)",
    "llm_direct_devstral": "Direct LLM buyer (devstral-2)",
    "llm_safety_devstral": "LLM + safety prompt (devstral-2)",
}
ATTACK_LABEL = {
    "clean": "Clean", "state_misrepresentation": "State misrep.",
    "selective_disclosure": "Selective disc.", "temporal_pressure": "Urgency",
    "contract_complexity": "Contract cplx.", "tail_incentive": "Tail incentive",
    "combined": "Combined",
}
ATTACK_ORDER = ["clean", "state_misrepresentation", "selective_disclosure",
                "temporal_pressure", "contract_complexity", "tail_incentive",
                "combined"]
# Table 1 carries the decision rules; the model-based arms are a different kind
# of object (one model family each, on a subsample) and get their own table.
SCRIPTED_ORDER = ["claimed_eu", "conservative", "verify_then_decide",
                  "dro_fixed", "robust_gate", "oracle"]
# Ordered by discrimination gap, high to low, so the table reads against the
# competence ordering it falsifies: deepseek and devstral are adjacent in
# competence and opposite in vulnerability.
MODEL_ORDER = ["llm_direct_gptoss", "llm_safety_gptoss",
               "llm_direct_gemma4", "llm_safety_gemma4",
               "llm_direct_deepseek", "llm_safety_deepseek",
               "llm_direct_devstral", "llm_safety_devstral",
               "llm_direct", "llm_safety",
               "llm_gate", "llm_gate_urgency_coupled"]
ARM_ORDER = SCRIPTED_ORDER + MODEL_ORDER


def wrap(body: str, caption: str, label: str, star: bool = False,
         size: str = "small") -> str:
    env = "table*" if star else "table"
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\{size}\n{body}"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


def not_run(caption: str, label: str) -> str:
    body = ("\\begin{tabular}{@{}l@{}}\n\\toprule\nStatus\\\\\n\\midrule\n"
            "Not yet run\\\\\n\\bottomrule\n\\end{tabular}\n")
    return wrap(body, caption, label)


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# env -> {kept arm: (label, *arms folded into it)}
MERGED_ROWS = {
    "DealOrNoDeal": {
        "claimed_eu": (r"Claimed-dist. EU $\equiv$ conservative price rule",
                       "claimed_eu", "conservative"),
        "robust_gate": (r"Adaptive robust gate (ours) $\equiv$ fixed-$\epsilon$ DRO",
                        "robust_gate", "dro_fixed"),
    },
}

ENV_PANEL = [
    ("CraigslistBargain",
     r"\emph{CraigslistBargain} --- 2{,}605 held-out tasks, 7 attack families, "
     r"sampled latent quality"),
    ("AmazonHistoryPrice",
     r"\emph{AmazonHistoryPrice} --- 463 held-out products, 7 attack families, "
     r"outside option is the recorded historical average price"),
    ("DealOrNoDeal",
     r"\emph{Deal or No Deal} --- 1{,}273 held-out tasks, 5 attack families, "
     r"oracle read from the corpus"),
]


def table_main(arms: pd.DataFrame, only=None, label="tab:main",
               extra_caption="") -> str:
    """One panel per environment.

    The environments differ in reservation utility, attack set and scale, so
    their rows are never pooled and never averaged; the panel header says what
    changes between them. ``only`` restricts the panels, which is how the
    primary environment stays in the body and the replications move to the
    appendix under their own label -- the numbers are identical either way.
    """
    if "environment" not in arms.columns:
        arms = arms.assign(environment="CraigslistBargain")
    panels = [(e, h) for e, h in ENV_PANEL if only is None or e in only]
    rows = []
    for k, (env, header) in enumerate(panels):
        sub = arms[arms.environment == env]
        if not len(sub):
            continue
        if k:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{6}{@{}l}{" + header + r"}\\[1pt]")
        # Arms that are provably identical in this environment are shown as one
        # row: printing them twice invites the reader to look for a difference
        # that cannot exist.
        merged = MERGED_ROWS.get(env, {})
        skip = {b for k, pair in merged.items() for b in pair[1:] if b != k}
        for a in SCRIPTED_ORDER:
            if a in skip:
                continue
            r = sub[sub.buyer_id == a]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append(
                r"\quad " + f"{merged.get(a, (ARM_LABEL[a],))[0]} & {int(r.n_tasks)} & "
                f"{fmt(r.harmful_accept)} [{fmt(r.harmful_accept_lo)}, "
                f"{fmt(r.harmful_accept_hi)}] & {fmt(r.valid_accept)} & "
                f"{fmt(r.missed_opportunity)} & {fmt(r.regret_norm, 4)}" + r"\\")
    body = (r"\begin{tabular}{@{}lrccc c@{}}" "\n" r"\toprule" "\n"
            r"Buyer policy & $n$ & HAR $\downarrow$ [95\% CI] & Valid acc. $\uparrow$ "
            r"& Missed opp. $\downarrow$ & Regret $\downarrow$\\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    # The caption has to match the panels actually printed: a single-panel table
    # must not talk about differences between panels, and only the Deal or No
    # Deal panel carries the "provably coincide" rows.
    multi = len(panels) > 1
    has_merged = any(MERGED_ROWS.get(e) for e, _ in panels)
    cap = ("Main results on held-out test tasks"
           + (", one panel per environment. " if multi else ". ")
           + "HAR is harmful acceptance; regret is normalized by the listing "
             "price (by the 10-point pie in Deal or No Deal). Intervals are "
             "task-level bootstrap CIs. ")
    if multi:
        cap += ("The panels differ in reservation utility, attack set and scale, "
                "so they are reported separately and never pooled. ")
    if has_merged:
        cap += (r"Rows marked $\equiv$ are arms that provably coincide in that "
                "environment. ")
    cap += "The model-based buyers are reported separately in Table~\\ref{tab:llm}."
    return wrap(body, cap + extra_caption, label, star=True, size="footnotesize")


def table_llm(arms: pd.DataFrame) -> str:
    """Model-based buyers, which exist only on the primary environment.

    Kept apart from Table 1 because they are a different kind of object: each row
    is one model family evaluated on a stratified subsample, so the $n$ differ
    from the scripted arms and from each other.
    """
    if "environment" not in arms.columns:
        arms = arms.assign(environment="CraigslistBargain")
    sub = arms[arms.environment == "CraigslistBargain"]
    rows = []
    for a in MODEL_ORDER:
        r = sub[sub.buyer_id == a]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append(
            f"{ARM_LABEL[a]} & {int(r.n_tasks)} & "
            f"{fmt(r.harmful_accept)} [{fmt(r.harmful_accept_lo)}, "
            f"{fmt(r.harmful_accept_hi)}] & {fmt(r.valid_accept)} & "
            f"{fmt(r.missed_opportunity)} & {fmt(r.regret_norm, 4)}" + r"\\")
    body = (r"\begin{tabular}{@{}lrccc c@{}}" "\n" r"\toprule" "\n"
            r"Model-based buyer & $n$ & HAR $\downarrow$ [95\% CI] & "
            r"Valid acc. $\uparrow$ & Missed opp. $\downarrow$ & Regret $\downarrow$\\"
            "\n" r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n"
            r"\end{tabular}" "\n")
    return wrap(body,
                "Model-based buyers on the CraigslistBargain test subsample, ordered "
                "by decreasing economic competence (acceptance given a good deal minus "
                "acceptance given a bad one, on the honest rendering). Each row is one "
                "model family, so the $n$ differ from Table~\\ref{tab:main} and from "
                "each other; arms are paired on the task wherever they are compared. "
                "The two adjacent middle families, deepseek-v4 and devstral-2, agree "
                "on competence to $0.010$ and differ by $+0.225$ $[+0.160,+0.287]$ in "
                "maximum attack effect, which is what falsifies the competence "
                "ordering; a difference is reported rather than a ratio because one "
                "arm's effect is too near zero to divide by. The last row is the "
                "hybrid arm with urgency decoupling switched off.",
                "tab:llm", star=True, size="footnotesize")


def table_attack(summary: pd.DataFrame) -> str:
    """Harmful acceptance by attack family, one panel per environment.

    Columns are the union of the two attack sets. Deal or No Deal has no price
    and no contract, so `contract_complexity` and `tail_incentive` have no
    channel there and are printed as `--` rather than as zeros -- a zero would
    read as "the attack was tried and failed", which is not what happened.
    """
    if "environment" not in summary.columns:
        summary = summary.assign(environment="CraigslistBargain")
    cols = [c for c in ATTACK_ORDER if c in set(summary.attack_id)]
    rows = []
    for k, (env, header) in enumerate(ENV_PANEL):
        sub = summary[summary.environment == env]
        if not len(sub):
            continue
        piv = sub.pivot_table(index="buyer_id", columns="attack_id",
                              values="harmful_accept")
        if k:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{@{}l}{"
                    + header + r"}\\[1pt]")
        merged = MERGED_ROWS.get(env, {})
        skip = {b for kk, pair in merged.items() for b in pair[1:] if b != kk}
        for a in ARM_ORDER:
            if a in skip or a not in piv.index:
                continue
            vals = " & ".join(
                fmt(piv.loc[a, c]) if c in piv.columns and pd.notna(piv.loc[a, c])
                else "--" for c in cols)
            label = merged.get(a, (ARM_LABEL[a],))[0]
            rows.append(r"\quad " + f"{label} & {vals}" + r"\\")
    head = " & ".join(ATTACK_LABEL[c] for c in cols)
    body = (r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}" "\n" r"\toprule" "\n"
            f"Buyer policy & {head}" + r"\\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    return wrap(body,
                "Harmful acceptance by attack family, one panel per environment. The "
                "clean column is the same latent state rendered honestly, and every "
                "attack is revenue-neutral, so differences along a row are decision "
                "distortion rather than a better deal. `--' marks an attack family "
                "with no channel in that environment: Deal or No Deal has no price "
                "and no contract. Rows marked $\\equiv$ are arms that provably "
                "coincide there. Panels are never pooled.",
                "tab:attack", star=True, size="footnotesize")


def table_ablation(abl: pd.DataFrame) -> str:
    """One panel per environment.

    The Deal or No Deal panel is deliberately uniform: every uncertainty score
    is constant across messages there, so a variant that changes a coefficient
    changes only the radius magnitude, never where uncertainty is placed. The
    identical rows are the evidence that no mechanism has a channel in that
    environment -- reported rather than suppressed.
    """
    if "environment" not in abl.columns:
        abl = abl.assign(environment="CraigslistBargain")
    rows = []
    for k, (env, header) in enumerate(ENV_PANEL):
        sub = abl[abl.environment == env]
        if not len(sub):
            continue
        if k:
            rows.append(r"\midrule")
        short = header.split("---")[0].strip()
        rows.append(r"\multicolumn{6}{@{}l}{" + short + r"}\\[1pt]")
        # If every variant coincides, say so once. Printing seven identical rows
        # invites the reader to hunt for a difference that cannot exist.
        metrics = ["HAR", "valid_accept", "missed_opportunity", "regret"]
        if len(sub) > 1 and sub[metrics].round(6).drop_duplicates().shape[0] == 1:
            r = sub.iloc[0]
            lo, hi = sub.mean_radius.min(), sub.mean_radius.max()
            rows.append(
                r"\quad all " + str(len(sub)) + r" variants coincide & "
                + f"{fmt(r.HAR)} & {fmt(r.valid_accept)} & "
                f"{fmt(r.missed_opportunity)} & {fmt(r.regret, 4)} & "
                f"{fmt(lo)}--{fmt(hi)}" + r"\\")
            continue
        for _, r in sub.iterrows():
            name = r.variant
            if name.startswith("- "):
                name = r"$-$ " + name[2:]
            rows.append(
                r"\quad " + f"{name} & {fmt(r.HAR)} & {fmt(r.valid_accept)} & "
                f"{fmt(r.missed_opportunity)} & {fmt(r.regret, 4)} & "
                f"{fmt(r.mean_radius)}" + r"\\")
    body = (r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n"
            r"Variant & HAR $\downarrow$ & Valid acc. $\uparrow$ & Missed opp. "
            r"$\downarrow$ & Regret $\downarrow$ & $\bar{\epsilon}$\\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n"
            r"\end{tabular}" "\n")
    return wrap(body,
                "Mechanism ablations and controls, one panel per environment. Urgency "
                "decoupling is inert for scripted policies, which never read the "
                "urgency channel; its necessity is evidenced by the model-based arms. "
                "`Evidence uninformative' is the boundary condition in which "
                "verification metadata is independent of manipulation. Every Deal or No "
                "Deal row coincides because that environment has no contract and no "
                "verification channel, so the uncertainty scores are constant and only "
                "the radius magnitude can move---which it does, over $0.018$--$0.030$, "
                "without changing any decision.",
                "tab:ablation", star=True, size="footnotesize")


def table_conformal(conf: pd.DataFrame) -> str:
    """Coverage and cost of the conformalized radius, per environment.

    Coverage is the fraction of messages whose true state lies inside the
    ambiguity set -- i.e. how often assumption A1 actually holds. It is measured
    on every message, not on the subset the gate accepted.
    """
    order = ["paper: additive eps(m)", "learned eps(m), uncalibrated",
             "conformal eps(m), alpha=0.20", "conformal eps(m), alpha=0.10",
             "conformal eps(m), alpha=0.05", "ORACLE radius (not deployable)"]
    label = {
        "study: additive eps(m)": r"additive $\epsilon(m)$ (Sec.~4.5)",
        "learned eps(m), uncalibrated": r"learned $\epsilon(m)$, uncalibrated",
        "conformal eps(m), alpha=0.20": r"\quad + conformal, $\alpha{=}0.20$",
        "conformal eps(m), alpha=0.10": r"\quad + conformal, $\alpha{=}0.10$",
        "conformal eps(m), alpha=0.05": r"\quad + conformal, $\alpha{=}0.05$",
        "ORACLE radius (not deployable)": r"oracle radius (not deployable)",
    }
    envs = [e for e, _ in ENV_PANEL if e in set(conf.environment)]
    rows = []
    for m in order:
        cells = []
        for e in envs:
            r = conf[(conf.environment == e) & (conf.method.str.startswith(m.split(" (")[0]))]
            if not len(r):
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            cells += [f"{r.coverage:.1%}".replace("%", r"\%"), fmt(r.HAR)]
        rows.append(f"{label[m]} & " + " & ".join(cells) + r"\\")
    head = " & ".join(r"\multicolumn{2}{c}{" + e.replace("CraigslistBargain", "Craigslist")
                      .replace("AmazonHistoryPrice", "AmazonHistory")
                      .replace("DealOrNoDeal", "Deal or No Deal") + "}" for e in envs)
    sub = " & ".join([r"cov.\ $\uparrow$ & HAR $\downarrow$"] * len(envs))
    body = (r"\begin{tabular}{@{}l" + "cc" * len(envs) + r"@{}}" "\n" r"\toprule" "\n"
            f"Radius rule & {head}" + r"\\" "\n"
            f" & {sub}" + r"\\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    return wrap(body,
                "Coverage of the ambiguity set --- how often assumption A1 actually "
                "holds --- and the harmful acceptance that follows. The study's "
                "additive radius under-covers on all three environments, so the "
                "safety property's precondition fails on a quarter to a third of "
                "messages. Split-conformal correction restores nominal coverage from "
                "calibration data alone. Coverage is measured on every message, never "
                "on the subset the gate accepted, which would inflate it.",
                "tab:conformal", star=True, size="footnotesize")


def table_strategic(cert: pd.DataFrame) -> str:
    """Whether a calibrated certificate survives a seller that best-responds to it.

    The two rows per level draw radii from one parametric family, so this is not
    a frontier comparison: it is a certificate that holds against one that does
    not, and the utility column is the price of the difference.
    """
    c = cert[cert.certified == True]                                   # noqa: E712
    envs = [e for e, _ in ENV_PANEL if e in set(c.environment)]
    rows = []
    for k, e in enumerate(envs):
        sub = c[c.environment == e]
        if not len(sub):
            continue
        if k:
            rows.append(r"\midrule")
        short = e.replace("CraigslistBargain", "Craigslist") \
                 .replace("AmazonHistoryPrice", "AmazonHistory") \
                 .replace("DealOrNoDeal", "Deal or No Deal")
        rows.append(r"\multicolumn{6}{@{}l}{" + short + r"}\\[1pt]")
        for a in sorted(sub.alpha.unique(), reverse=True):
            for lab, name in [("mixture", r"mixture"), ("worst-case", r"worst-case")]:
                r = sub[(sub.alpha == a) & (sub.certified_on == lab)]
                if not len(r):
                    continue
                r = r.iloc[0]
                # The violated cell is the whole point of the table, so it is
                # marked rather than left for the reader to compare against alpha.
                br = fmt(r.risk_best_response)
                if bool(r.violated):
                    br = r"\bf " + br + r"$^{\dagger}$"
                rows.append(r"\quad $\alpha{=}" + f"{a:.2f}$, {name} & "
                            f"{fmt(r.offset)} & {fmt(r.risk_mixture)} & {br} & "
                            f"{fmt(r.risk_worst_case)} & {fmt(r.valid_best_response)}"
                            + r"\\")
    body = (r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n"
            r"Certified on & $\hat\epsilon$ & $R\vert$mix. $\downarrow$ & "
            r"$R\vert$best-resp. $\downarrow$ & $R\vert$worst $\downarrow$ & "
            r"Valid acc. $\uparrow$\\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    n_mix = int(c[c.certified_on == "mixture"].violated.sum())
    n_wc = int(c[c.certified_on == "worst-case"].violated.sum())
    tot = len(c[c.certified_on == "mixture"])
    return wrap(body,
                "Certifying the decision risk on the calibration mixture---the "
                "exchangeability assumption every conformal calibration of an "
                "ambiguity set relies on---does not bound the risk a seller that "
                "best-responds to the published radius actually inflicts "
                f"($\\dagger$: {n_mix} of {tot} levels violated). Maximizing over the "
                "seller's action set per task before aggregating dominates every "
                f"seller policy over that set, and is violated at {n_wc} of {tot}. "
                "Both rows draw the radius from one family, so the difference is "
                "validity, not sharpness; the last column prices it. Where no policy "
                "can exploit the gate the two certificates select the same radius and "
                "the bound is free. Levels at which no offset certifies from the "
                "available calibration tasks are omitted.",
                "tab:strategic", star=True, size="footnotesize")


def table_stats(rq2: pd.DataFrame) -> str:
    rows = []
    for m, lab in [("harmful_accept", "HAR"), ("valid_accept", "Valid acc."),
                   ("regret_norm", "Regret")]:
        s = rq2[rq2.metric == m]
        for _, r in s.iterrows():
            if r.other == "oracle":
                continue
            star = "$^{*}$" if r.reject_holm else ""
            rows.append(f"{lab} & {ARM_LABEL.get(r.other, r.other)} & "
                        f"{r['diff']:+.4f} & [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & "
                        f"{r.p_value:.4f}{star}\\\\")
    body = ("\\begin{tabular}{@{}llccc@{}}\n\\toprule\n"
            "Metric & Compared with & $\\Delta$ (ours $-$ other) & 95\\% CI & $p$\\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return wrap(body, "Paired task-level comparisons against the adaptive robust gate. "
                "Differences are bootstrapped over tasks; $p$ from a sign-flip "
                "permutation test. $^{*}$ survives Holm--Bonferroni at $\\alpha=0.05$.",
                "tab:stats", star=True)


def table_calibration(gc: dict) -> str:
    a, f = gc["adaptive"], gc["fixed"]
    rows = [
        f"$\\epsilon_0$ & {a['eps0']:.4f} & {f['radius']:.4f} (fixed $\\epsilon$)\\\\",
        f"$\\lambda_x$ (state) & {a['lambda_x']:.4f} & --\\\\",
        f"$\\lambda_\\phi$ (contract) & {a['lambda_phi']:.4f} & --\\\\",
        f"$\\lambda_s$ (seller) & {a['lambda_s']:.4f} & --\\\\",
        "\\midrule",
        f"Calibration regret & {a['regret']:.5f} & {f['regret']:.5f}\\\\",
        f"Calibration HAR & {a['HAR']:.4f} & {f['HAR']:.4f}\\\\",
        f"Calibration valid acc. & {a['valid']:.4f} & {f['valid']:.4f}\\\\",
    ]
    body = ("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
            "Parameter & Adaptive & Fixed\\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    n = gc.get("matched_points", 0)
    w = gc.get("adaptive_wins", 0)
    return wrap(body, f"Gate coefficients fitted on the calibration split "
                f"({gc['n_tasks']} tasks) and frozen before test evaluation. The "
                f"adaptive family contains the fixed one at $\\lambda=0$. At matched "
                f"valid-acceptance the adaptive radius is strictly better at {w} of "
                f"{n} operating points.", "tab:calibration")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="outputs/processed")
    ap.add_argument("--output", default="outputs/tables")
    ap.add_argument("--gate-config", default="outputs/gate_config.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    proc, out = root / args.processed, root / args.output
    out.mkdir(parents=True, exist_ok=True)

    def maybe(name, fn, caption, label):
        p = proc / name
        if not p.exists():
            print(f"  {label}: input {name} missing -> 'not yet run'")
            return not_run(caption, label)
        return fn(pd.read_csv(p))

    ap_ = proc / "arm_summary.csv"
    if ap_.exists():
        A = pd.read_csv(ap_)
        (out / "main_results.tex").write_text(
            table_main(A, only={"CraigslistBargain"},
                       extra_caption=" The two replication environments are "
                       "Table~\\ref{tab:main-extra}."))
        (out / "main_results_extra.tex").write_text(
            table_main(A, only={"AmazonHistoryPrice", "DealOrNoDeal"},
                       label="tab:main-extra",
                       extra_caption=" These are the replication environments for "
                       "Table~\\ref{tab:main}; the primary benchmark is reported there."))
    else:
        (out / "main_results.tex").write_text(not_run("Main results.", "tab:main"))
        (out / "main_results_extra.tex").write_text(
            not_run("Replication environments.", "tab:main-extra"))
    (out / "llm_results.tex").write_text(
        maybe("arm_summary.csv", table_llm, "Model-based buyers.", "tab:llm"))
    (out / "attack_results.tex").write_text(
        maybe("summary.csv", table_attack, "Attack breakdown.", "tab:attack"))
    (out / "ablation_results.tex").write_text(
        maybe("ablations.csv", table_ablation, "Mechanism ablation.", "tab:ablation"))
    cp = proc / "conformal.csv"
    (out / "conformal_results.tex").write_text(
        table_conformal(pd.read_csv(cp)) if cp.exists()
        else not_run("Conformal coverage.", "tab:conformal"))
    sp = proc / "strategic_certificate.csv"
    (out / "strategic_results.tex").write_text(
        table_strategic(pd.read_csv(sp)) if sp.exists()
        else not_run("Strategic certification.", "tab:strategic"))
    (out / "statistical_tests.tex").write_text(
        maybe("rq2_arm_comparison.csv", table_stats, "Paired comparisons.", "tab:stats"))

    gp = root / args.gate_config
    (out / "calibration_results.tex").write_text(
        table_calibration(json.loads(gp.read_text())) if gp.exists()
        else not_run("Gate calibration.", "tab:calibration"))

    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
