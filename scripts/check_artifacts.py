"""Integrity checks on the experimental artifacts, and on the numbers the paper quotes.

    python scripts/check_artifacts.py --mode claims

WHY THIS EXISTS. Every defect found on 2026-08-20 passed the unit tests and the
submission checker, because those cover code logic and PDF hygiene while the
failures were in the *artifacts*:

  * `run_main.py --tag` did not reach the scripted output filename, so re-running
    a second environment's scripted arms silently overwrote the primary
    benchmark's results file and the headline numbers could no longer be
    reproduced;
  * two side experiments reused the main runs' `buyer_id`, so the pooling step in
    `analyze.py` mixed a seven-family run with a three-family one;
  * a pre-fix and a post-fix copy of the same environment were pooled together,
    which produced plausible-looking Deal or No Deal numbers that were written
    into the write-up twice before the contamination was noticed.

None of that is detectable from the code. It is detectable from the artifacts, so
this script asserts what the pipeline is supposed to have produced and, most
usefully, recomputes every number the write-up quotes and fails if the text and
the data disagree.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_dnd import DND_ATTACKS

ROOT = Path(__file__).resolve().parents[1]

# Files whose rows enter the pooled analysis. Anything else under raw/ must be
# prefixed with "_" so `analyze.py` skips it; a side experiment that reuses a
# main run's buyer_id is otherwise indistinguishable from more data for that arm.
POOLED = {
    "main_scripted.jsonl": "CraigslistBargain",
    "main_ahp.jsonl": "AmazonHistoryPrice",
    "main_dnd.jsonl": "DealOrNoDeal",
    "main_llm.jsonl": "CraigslistBargain",
    "main_llm_gemma4.jsonl": "CraigslistBargain",
    "main_llm_gptoss.jsonl": "CraigslistBargain",
    "main_llm_deepseek.jsonl": "CraigslistBargain",
    "main_llm_devstral.jsonl": "CraigslistBargain",
    "main_llm_gpt5mini.jsonl": "CraigslistBargain",
    "main_llm_gpt4omini.jsonl": "CraigslistBargain",
    "main_hybrid.jsonl": "CraigslistBargain",
    "main_dnd_rounds.jsonl": "DealOrNoDealRounds",
    "main_dnd_rounds_commit.jsonl": "DealOrNoDealRounds",
}

# Artifacts that are not negotiation trial records and carry their own schema.
# They are registered so the "nothing unexpected is pooled" check stays strict,
# but they are checked against their own invariants rather than an environment.
STANDALONE = {"crossdomain.jsonl", "main_improvements.jsonl",
              "ablations_boundary.jsonl"}   # own attack vocabulary
SCRIPTED = {"claimed_eu", "conservative", "verify_then_decide", "dro_fixed",
            "robust_gate", "oracle"}
MODEL_FAMILIES = {"llm_direct", "llm_direct_gemma4", "llm_direct_gptoss",
                  "llm_direct_deepseek", "llm_direct_devstral",
                  "llm_direct_gpt5mini", "llm_direct_gpt4omini"}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "processed" / name)


def arm(env: str, buyer: str, col: str) -> float:
    d = load("arm_summary.csv")
    r = d[(d.environment == env) & (d.buyer_id == buyer)]
    if not len(r):
        raise KeyError(f"{buyer} missing from {env} in arm_summary")
    return float(r[col].iloc[0])


def cell(env: str, buyer: str, attack: str, col: str) -> float:
    """One (environment, arm, attack) cell. `arm()` averages over every attack,
    which is not the clean rate and reads 0.0821 where the clean rate is 0.000."""
    d = load("summary.csv")
    r = d[(d.environment == env) & (d.buyer_id == buyer) & (d.attack_id == attack)]
    if not len(r):
        raise KeyError(f"{buyer}/{attack} missing from {env} in summary")
    return float(r[col].iloc[0])


def effect(env: str, buyer: str, attack: str) -> float:
    d = load("summary.csv")
    s = d[(d.environment == env) & (d.buyer_id == buyer)]
    a = s[s.attack_id == attack].harmful_accept
    c = s[s.attack_id == "clean"].harmful_accept
    if not len(a) or not len(c):
        raise KeyError(f"{buyer}/{attack} missing from {env} in summary")
    return float(a.iloc[0]) - float(c.iloc[0])


def prompt_defence_spread(attack: str) -> float:
    """Spread across model families of the protection a prompt defence delivers.

    Proposition `blackbox` says a defence that does not change the representation
    can only be proof for the particular model it was written against, so this
    spread must be large; the canonicalised channels are checked separately and
    are exactly zero. If a future run made these agree, the dichotomy the paper
    draws would be gone and the claim would need rewriting rather than rescaling.
    """
    d = load("summary.csv")
    d = d[(d.environment == "CraigslistBargain") & (d.attack_id == attack)]
    vals = []
    for fam in ("", "_gemma4", "_gptoss", "_deepseek", "_devstral",
                "_gpt5mini", "_gpt4omini"):
        a = d[d.buyer_id == "llm_direct" + fam].harmful_accept
        b = d[d.buyer_id == "llm_safety" + fam].harmful_accept
        if not len(a) or not len(b):
            raise KeyError(f"family {fam or 'qwen'} missing for {attack}")
        vals.append(float(a.iloc[0]) - float(b.iloc[0]))
    return max(vals) - min(vals)


def cd_harm(channel: str, defence: str, which: str = "max") -> float:
    """Harm rate for one out-of-domain cell, across the three families.

    The paper's claim there is a pattern of exact zeros, so `max` is the right
    reduction: a structural defence that leaked on a single family would be
    hidden by a mean and is the failure this is meant to catch.
    """
    d = load("crossdomain_summary.csv")
    r = d[(d.channel == channel) & (d.defence == defence)]
    if not len(r):
        raise KeyError(f"{channel}/{defence} missing from crossdomain_summary")
    if which == "min_nonzero":
        nz = r.harm[r.harm > 0]
        if not len(nz):
            raise KeyError(f"{channel}/{defence} has no non-zero family")
        return float(nz.min())
    return float(r.harm.max() if which == "max" else r.harm.min())


def curve_effect(gamma: float, buyer: str, attack: str) -> float:
    """Clean-relative effect in the utility-form sensitivity run."""
    f = ROOT / "outputs" / "raw" / "_utility_sensitivity.jsonl"
    d = pd.read_json(f, lines=True)
    d = d[(d.quality_curve == gamma) & (d.buyer_id == buyer)]
    a = d[d.attack_id == attack].harmful_accept.mean()
    c = d[d.attack_id == "clean"].harmful_accept.mean()
    return float(a - c)


def cd_price(defence: str, which: str = "max") -> float:
    """Share of honest documents whose action the defence changes.

    Only meaningful under common random numbers; with the defence in the seed
    this quantity is decoding noise. The runner's seed must therefore not depend
    on the defence, which `test_crossdomain_cell_seeds_are_stable` also guards.
    """
    d = load("crossdomain_summary.csv")
    r = d[(d.channel == "clean") & (d.defence == defence)]
    if not len(r):
        raise KeyError(f"{defence} missing from crossdomain_summary")
    return float(r.distortion.max() if which == "max" else r.distortion.min())



def _agentdojo(model: str = "gpt-4o-mini") -> pd.DataFrame:
    """The processed AgentDojo summary, so claims work from the shipped files."""
    d = load("agentdojo_summary.csv")
    return d[d.model_id == model]


def adj_exposure(model: str, column: str) -> float:
    """One cell of the exposure table: did the payload arrive, and what followed."""
    d = load("agentdojo_exposure.csv")
    r = d[d.model_id == model]
    if not len(r):
        raise KeyError(f"{model} missing from agentdojo_exposure")
    return float(r[column].iloc[0])


def prose(model: str, form: str, attack: str) -> float:
    """One cell of the prompt-form study.

    `form` is "json", "prose", or "prose_minus_json" for the paired difference
    on the tasks both forms evaluated.
    """
    d = load("prose_summary.csv")
    r = d[(d.model_id == model) & (d.form == form) & (d.attack_id == attack)]
    if not len(r):
        raise KeyError(f"{model}/{form}/{attack} missing from prose_summary")
    return float(r.effect.iloc[0])


def adj(arm: str, column: str = "attack_success",
        model: str = "gpt-4o-mini", vector: str = "all",
        suite: str = "banking") -> float:
    """AgentDojo arm mean, as their own security/utility verdicts scored it."""
    d = _agentdojo(model)
    r = d[(d.arm == arm) & (d.vector == vector) & (d.scope == "all")
          & (d.suite == suite)]
    if not len(r):
        raise KeyError(f"agentdojo {arm}/{vector} missing")
    return float(r[column].iloc[0])


def adj_channel(arm: str, on_channel: bool,
                suite: str = "banking") -> float:
    """Attack success split by whether the injection needs the closed channel.

    Eight of banking's nine injection goals require sending money to an
    attacker-controlled recipient, which is the class-two channel; the ninth
    changes a password instead and is off it. The paper claims exactness on the
    first group and inertness on the second, so a mean over all nine would hide
    both halves.
    """
    d = _agentdojo()
    scope = "on_channel" if on_channel else "off_channel"
    r = d[(d.arm == arm) & (d.vector == "all") & (d.scope == scope)
          & (d.suite == suite)]
    if not len(r):
        raise KeyError(f"agentdojo {arm}/{scope} missing")
    return float(r.attack_success.iloc[0])


def improvement_response(buyer: str, improvement: str) -> float:
    """Change in acceptance when the seller genuinely concedes money."""
    d = pd.read_json(ROOT / "outputs" / "raw" / "main_improvements.jsonl",
                     lines=True)
    pv = d.pivot_table(index="task_id", columns=["buyer_id", "attack_id"],
                       values="accepted", aggfunc="first")
    return float((pv[(buyer, improvement)] - pv[(buyer, "clean")]).mean())


def rounds_effect(buyer: str, attack: str) -> float:
    d = load("dnd_rounds_summary.csv")
    r = d[(d.environment == "DealOrNoDealRounds") & (d.buyer_id == buyer)
          & (d.attack_id == attack)]
    if not len(r):
        raise KeyError(f"{buyer}/{attack} missing from dnd_rounds_summary")
    return float(r.effect.iloc[0])


def rounds_valid(buyer: str, attack: str) -> float:
    d = load("dnd_rounds_summary.csv")
    r = d[(d.environment == "DealOrNoDealRounds") & (d.buyer_id == buyer)
          & (d.attack_id == attack)]
    if not len(r):
        raise KeyError(f"{buyer}/{attack} missing from dnd_rounds_summary")
    return float(r.valid_accept.iloc[0])


def prompt_defence_extreme(attack: str, which: str) -> float:
    d = load("summary.csv")
    d = d[(d.environment == "CraigslistBargain") & (d.attack_id == attack)]
    vals = [float(d[d.buyer_id == "llm_direct" + f].harmful_accept.iloc[0])
            - float(d[d.buyer_id == "llm_safety" + f].harmful_accept.iloc[0])
            for f in ("", "_gemma4", "_gptoss", "_deepseek", "_devstral",
                      "_gpt5mini", "_gpt4omini")]
    return max(vals) if which == "max" else min(vals)


def hybrid(metric: str) -> float:
    """Paired hybrid-vs-direct comparison on the shared 401-task subset."""
    d = load("hybrid_vs_others.csv")
    r = d[(d.other == "llm_direct") & (d.metric == metric)]
    if not len(r):
        raise KeyError(f"hybrid/{metric} missing")
    return float(r["diff"].iloc[0])


def boundary_acceptance(arm: str) -> float:
    """Acceptance shift when the completion sets are made incomparable.

    This is the failing side of Proposition 1's precondition. An earlier draft
    quoted a harmful-acceptance effect of +0.014 here that no artifact
    reproduced; what the experiment actually shows is a shift in the decision,
    with harm unchanged on this benchmark.
    """
    d = pd.read_json(ROOT / "outputs" / "raw" / "ablations_boundary.jsonl",
                     lines=True)
    g = d[d.buyer_id == arm].groupby("attack_id").accepted.mean()
    return float(g["incomparable_swap"] - g["clean"])


def _unused_boundary_effect() -> float:
    """Contract-channel effect once the completion sets are made incomparable.

    This is the failing side of Proposition 1's precondition, so it is the one
    number in the paper whose value is supposed to be non-zero under closure.
    """
    d = pd.read_json(ROOT / "outputs" / "raw" / "ablations_boundary.jsonl",
                     lines=True) if (ROOT / "outputs" / "raw" /
                                     "ablations_boundary.jsonl").exists() else None
    if d is None:
        raise KeyError("ablations_boundary.jsonl absent")
    g = d.groupby("attack_id").harmful_accept.mean()
    return float(g.max() - g.min())


def side(study: str, k1, k2, k3="") -> float:
    """A pre-aggregated side-experiment cell.

    These read outputs/processed rather than outputs/raw so that `--mode claims`
    needs no large files and the released artifact can omit the 192 MB curvature
    sweep and the 22 MB improvement run.
    """
    d = load("side_experiments.csv")
    m = (d.study == study) & (d.key1.astype(str) == str(k1)) & (d.key2 == k2)
    if k3:
        m &= (d.key3 == k3)
    r = d[m]
    if not len(r):
        raise KeyError(f"{study}/{k1}/{k2}/{k3} missing from side_experiments")
    return float(r.value.iloc[0])


def cross(arm: str, col: str = "max_attack_effect_HAR") -> float:
    """One row of the per-family comparison table."""
    d = load("cross_model.csv")
    r = d[d.arm == arm]
    if not len(r):
        raise KeyError(f"{arm} missing from cross_model")
    return float(r[col].iloc[0])


def improvement_gap(arm: str, improvement: str) -> float:
    """How far an arm's response to a genuine concession sits from the oracle's."""
    d = pd.read_json(ROOT / "outputs" / "raw" / "main_improvements.jsonl", lines=True)
    pv = d.pivot_table(index="task_id", columns=["buyer_id", "attack_id"],
                       values="accepted", aggfunc="first")
    r = (pv[(arm, improvement)] - pv[(arm, "clean")]).mean()
    o = (pv[("oracle", improvement)] - pv[("oracle", "clean")]).mean()
    return float(r - o)


def competence(family: str) -> float:
    d = load("discrimination.csv")
    r = d[d.buyer_id == family]
    if not len(r):
        raise KeyError(f"{family} missing from discrimination")
    return float(r.discrimination_gap.iloc[0])


def competence_effect_rho() -> float:
    """Spearman between honest-offer competence and worst-case attack effect."""
    from scipy import stats
    d = load("cross_model.csv")
    d = d[d.arm.str.startswith("llm_direct")]
    return float(stats.spearmanr(d.discrimination if "discrimination" in d
                                 else [competence(a) for a in d.arm],
                                 d.max_attack_effect_HAR).statistic)


# Every number the write-up states, with the computation that must reproduce
# it. `text` is searched for verbatim in the source, so a value edited in one
# place and not the other is caught as well.
CLAIMS: List[Tuple[str, str, Callable[[], float], float]] = [
    ("claimed-EU harmful acceptance", "0.097",
     lambda: arm("CraigslistBargain", "claimed_eu", "harmful_accept"), 0.001),
    ("contract closure harmful acceptance", "0.037",
     lambda: arm("CraigslistBargain", "conservative", "harmful_accept"), 0.001),
    ("full gate harmful acceptance", "0.027",
     lambda: arm("CraigslistBargain", "robust_gate", "harmful_accept"), 0.001),
    ("frontier family, largest attack effect", "0.270",
     lambda: effect("CraigslistBargain", "llm_direct_gpt5mini", "combined"), 0.001),
    ("frontier family, clean harmful acceptance", "0.000",
     lambda: cell("CraigslistBargain", "llm_direct_gpt5mini", "clean",
                  "harmful_accept"), 1e-9),
    ("RQ1 combined, Craigslist", "0.326",
     lambda: effect("CraigslistBargain", "claimed_eu", "combined"), 0.001),
    ("RQ1 combined, AmazonHistoryPrice", "0.184",
     lambda: effect("AmazonHistoryPrice", "claimed_eu", "combined"), 0.001),
    ("RQ1 combined, Deal or No Deal", "0.266",
     lambda: effect("DealOrNoDeal", "claimed_eu", "combined"), 0.001),
    ("contract channel closed on the gate", "0.0000",
     lambda: effect("CraigslistBargain", "robust_gate", "contract_complexity"), 1e-9),
    ("tail channel closed on the gate", "0.0000",
     lambda: effect("CraigslistBargain", "robust_gate", "tail_incentive"), 1e-9),
    ("DND claimed-EU harmful acceptance", "0.120",
     lambda: arm("DealOrNoDeal", "claimed_eu", "harmful_accept"), 0.001),
    ("DND gate harmful acceptance", "0.064",
     lambda: (arm("DealOrNoDeal", "conservative", "harmful_accept")
              - arm("DealOrNoDeal", "robust_gate", "harmful_accept")), 0.001),
    ("DND gate valid acceptance", "0.352",
     lambda: arm("DealOrNoDeal", "robust_gate", "valid_accept"), 0.001),
    ("DND claimed-EU valid acceptance", "0.725",
     lambda: arm("DealOrNoDeal", "claimed_eu", "valid_accept"), 0.001),
    ("prompt-defence spread, smallest channel", "0.092",
     lambda: min(prompt_defence_spread(a) for a in
                 ("selective_disclosure", "state_misrepresentation")), 0.001),
    ("prompt-defence spread, largest channel", "0.190",
     lambda: prompt_defence_spread("combined"), 0.001),
    ("tail incentive, weakest family", "0.010",
     lambda: prompt_defence_extreme("tail_incentive", "min"), 0.001),
    ("tail incentive, strongest family", "0.170",
     lambda: prompt_defence_extreme("tail_incentive", "max"), 0.001),
    ("selective invariance, closure response to a large concession", "0.258",
     lambda: side("improvement", "conservative", "genuine_discount_large"), 0.001),
    ("selective invariance, oracle response to the same concession", "0.258",
     lambda: side("improvement", "oracle", "genuine_discount_large"), 0.001),
    ("multi-round, undefended inflation", "0.242",
     lambda: rounds_effect("claimed_eu", "inconsistent_inflation"), 0.001),
    ("multi-round, closure on the violating half", "0.0000",
     lambda: rounds_effect("consistency_closed", "inconsistent_inflation"), 1e-9),
    ("multi-round, closure on the respecting half", "0.237",
     lambda: rounds_effect("consistency_closed", "consistent_inflation"), 0.001),
    ("multi-round, full gate on the violating half", "0.088",
     lambda: rounds_effect("robust_gate", "inconsistent_inflation"), 0.001),
    ("curvature, undefended contract effect at gamma=0.5", "0.117",
     lambda: side("curvature", 0.5, "claimed_eu", "contract_complexity"), 0.001),
    ("curvature, undefended contract effect at gamma=2", "0.098",
     lambda: side("curvature", 2.0, "claimed_eu", "contract_complexity"), 0.001),
    ("multi-round, price of the closure", "0.725",
     lambda: rounds_valid("consistency_closed", "clean"), 0.001),
    ("out-of-domain, projection on its own channel", "0.000",
     lambda: cd_harm("irrelevant_field", "project_out"), 1e-9),
    ("out-of-domain, canonicalisation on its own channel", "0.000",
     lambda: cd_harm("enumerable_action", "close_exact"), 1e-9),
    ("out-of-domain, class three under structural defence", "1.000",
     lambda: cd_harm("payload_content", "project_out", "min"), 1e-9),
    ("out-of-domain, prompt defence worst family", "0.875",
     lambda: cd_harm("payload_content", "instruction"), 0.001),
    ("out-of-domain, spotlighting baseline worst family", "1.000",
     lambda: cd_harm("payload_content", "spotlight"), 0.001),
    ("out-of-domain, spotlighting baseline best family", "0.150",
     lambda: cd_harm("payload_content", "spotlight", "min"), 0.001),
    ("out-of-domain, undefended attack success", "0.950",
     lambda: min(cd_harm(c, "none", "min")
                 for c in ("irrelevant_field", "payload_content")), 0.001),
    ("out-of-domain, undefended task accuracy", "0.800",
     lambda: float(load("crossdomain_summary.csv").query(
         "channel=='clean' and defence=='none'").clean_accuracy.max()), 0.001),
    ("out-of-domain, price of canonicalisation", "0.000",
     lambda: cd_price("close_exact", "max"), 0.001),
    ("out-of-domain, price of projection", "0.150",
     lambda: cd_price("project_out", "max"), 0.001),
    ("out-of-domain, price of the soft defences", "0.375",
     lambda: max(cd_price(d, "max") for d in ("spotlight", "instruction")), 0.001),
    ("out-of-domain, prompt defence best family", "0.000",
     lambda: cd_harm("payload_content", "instruction", "min"), 1e-9),
    # The one cell the criterion is silent on: projection is not supposed to
    # act on the class-two channel, and the table shows it moving there. The
    # size of that move is quoted in the body, so it is recomputed here.
    ("out-of-domain, projection off its own channel", "0.103",
     lambda: cd_harm("enumerable_action", "none", "min_nonzero"), 0.001),
    # --- prompt form: the same offer as a seller's message ------------------
    ("prose, frontier buyer combined channel, structured", "0.290",
     lambda: prose("gpt-5-mini", "json", "combined"), 0.001),
    ("prose, frontier buyer combined channel, prose", "0.370",
     lambda: prose("gpt-5-mini", "prose", "combined"), 0.001),
    ("prose, frontier buyer paired form difference", "+0.080",
     lambda: prose("gpt-5-mini", "prose_minus_json", "combined"), 0.001),
    ("prose, clean harmful acceptance of the weak-in-prose family", "0.220",
     lambda: prose("gpt-4o-mini", "prose", "clean"), 0.001),
    # --- AgentDojo ----------------------------------------------------------
    ("agentdojo, undefended attack success", "0.465",
     lambda: adj("undefended"), 0.001),
    ("agentdojo, canonicalization on its own channel", "0.000",
     lambda: adj_channel("canonicalization", True), 1e-9),
    ("agentdojo, canonicalization off its own channel", "0.438",
     lambda: adj_channel("canonicalization", False), 0.001),
    ("agentdojo, undefended off that same channel", "0.438",
     lambda: adj_channel("undefended", False), 0.001),
    ("agentdojo, published datamarking baseline", "0.528",
     lambda: adj("datamarking"), 0.001),
    ("agentdojo, prompt defense", "0.403",
     lambda: adj("prompt_defense"), 0.001),
    ("agentdojo, projection on its isolated channel", "0.000",
     lambda: adj("projection", vector="injection_incoming_transaction"), 1e-9),
    ("agentdojo, undefended on that isolated channel", "0.271",
     lambda: adj("undefended", vector="injection_incoming_transaction"), 0.001),
    ("agentdojo travel, closure on its own channel", "0.000",
     lambda: adj_channel("canonicalization", True, "travel"), 1e-9),
    ("agentdojo travel, undefended on that channel", "0.250",
     lambda: adj_channel("undefended", True, "travel"), 0.001),
    ("agentdojo travel, closure off channel", "0.300",
     lambda: adj_channel("canonicalization", False, "travel"), 0.001),
    ("agentdojo travel, undefended off channel", "0.310",
     lambda: adj_channel("undefended", False, "travel"), 0.001),
    ("agentdojo travel, prompt defense aggregate", "0.121",
     lambda: adj("prompt_defense", suite="travel"), 0.001),
    ("agentdojo travel, undefended aggregate", "0.293",
     lambda: adj("undefended", suite="travel"), 0.001),
    ("agentdojo slack, on-channel goal under closure", "0.429",
     lambda: adj_channel("canonicalization", True, "slack"), 0.001),
    ("agentdojo slack, on-channel goal undefended", "0.810",
     lambda: adj_channel("undefended", True, "slack"), 0.001),
    ("agentdojo slack, closure off channel", "0.619",
     lambda: adj_channel("canonicalization", False, "slack"), 0.001),
    ("agentdojo slack, undefended off channel", "0.607",
     lambda: adj_channel("undefended", False, "slack"), 0.001),
    ("agentdojo banking, closure aggregate", "0.049",
     lambda: adj("canonicalization"), 0.001),
    ("agentdojo, payload reached the second model", "0.938",
     lambda: adj_exposure("gpt-5-mini", "saw_injection"), 0.001),
    ("agentdojo, payload reached the first model", "0.847",
     lambda: adj_exposure("gpt-4o-mini", "saw_injection"), 0.001),
    ("agentdojo, first model acted on the payload", "0.542",
     lambda: adj_exposure("gpt-4o-mini", "attempted_attacker_action"), 0.001),
    ("agentdojo, second model acted on the payload", "0.007",
     lambda: adj_exposure("gpt-5-mini", "attempted_attacker_action"), 0.001),
    ("agentdojo, second model flagged it", "0.549",
     lambda: adj_exposure("gpt-5-mini", "flagged_in_reply"), 0.001),
    ("agentdojo, first model flagged it", "0.028",
     lambda: adj_exposure("gpt-4o-mini", "flagged_in_reply"), 0.001),
    ("agentdojo, second model undefended attack success", "0.007",
     lambda: adj("undefended", model="gpt-5-mini"), 0.001),
    ("agentdojo, second model under closure", "0.000",
     lambda: adj("canonicalization", model="gpt-5-mini"), 1e-9),
    ("agentdojo, second model utility", "0.542",
     lambda: adj("undefended", "utility", model="gpt-5-mini"), 0.001),
    ("agentdojo, second model addressed generically", "0.000",
     lambda: adj("undefended", model="gpt-5-mini@generic-name"), 1e-9),
    ("agentdojo, datamarking paired difference", "+0.062",
     lambda: adj("datamarking") - adj("undefended"), 0.001),
    ("agentdojo, prompt defense paired difference", "-0.062",
     lambda: adj("prompt_defense") - adj("undefended"), 0.001),
    ("agentdojo, aggregate utility price of closure", "0.007",
     lambda: adj("undefended", "utility") - adj("canonicalization", "utility"),
     0.001),
    ("agentdojo, utility undefended", "0.410",
     lambda: adj("undefended", "utility"), 0.001),
    ("agentdojo, utility under canonicalization", "0.403",
     lambda: adj("canonicalization", "utility"), 0.001),
    # --- the remaining body numbers, so nothing quoted is unguarded ---------
    ("contract channel, undefended", "0.093",
     lambda: effect("CraigslistBargain", "claimed_eu", "contract_complexity"), 0.001),
    ("tail channel, undefended", "0.127",
     lambda: effect("CraigslistBargain", "claimed_eu", "tail_incentive"), 0.001),
    ("disclosure channel, undefended", "0.048",
     lambda: effect("CraigslistBargain", "claimed_eu", "selective_disclosure"), 0.001),
    ("qwen combined effect", "0.042",
     lambda: effect("CraigslistBargain", "llm_direct", "combined"), 0.001),
    ("deepseek combined effect", "0.015",
     lambda: -effect("CraigslistBargain", "llm_direct_deepseek", "combined"), 0.001),
    ("qwen clean harmful acceptance", "0.227",
     lambda: cell("CraigslistBargain", "llm_direct", "clean", "harmful_accept"), 0.001),
    ("deepseek clean harmful acceptance", "0.095",
     lambda: cell("CraigslistBargain", "llm_direct_deepseek", "clean",
                  "harmful_accept"), 0.001),
    ("frontier competence", "0.976", lambda: competence("llm_direct_gpt5mini"), 0.001),
    ("weakest competence", "0.048", lambda: competence("llm_direct_gpt4omini"), 0.001),
    ("extreme-pair effect gap", "0.065",
     lambda: cross("llm_direct_gpt5mini") - cross("llm_direct_gpt4omini"), 0.001),
    ("matched-pair effect gap", "0.230",
     lambda: cross("llm_direct_devstral") - cross("llm_direct_deepseek"), 0.001),
    ("competence-effect rank correlation", "0.250",
     competence_effect_rho, 0.001),
    ("closure response, small concession", "0.096",
     lambda: side("improvement", "conservative", "genuine_discount_small"), 0.001),
    ("gate response minus oracle", "0.0008",
     lambda: (side("improvement", "robust_gate", "genuine_discount_large")
              - side("improvement", "oracle", "genuine_discount_large")), 0.0002),
    ("ablation, full method", "0.027",
     lambda: side("ablation", "CraigslistBargain", "full method", "HAR"), 0.001),
    ("ablation, contract closure removed", "0.080",
     lambda: side("ablation", "CraigslistBargain",
                  "- contract canonicalization", "HAR"), 0.001),
    ("boundary, closure acceptance shift", "0.0226",
     lambda: side("boundary", "conservative", "incomparable_swap"), 0.0002),
    ("boundary, gate acceptance shift", "0.0334",
     lambda: side("boundary", "robust_gate", "incomparable_swap"), 0.0002),
    ("boundary, unprotected acceptance shift", "0.0015",
     lambda: side("boundary", "claimed_eu", "incomparable_swap"), 0.0002),
    ("hybrid harmful-acceptance gain", "0.176",
     lambda: -hybrid("harmful_accept"), 0.001),
    ("hybrid valid-acceptance gain", "0.054",
     lambda: hybrid("valid_accept"), 0.001),
    ("multi-round gate valid acceptance", "0.158",
     lambda: float(load("dnd_rounds_summary.csv").query(
         "environment=='DealOrNoDealRounds' and buyer_id=='robust_gate' "
         "and attack_id=='clean'").valid_accept.iloc[0]), 0.001),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="", help="directory holding a write-up to cross-check")
    ap.add_argument("--tex", default="main.tex")
    ap.add_argument("--mode", choices=("full", "claims"), default="full",
                    help="`claims` skips the checks that need outputs/raw, which "
                         "is 240 MB and does not belong in a repository; the "
                         "claim recomputation needs only the 172 KB of processed "
                         "CSVs and is what CI should run.")
    args = ap.parse_args()
    fails: List[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f" -- {detail}" if not ok else ""))
        if not ok:
            fails.append(label)

    raw = ROOT / "outputs" / "raw"
    if args.mode == "full" and not raw.exists():
        print("FAIL  outputs/raw present -- run the experiments, or use --mode claims")
        return 1

    # ---- the pooled summaries must be current ----------------------------
    # Claims are recomputed from these five, so a stale one makes the whole
    # recomputation vacuous: a pooling change once sat behind them and this gate
    # reported green while the write-up no longer matched the data. Only the
    # files analyze.py writes are checked, and only against the files it reads;
    # ablation and conformal CSVs come from other scripts and do not go stale
    # when a new model family is added.
    POOLED_CSVS = ("summary.csv", "arm_summary.csv", "rq1_attack_effects.csv",
                   "discrimination.csv", "rq2_arm_comparison.csv",
                   "cross_model.csv", "hybrid_vs_others.csv")
    if args.mode == "full" and raw.exists():
        newest_raw = max((f.stat().st_mtime for f in raw.glob("*.jsonl")
                          if f.name in POOLED), default=0)
        proc = ROOT / "outputs" / "processed"
        stale = [n for n in POOLED_CSVS
                 if (proc / n).exists() and (proc / n).stat().st_mtime < newest_raw]
        check(not stale, "pooled summaries are current with the raw records",
              f"{stale} predate a pooled raw file; run scripts/analyze.py")

    # ---- one environment per pooled file, matching its declared environment ----
    for name, env in (POOLED.items() if args.mode == "full" else ()):
        f = raw / name
        if not f.exists():
            check(False, f"{name} present", "missing")
            continue
        d = pd.read_json(f, lines=True)
        if "environment" not in d.columns:
            # An arm written without the column cannot be placed, and the pooling
            # step will silently assign it to the primary environment.
            check(False, f"{name} declares its environment",
                  "no `environment` column; pooling would guess")
            continue
        got = set(d.environment.dropna().unique())
        check(got == {env}, f"{name} holds only {env}", f"found {sorted(got)}")

    # ---- nothing unexpected is being pooled --------------------------------
    if args.mode == "full":
        pooled_now = {f.name for f in raw.glob("*.jsonl")
                      if not f.name.startswith(("ablations", "_"))}
        extra = sorted(pooled_now - set(POOLED) - STANDALONE)
        check(not extra, "no unregistered file enters the pooled analysis",
              f"{extra} would be pooled; prefix with '_' or register it")

    # ---- the arm sets the analysis is supposed to contain -------------------
    A = load("arm_summary.csv")
    for env in ("CraigslistBargain", "AmazonHistoryPrice", "DealOrNoDeal"):
        got = set(A[A.environment == env].buyer_id)
        check(SCRIPTED <= got, f"{env} has all six scripted arms",
              f"missing {sorted(SCRIPTED - got)}")
    dnd = pd.read_json(raw / "main_dnd.jsonl", lines=True) if args.mode == "full" else None
    if dnd is not None:
        got = set(dnd.attack_id)
        check(got == set(DND_ATTACKS), "Deal or No Deal ran its own attack set",
              f"found {sorted(got)}; running it through the Craigslist renderer "
              f"silently turns three of its families into no-ops")

    # ---- the multi-round result is an invariance, not an average -----------
    # The paper reports 0.0000 for the closure on the identity-violating
    # inflation. That is only meaningful if no single task was non-zero, so the
    # per-task flag is asserted rather than the mean being re-rounded.
    R = load("dnd_rounds_summary.csv")
    r = R[(R.environment == "DealOrNoDealRounds")
          & (R.buyer_id == "consistency_closed")]
    viol = r[r.attack_id == "inconsistent_inflation"]
    check(len(viol) == 1 and bool(viol.exact_zero.iloc[0]),
          "consistency closure is exactly zero on every task, not on average",
          "at least one task moved; the reported 0.0000 would be a mean of "
          "offsetting errors")
    resp = r[r.attack_id == "consistent_inflation"]
    nodef = R[(R.buyer_id == "claimed_eu")
              & (R.attack_id == "consistent_inflation")]
    check(len(resp) == 1 and len(nodef) == 1
          and abs(float(resp.effect.iloc[0]) - float(nodef.effect.iloc[0])) < 1e-12,
          "consistency closure gives no protection at all on the consistent half",
          "the negative half of the dichotomy has changed; if the closure now "
          "helps here it is reading more than the public-count identity")
    cm = R[R.environment == "DealOrNoDealRoundsCommit"]
    cc = cm[cm.buyer_id == "consistency_closed"].sort_values("attack_id")
    va = list(cc.valid_accept)
    check(va == sorted(va) and all(h == 0.0 for h in cc.harmful_accept),
          "commitment sweep is monotone at zero harmful acceptance",
          f"valid acceptance {va} is not non-decreasing, or harm is non-zero")

    # ---- the out-of-domain test must stay a prediction, not a fit ----------
    # The closure must not increase harm at any curvature. Equality to zero is
    # more than Proposition 1 gives: at gamma=0.5 the measured cell is -0.0004,
    # a decrease, on a benchmark whose base rate of bad offers shifts with the
    # curve. Asserting <= 0 is the claim; asserting == 0 would fail honestly.
    if args.mode == "full":
        f = raw / "_utility_sensitivity.jsonl"
        if f.exists():
            u = pd.read_json(f, lines=True)
            worst = []
            for g in sorted(u.quality_curve.unique()):
                for arm in ("conservative", "robust_gate"):
                    for atk in ("contract_complexity", "tail_incentive"):
                        d_ = u[(u.quality_curve == g) & (u.buyer_id == arm)]
                        e = (d_[d_.attack_id == atk].harmful_accept.mean()
                             - d_[d_.attack_id == "clean"].harmful_accept.mean())
                        worst.append(e)
            check(max(worst) <= 1e-9,
                  "closure never increases harm at any curvature",
                  f"largest effect {max(worst):+.4f}; the exact closure would be "
                  f"an artifact of the affine quality map")
        else:
            check(False, "utility sensitivity present",
                  "run scripts/run_utility_sensitivity.py")

    # ---- selective invariance needs BOTH halves ---------------------------
    # Invariance to revenue-neutral manipulation is trivially satisfied by a rule
    # that rejects everything. The claim is only meaningful beside a rule's
    # response to a genuine concession, so that response is asserted against the
    # oracle's rather than reported alone.
    if args.mode == "full":
        f = raw / "main_improvements.jsonl"
        if f.exists():
            im = pd.read_json(f, lines=True)
            pv = im.pivot_table(index="task_id", columns=["buyer_id", "attack_id"],
                                values="accepted", aggfunc="first")
            for arm, tol in (("conservative", 0.0), ("robust_gate", 0.02)):
                worst = 0.0
                for k in ("genuine_discount_small", "genuine_discount_large"):
                    r = (pv[(arm, k)] - pv[(arm, "clean")]).mean()
                    o = (pv[("oracle", k)] - pv[("oracle", "clean")]).mean()
                    worst = max(worst, abs(r - o))
                check(worst <= tol + 1e-9,
                      f"{arm} tracks a genuine concession like the oracle",
                      f"off by {worst:.4f}; the closure would be buying its "
                      f"invariance with insensitivity rather than structure")
        else:
            check(False, "genuine-improvement run present",
                  "run scripts/run_selective_invariance.py")

    C = load("crossdomain_summary.csv")
    clean = C[C.channel == "clean"]
    check(len(clean) and float(clean.harm.max()) == 0.0,
          "out-of-domain harm has no false positives on honest documents",
          "an honest document produced the attacker's action; the harm metric "
          "is measuring something other than the injection")
    undef = C[(C.channel != "clean") & (C.defence == "none")]
    check(len(undef) and float(undef.harm.max()) > 0.5,
          "the out-of-domain attack actually lands undefended",
          "attack success is near zero everywhere, so the defences cannot be "
          "separated and the comparison is vacuous")
    # The published baseline must not be exact on any channel: datamarking
    # annotates the payload instead of removing it, so an exact zero there would
    # falsify the black-box proposition's empirical half rather than support it.
    for ch in ("irrelevant_field", "enumerable_action", "payload_content"):
        cell = C[(C.channel == ch) & (C.defence == "spotlight")]
        check(len(cell) and float(cell.harm.max()) > 0.0,
              f"spotlighting is not exact on {ch}",
              "a marking defence came out exactly zero; it is not an invariance, "
              "so this would contradict Proposition 3 rather than confirm it")

    spread = C[(C.channel == "payload_content") & (C.defence == "instruction")]
    check(len(spread) >= 3
          and float(spread.harm.max()) - float(spread.harm.min()) > 0.5,
          "the prompt defence is model-dependent out of domain",
          "the spread collapsed; Proposition 3's empirical consequence would "
          "no longer hold and the paper's claim needs rewriting")

    fams = set(load("discrimination.csv").buyer_id) & MODEL_FAMILIES
    check(fams == MODEL_FAMILIES, "exactly the seven model families are present",
          f"got {sorted(fams)}")

    # ---- the write-up's numbers must recompute ---------------------------
    # A write-up is optional here; without one the values are still recomputed
    # from the shipped summaries, so the checks stay meaningful on their own.
    tex = (ROOT / args.paper / args.tex).resolve() if args.paper else None
    src = tex.read_text(encoding="utf8") if tex and tex.exists() else ""
    if args.paper:
        check(bool(src), f"write-up readable at {tex}", "not found")
    # An older write-up may predate the AgentDojo and prompt-form studies
    # and does not quote their numbers. Those claims still have to recompute --
    # a broken artifact is broken for either paper -- but requiring their
    # strings to appear in a write-up that never made the claim would fail a
    # correct file. Presence of the study in the source decides which rule
    # applies, so nothing quoted anywhere goes unguarded.
    added_later = ("agentdojo", "prose")
    has_new = "AgentDojo" in src
    for label, text, fn, tol in CLAIMS:
        try:
            got = fn()
        except KeyError as exc:
            check(False, f"claim recomputes: {label}", str(exc))
            continue
        if label.startswith(added_later) and not has_new:
            check(abs(got - float(text)) <= tol,
                  f"claim recomputes (not quoted by this write-up): {label}",
                  f"data gives {got:.4f}, expected {text}")
            continue
        near = abs(got - float(text)) <= tol
        shown = (not src) or re.search(r"\$?" + re.escape(text) + r"\$?", src) is not None
        check(near and shown, f"claim recomputes and appears in text: {label}",
              f"data gives {got:.4f}, text says {text}"
              + ("" if shown else "; and that string is absent from the source"))

    print(f"\n{len(fails)} failed check(s)" if fails else "\nall artifact checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
