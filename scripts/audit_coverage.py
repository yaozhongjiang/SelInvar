"""Check that each claimed capability is implemented *and* exercised.

The original version grepped the whole tree for a token, which passes as long
as the string exists somewhere -- including inside dead scaffold code. Here each
requirement names the module that must define it and, where applicable, the
output artefact that proves it actually ran.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# requirement -> (symbol, module it must live in, artefact proving it ran)
REQUIRED = {
    "paired clean/attack renderings": ("def render", "src/attacks.py", None),
    "revenue-neutral attacks": ("assert_revenue_neutral", "src/attacks.py", None),
    "evidence-bounded manipulation": ("manipulation_capacity", "src/attacks.py", None),
    "Wasserstein DRO (LP)": ("wasserstein_worst_case", "src/dro_solver.py", None),
    "DRO dual (fast curve)": ("def bound_curve", "src/buyers.py", None),
    "adaptive ambiguity radius": ("def adaptive_radius", "src/buyers.py", None),
    "contract canonicalization": ("def safe_price", "src/buyers.py", None),
    "urgency decoupling": ("use_urgency_decoupling", "src/buyers.py", None),
    "distinct buyer arms": ("SCRIPTED = {", "src/buyers.py", None),
    "LLM buyer arms": ("SYSTEMS = {", "src/llm.py", None),
    "task-level bootstrap": ("def bootstrap_mean", "src/stats.py", None),
    "paired permutation test": ("def paired_diff", "src/stats.py", None),
    "multiple-comparison correction": ("def holm_bonferroni", "src/stats.py", None),
    "held-out calibration": ("def main", "scripts/calibrate_gate.py",
                             "outputs/gate_config.json"),
    "main test evaluation": ("def main", "scripts/run_main.py",
                             "outputs/raw/main_scripted.jsonl"),
    "mechanism ablations": ("def main", "scripts/run_ablations.py",
                            "outputs/processed/ablations.csv"),
    "adaptive attacker": ("def fit_policy", "scripts/run_adaptive_attacker.py",
                          "outputs/processed/adaptive_attacker.csv"),
    "result tables": ("def table_main", "scripts/make_tables.py",
                      "outputs/tables/main_results.tex"),
    "figures": ("def fig_frontier", "scripts/make_figures.py",
                "outputs/figures/fig_frontier.pdf"),
    "cost accounting": ("estimate_api_cost", "src/cost.py", None),
    "verifier abstraction": ("class Verifier", "src/verifier.py", None),
}

report, ok = [], True
for name, (symbol, module, artefact) in REQUIRED.items():
    p = ROOT / module
    implemented = p.exists() and symbol in p.read_text(errors="ignore")
    ran = None if artefact is None else (ROOT / artefact).exists()
    passed = implemented and (ran is not False)
    ok &= passed
    report.append({"requirement": name, "module": module,
                   "implemented": implemented, "artefact": artefact, "ran": ran})
    flag = "PASS" if passed else "FAIL"
    extra = "" if ran is None else ("  [ran]" if ran else "  [NOT RUN]")
    print(f"{flag}  {name:32s} {module}{extra}")

out = ROOT / "outputs" / "coverage_audit.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
raise SystemExit(0 if ok else 1)
