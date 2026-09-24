# Verification entry points.
#
#   make all       both of the below
#   make verify    every reported number, recomputed from the shipped summaries
#   make test      the invariants that guard how those numbers were produced
#
# `verify` needs only outputs/processed. The raw per-trial records are ~500 MB
# and are not included; `make regenerate` lists the runners that rebuild them.

# Not TEX: a TeX distribution exports TEX and `?=` defers to the environment,
# so $(TEX) once expanded to an engine path and the verification silently
# checked an empty file, which made every value read as absent.
PY ?= python3

.PHONY: all verify test regenerate

all: test verify

test:
	@echo "== invariants =="
	@$(PY) -m pytest tests/test_core.py -q

verify:
	@echo "\n== every reported number, recomputed =="
	@$(PY) scripts/check_artifacts.py --mode claims

regenerate:
	@echo "Rebuild order; each step writes into outputs/raw, then re-run step 7:"
	@echo "  1. build_dataset.py  build_ahp.py  build_dnd.py"
	@echo "  2. calibrate_gate.py"
	@echo "  3. run_main.py                 (once per model family; --tag names the arm)"
	@echo "  4. run_dnd_rounds.py  run_selective_invariance.py  run_prose.py"
	@echo "  5. run_boundary.py    run_utility_sensitivity.py"
	@echo "  6. run_crossdomain.py  run_agentdojo.py   (need a chat-completions endpoint)"
	@echo "  7. analyze.py  analyze_extra.py  compare_models.py"
