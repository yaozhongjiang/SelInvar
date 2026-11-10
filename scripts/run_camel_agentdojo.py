"""CaMeL inside the same AgentDojo run as run_agentdojo.py.

Run with the CaMeL environment, from this `code/` directory:

    CAMEL=../external/camel-prompt-injection/.venv/bin/python
    $CAMEL scripts/run_camel_agentdojo.py --dry-run
    OPENAI_API_KEY=... $CAMEL scripts/run_camel_agentdojo.py --suite banking --user-tasks user_task_0

Same as `run_agentdojo.py`: AgentDojo 0.1.35, suite version v1.2.1, attack
`important_instructions`, and AgentDojo's own `security`/`utility` verdicts.

Arms: `camel` is CaMeL with no capability policy; `camel_secpol` replays the
`camel` run under CaMeL's published per-suite policies, so `camel` must run first.
Model names are the dated snapshots CaMeL registers; `run_agentdojo.py` used
OpenAI's undated aliases, which is a difference to report.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

from agentdojo import attacks, benchmark, logging
from agentdojo.task_suite.load_suites import get_suites

import camel.custom_yaml  # noqa: F401
from camel.interpreter.interpreter import MetadataEvalMode
from camel.models import make_tools_pipeline

ROOT = Path(__file__).resolve().parents[1]
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _count(resp):
    u = getattr(resp, "usage", None)
    if u is not None:
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += u.prompt_tokens or 0
        USAGE["completion_tokens"] += u.completion_tokens or 0
    return resp


def track_usage():
    """Count tokens on every OpenAI chat completion, sync or async."""
    from openai.resources.chat.completions import AsyncCompletions, Completions
    sync_create, async_create = Completions.create, AsyncCompletions.create

    def create(self, *a, **k):
        return _count(sync_create(self, *a, **k))

    async def acreate(self, *a, **k):
        return _count(await async_create(self, *a, **k))

    Completions.create, AsyncCompletions.create = create, acreate
ATTACK = "important_instructions"
SNAPSHOT = {"gpt-4o-mini": "openai:gpt-4o-mini-2024-07-18", "gpt-4o": "openai:gpt-4o-2024-08-06"}
ARMS = {"camel": False, "camel_secpol": True}


def pipeline_for(model, replay, suite, attack=ATTACK):
    return make_tools_pipeline(SNAPSHOT[model], False, replay, attack, "medium", None, suite, None,
                               MetadataEvalMode.NORMAL, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini", choices=sorted(SNAPSHOT))
    ap.add_argument("--suite", default="banking", choices=["banking", "travel", "slack"])
    ap.add_argument("--arms", default="camel,camel_secpol")
    ap.add_argument("--user-tasks", default="")
    ap.add_argument("--injection-tasks", default="")
    ap.add_argument("--version", default="v1.2.1")
    ap.add_argument("--output", default="")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--no-attack", action="store_true", help="benign utility: user tasks without injections")
    ap.add_argument("--dry-run", action="store_true", help="build everything; make no model call")
    a = ap.parse_args()

    if a.dry_run:
        os.environ.setdefault("OPENAI_API_KEY", "dry-run-not-a-key")
    elif not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")

    track_usage()
    suite = get_suites(a.version)[a.suite]
    ut = [t for t in a.user_tasks.split(",") if t] or None
    it = [t for t in a.injection_tasks.split(",") if t] or None
    # CaMeL's replay reads Path("logs") relative to the working directory.
    os.chdir(ROOT)
    logdir = ROOT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    out = ROOT / (a.output or f"outputs/harness_runs/camel_{a.model}_{a.suite}.jsonl")
    done = []

    for arm in a.arms.split(","):
        pipe = pipeline_for(a.model, ARMS[arm], a.suite, None if a.no_attack else ATTACK)
        n_u = len(ut or suite.user_tasks)
        if a.no_attack:
            print(f"{arm}: pipeline={pipe.name} no attack, user tasks={n_u}", flush=True)
            if a.dry_run:
                continue
            with logging.OutputLogger(str(logdir)):
                res = benchmark.benchmark_suite_without_injections(
                    pipe, suite, logdir, force_rerun=a.force_rerun, user_tasks=ut)
            util = res["utility_results"]
            for (u, _), ok in util.items():
                done.append({"model_id": a.model, "snapshot": SNAPSHOT[a.model], "suite": a.suite, "arm": arm,
                             "user_task": u, "injection_task": None, "attack": "none", "vector": "none",
                             "attack_success": None, "utility": bool(ok)})
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in done))
            print(f"  benign utility {sum(map(bool, util.values())) / max(len(util), 1):.3f}, n={len(util)}, "
                  f"usage so far {USAGE}", flush=True)
            continue
        attack = attacks.load_attack(ATTACK, suite, pipe)
        n_i = len(it or suite.injection_tasks)
        print(f"{arm}: pipeline={pipe.name} attack={attack.name} pairs={n_u * n_i}", flush=True)
        if a.dry_run:
            continue
        with logging.OutputLogger(str(logdir)):
            res = benchmark.benchmark_suite_with_injections(
                pipe, suite, attack, logdir, force_rerun=a.force_rerun,
                user_tasks=ut, injection_tasks=it, verbose=False)
        util, sec = res["utility_results"], res["security_results"]
        for (u, i), hit in sec.items():
            done.append({"model_id": a.model, "snapshot": SNAPSHOT[a.model], "suite": a.suite, "arm": arm,
                         "user_task": u, "injection_task": i, "attack": ATTACK, "vector": "all",
                         "attack_success": bool(hit), "utility": bool(util.get((u, i), False))})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in done))
        k = max(len(sec), 1)
        print(f"  attack success {sum(map(bool, sec.values())) / k:.3f}, "
              f"utility {sum(map(bool, util.values())) / k:.3f}, n={len(sec)}, usage so far {USAGE}", flush=True)


if __name__ == "__main__":
    main()
