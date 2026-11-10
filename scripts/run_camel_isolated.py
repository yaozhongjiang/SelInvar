"""Run CaMeL one (user, injection) pair per subprocess, so an interpreter crash costs one pair.

    $CAMEL scripts/run_camel_isolated.py --suite slack --workers 4

A pair that never finishes is recorded with attack_success/utility = null and a
`status` of interpreter_crash, api_error or timeout: it counts neither as a
blocked attack nor as a failed task. Finished pairs resume from logs/ at no cost.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suites

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts/run_camel_agentdojo.py"
API = re.compile(r"RateLimitError|APIConnectionError|APITimeoutError|InternalServerError|insufficient_quota|AuthenticationError")
ERR = re.compile(r"^[A-Za-z_.]*(?:Error|Exception)\b.*$", re.M)


def key(s):
    return int(s.rsplit("_", 1)[1])


def attempt(a, arm, u, i):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        cmd = [sys.executable, str(RUN), "--suite", a.suite, "--model", a.model, "--arms", arm,
               "--user-tasks", u, "--injection-tasks", i, "--output", str(out)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, cwd=ROOT)
        except subprocess.TimeoutExpired:
            return None, "timeout", f"exceeded {a.timeout}s"
        if p.returncode == 0 and out.exists():
            return json.loads(out.read_text().splitlines()[0]), "ok", ""
        errs = ERR.findall(p.stderr + p.stdout)
        msg = errs[-1][:200] if errs else f"exit {p.returncode}"
        return None, ("api_error" if API.search(p.stderr + p.stdout) else "interpreter_crash"), msg


def one(a, arm, u, i):
    for _ in range(a.tries):
        row, status, msg = attempt(a, arm, u, i)
        if status == "ok":
            break
    row = row or {"model_id": a.model, "suite": a.suite, "arm": arm, "user_task": u, "injection_task": i,
                  "attack": "important_instructions", "vector": "all", "attack_success": None, "utility": None}
    row.update(status=status, error=msg)
    print(arm, u, i, status, msg, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only, for testing")
    ap.add_argument("--injection-tasks", default="", help="comma-separated subset, default all")
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    s = get_suites("v1.2.1")[a.suite]
    pairs = [(u, i) for u in sorted(s.user_tasks, key=key) for i in sorted(s.injection_tasks, key=key)]
    if a.injection_tasks:
        keep = set(a.injection_tasks.split(","))
        pairs = [p for p in pairs if p[1] in keep]
    if a.limit:
        pairs = pairs[:a.limit]
    out = ROOT / (a.output or f"outputs/harness_runs/camel_{a.model}_{a.suite}.jsonl")
    with ThreadPoolExecutor(a.workers) as ex:
        first = list(ex.map(lambda p: one(a, "camel", *p), pairs))
    rows = list(first)
    ok = [(r["user_task"], r["injection_task"]) for r in first if r["status"] == "ok"]
    for r in first:
        if r["status"] != "ok":
            rows.append({**r, "arm": "camel_secpol", "status": "not_replayed", "error": "camel run did not finish"})
    with ThreadPoolExecutor(a.workers) as ex:
        rows += list(ex.map(lambda p: one(a, "camel_secpol", *p), ok))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", len(rows), "rows to", out, "; camel not ok:", sum(r["status"] != "ok" for r in first))


if __name__ == "__main__":
    main()
