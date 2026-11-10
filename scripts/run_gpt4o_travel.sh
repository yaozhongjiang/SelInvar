#!/bin/zsh
# gpt-4o-2024-08-06 on travel on-channel goals (injection_task_1, injection_task_3).
cd "$(dirname "$0")/.." || exit 1
set -a; . ../external/camel-prompt-injection/.env; set +a
PY=../external/camel-prompt-injection/.venv/bin/python
L=outputs/harness_runs/full_logs; mkdir -p $L
$PY scripts/run_camel_isolated.py --suite travel --model gpt-4o --workers 4 \
  --injection-tasks injection_task_1,injection_task_3 > $L/gpt4o_travel_camel.log 2>&1 &
c=$!
$PY - > $L/gpt4o_travel_base.log 2>&1 <<'P' &
import sys, runpy
sys.path.insert(0, 'scripts')
import run_camel_agentdojo as r
r.track_usage()
sys.argv = ['run_agentdojo.py', '--model', 'gpt-4o-2024-08-06', '--address-as', 'gpt-4o-2024-05-13',
            '--suite', 'travel', '--arms', 'undefended,canonicalization',
            '--injection-tasks', 'injection_task_1,injection_task_3',
            '--logdir', 'outputs/harness_runs/adj_gpt4o_runs', '--output', 'outputs/harness_runs/snapshot_gpt4o_travel.jsonl']
try:
    runpy.run_path('scripts/run_agentdojo.py', run_name='__main__')
finally:
    print('BASE_USAGE', r.USAGE)
P
b=$!
wait $c; ce=$?; wait $b; be=$?
echo "camel exit $ce, baseline exit $be"
tail -n 1 $L/gpt4o_travel_camel.log | cut -c1-200
grep -E 'BASE_USAGE' $L/gpt4o_travel_base.log
