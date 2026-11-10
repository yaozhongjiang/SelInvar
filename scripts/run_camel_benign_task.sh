#!/bin/zsh
# One benign (no-attack) CaMeL run for travel user task $1; used by xargs -P.
cd "$(dirname "$0")/.." || exit 1
D=outputs/harness_runs/benign_travel
../external/camel-prompt-injection/.venv/bin/python scripts/run_camel_agentdojo.py \
  --suite travel --no-attack --user-tasks "user_task_$1" --output "$D/user_task_$1.jsonl" \
  > "$D/user_task_$1.log" 2>&1 \
  || echo "user_task_$1 FAILED: $(grep -E '^[A-Za-z_.]*(Error|Exception)' "$D/user_task_$1.log" | tail -n 1 | cut -c1-160)"
