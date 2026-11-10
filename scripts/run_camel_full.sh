#!/bin/zsh
# CaMeL and snapshot-matched baselines: full gpt-4o-mini run on banking, travel and slack.
# Each suite runs CaMeL (resumable from logs/) and the snapshot-matched baselines.
cd "$(dirname "$0")/.." || exit 1
set -a; . ../external/camel-prompt-injection/.env; set +a
PY=../external/camel-prompt-injection/.venv/bin/python
SNAP=gpt-4o-mini-2024-07-18
OUT=outputs/harness_runs
mkdir -p $OUT/full_logs

run_camel() {
  local s=$1 try
  for try in 1 2 3 4 5; do
    echo "[camel $s] attempt $try $(date)" >> $OUT/full_logs/camel_$s.log
    $PY scripts/run_camel_agentdojo.py --suite $s >> $OUT/full_logs/camel_$s.log 2>&1 && return 0
    sleep 60
  done
  return 1
}

run_base() {
  local s=$1 try
  for try in 1 2 3; do
    echo "[base $s] attempt $try $(date)" >> $OUT/full_logs/base_$s.log
    $PY scripts/run_agentdojo.py --model $SNAP --suite $s --address-as $SNAP \
      --arms undefended,canonicalization --logdir $OUT/adj_snapshot_runs \
      --output $OUT/snapshot_$s.jsonl >> $OUT/full_logs/base_$s.log 2>&1 && return 0
    sleep 60
  done
  return 1
}

pids=()
for s in banking travel slack; do
  run_camel $s & pids+=($!)
  run_base $s & pids+=($!)
done
fail=0
for p in $pids; do wait $p || fail=$((fail+1)); done
echo "finished $(date), failed jobs: $fail"
exit $fail
