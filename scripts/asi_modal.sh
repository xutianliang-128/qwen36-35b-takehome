#!/usr/bin/env bash
# ASI-Bench (seed31415, B4) for one model: Modal H100 vLLM tunnel + Modal CPU container running the official
# harness, then local scoring with public references (non-official). GPU app always stopped on exit.
#   bash scripts/asi_modal.sh <model_dir_in_volume> <served_name> [levels=b4] [run_id=asi-b4-run1]
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
MODAL=~/.conda/envs/qwen36/bin/modal; PY=~/.conda/envs/qwen36/bin/python
DIR=$1; NAME=$2; LV=${3:-b4}; RUN=${4:-asi-b4-run1}; KEY=asi-$NAME
OUT=results/$NAME/asi-bench/$RUN
ts() { date -u +%FT%TZ; }
mkdir -p "$OUT" logs/asi
if [ ! -f "$OUT/run_done" ]; then
  $PY -c "import modal; d=modal.Dict.from_name('qwen36-tunnels', create_if_missing=True); d.pop('$KEY', None)"
  $MODAL run --detach modal_app/serve_vllm.py::tunnel --model-dir "$DIR" --served-name "$NAME" --dict-key "$KEY" > "logs/asi/tunnel_$NAME.log" 2>&1 &
  APP=""
  trap '[ -n "$APP" ] && $MODAL app stop "$APP" -y >/dev/null 2>&1; echo "[$(ts)] tunnel $APP stopped"' EXIT
  for _ in $(seq 120); do APP=$(grep -oE "ap-[A-Za-z0-9]+" "logs/asi/tunnel_$NAME.log" | head -1 || true); [ -n "$APP" ] && break; sleep 5; done
  for _ in $(seq 240); do
    URL=$($PY -c "import modal; print(modal.Dict.from_name('qwen36-tunnels').get('$KEY') or '')")
    [ -n "$URL" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$URL/health")" = "200" ] && break; sleep 10
  done
  echo "[$(ts)] $NAME served at $URL; starting ASI ($LV)"
  $MODAL run modal_app/asi_runner.py::asi_run --served-name "$NAME" --api-base "$URL/v1" --levels "$LV" \
    --parallel 60 --run-id "$RUN" >> "logs/asi/run_$NAME.log" 2>&1
  $MODAL app stop "$APP" -y >/dev/null 2>&1; APP=""
  touch "$OUT/run_done"
fi
# pull outputs + the full instances (with references) and score locally (judge = OpenAI, task-declared models)
A=~/interview/eval_data/asi_runs; mkdir -p "$A"
[ -d "$A/instances_seed31415@0fa14219" ] || $MODAL volume get qwen36-asi "instances_seed31415@0fa14219" "$A/" >/dev/null
rm -rf "$A/$NAME-$RUN"; mkdir -p "$A/$NAME-$RUN"
$MODAL volume get qwen36-asi "out/$NAME/$RUN" "$A/$NAME-$RUN/" >/dev/null
RD=$A/$NAME-$RUN/$RUN
( cd ~/interview/eval_data/ASI-Bench && ~/.conda/envs/qwen36-asi/bin/asibench score --repo seed31415 --results-dir "$RD" \
    --instances-dir "$A/instances_seed31415@0fa14219" --tasks-dir tasks/ --judge-api-key-env OPENAI_API_KEY --parallel 8 ) \
  >> "logs/asi/score_$NAME.log" 2>&1
cp "$RD/local_score_seed31415.json" "$OUT/" 2>/dev/null
# audit: did the agent touch reference answers / graders / the network for answers?
grep -lE "hidden_truth|/reference/|custom_scorer|task_eval\.yaml|huggingface\.co/datasets/Apexintelligence" -r "$RD" --include=*.jsonl --include=*trajectory.json > "$OUT/audit_hits.txt" 2>/dev/null
echo "[$(ts)] ASI $NAME: $(python3 -c "import json;d=json.load(open('$OUT/local_score_seed31415.json'));print(d.get('total_score'),'/',d.get('total_max_score'))" 2>/dev/null); audit hits: $(wc -l < "$OUT/audit_hits.txt")"
