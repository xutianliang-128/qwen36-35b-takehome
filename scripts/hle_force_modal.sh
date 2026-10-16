#!/usr/bin/env bash
# Thinking-budget forcing pass for one model's finished HLE run on Modal, then judge the forced run.
# GPU app is always stopped on exit.   bash scripts/hle_force_modal.sh <model_dir_in_volume> <served_name>
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
MODAL=~/.conda/envs/qwen36/bin/modal; PY=~/.conda/envs/qwen36/bin/python
DIR=$1; NAME=$2; RUN=results/$NAME/hle/run1; BF=${RUN}_bf
ts() { date -u +%FT%TZ; }
n_bf() { [ -f "$BF/responses.jsonl" ] && wc -l < "$BF/responses.jsonl" || echo 0; }
if [ "$(n_bf)" -lt 2500 ]; then
  $PY -c "import modal; d=modal.Dict.from_name('qwen36-tunnels', create_if_missing=True); d.pop('$NAME-bf', None)"
  $MODAL run --detach modal_app/serve_vllm.py::tunnel --model-dir "$DIR" --served-name "$NAME" > "logs/modal_tunnel_$NAME-bf.log" 2>&1 &
  APP=""
  trap '[ -n "$APP" ] && $MODAL app stop "$APP" -y >/dev/null 2>&1; echo "[$(ts)] modal app $APP stopped"' EXIT
  for _ in $(seq 120); do APP=$(grep -oE "ap-[A-Za-z0-9]+" "logs/modal_tunnel_$NAME-bf.log" | head -1 || true); [ -n "$APP" ] && break; sleep 5; done
  URL=""
  for _ in $(seq 240); do
    URL=$($PY -c "import modal; print(modal.Dict.from_name('qwen36-tunnels').get('$NAME') or '')")
    if [ -n "$URL" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$URL/health")" = "200" ]; then break; fi
    sleep 10
  done
  echo "[$(ts)] forcing $NAME via $URL"
  PYTHONPATH=src/eval VLLM_URL=$URL $PY src/eval/hle_budget_force.py --model "$NAME" --run "$RUN" >> "logs/hle_force_$NAME.log" 2>&1
  [ "$(n_bf)" -ge 2500 ] || { echo "[$(ts)] forcing incomplete for $NAME"; exit 1; }
  $MODAL app stop "$APP" -y >/dev/null 2>&1; APP=""
fi
PYTHONPATH=src/eval $PY src/eval/hle_judge.py --run "$BF" >> "logs/judge_hle_$NAME-bf.log" 2>&1
echo "[$(ts)] HLE-bf $NAME: $(python3 -c "import json;s=json.load(open('$BF/summary.json'));print(s['accuracy'],'truncated->forced')")"
