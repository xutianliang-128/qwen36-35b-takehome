#!/usr/bin/env bash
# Deploy vLLM on Modal, wait for health, run one request, then ALWAYS stop the app.
# Usage (repo root): bash scripts/smoke_modal.sh    [MAX_WAIT_S=900]
set -uo pipefail
cd "$(dirname "$0")/.."
MODAL=${MODAL:-$HOME/.conda/envs/qwen36/bin/modal}
PY=${PY:-$HOME/.conda/envs/qwen36/bin/python}
MAX_WAIT_S=${MAX_WAIT_S:-900}
APP=qwen36-vllm
set -a; . ./.env; set +a
export VLLM_URL=https://tianlix--qwen36-vllm-serve.us-east.modal.run

trap '"$MODAL" app stop "$APP" -y >/dev/null 2>&1; echo "[$(date -u +%T)] app stopped"' EXIT

"$MODAL" deploy modal_app/serve_vllm.py | tail -2
t0=$(date +%s)
echo "[$(date -u +%T)] waiting for /health (max ${MAX_WAIT_S}s)"
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 30 "$VLLM_URL/health")" = "200" ]; do
  if [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT_S" ]; then
    echo "TIMEOUT: not healthy after ${MAX_WAIT_S}s; last server logs:"
    timeout 20 "$MODAL" app logs "$APP" 2>&1 | grep -E "Error|error" | tail -5
    exit 1
  fi
  sleep 15
done
echo "[$(date -u +%T)] healthy, cold start $(( $(date +%s) - t0 ))s"
echo "request without key -> HTTP $(curl -s -o /dev/null -w '%{http_code}' "$VLLM_URL/v1/models") (expect 401)"
"$PY" scripts/smoke_test.py
