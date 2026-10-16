#!/usr/bin/env bash
# Reasoning-dev accuracy (212 held-out WebInstruct questions, k=1, 8K) for one model on Modal 2x H100.
#   bash scripts/dev_reason_modal.sh <model_dir_in_volume> <served_name>
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
MODAL=~/.conda/envs/qwen36/bin/modal; PY=~/.conda/envs/qwen36/bin/python
DIR=$1; NAME=$2; OUT=results/dev/$NAME/reason_dev
[ -f "$OUT/gen_stats.json" ] && exit 0
mkdir -p "$OUT"
$PY -c "import modal; d=modal.Dict.from_name('qwen36-tunnels', create_if_missing=True); d.pop('$NAME', None)"
$MODAL run --detach modal_app/serve_vllm.py::tunnel --model-dir "$DIR" --served-name "$NAME" > "logs/modal_tunnel_dev_$NAME.log" 2>&1 &
APP=""
trap '[ -n "$APP" ] && $MODAL app stop "$APP" -y >/dev/null 2>&1' EXIT
for _ in $(seq 120); do APP=$(grep -oE "ap-[A-Za-z0-9]+" "logs/modal_tunnel_dev_$NAME.log" | head -1 || true); [ -n "$APP" ] && break; sleep 5; done
for _ in $(seq 240); do
  URL=$($PY -c "import modal; print(modal.Dict.from_name('qwen36-tunnels').get('$NAME') or '')")
  [ -n "$URL" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$URL/health")" = "200" ] && break; sleep 10
done
VLLM_URL=$URL $PY src/data/gen_rft.py --model "$NAME" --pool data/reason/pool_dev.jsonl --out "$OUT" --k 1 \
  --max-tokens 8192 --concurrency 128 >> "logs/dev_reason_$NAME.log" 2>&1
