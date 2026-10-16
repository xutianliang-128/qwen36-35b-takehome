#!/usr/bin/env bash
# SWE-bench Pro HARD-51 for one model with inference on Modal 2x H100 (tunnel); host-side agent + Modal sandboxes.
#   bash scripts/swepro_modal.sh <model_dir_in_volume> <served_name>
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
MODAL=~/.conda/envs/qwen36/bin/modal; PY=~/.conda/envs/qwen36/bin/python
DIR=$1; NAME=$2; KEY=swe-$NAME
ts() { date -u +%FT%TZ; }
R=results/$NAME/swebench-pro/swepro-run1/swepro-run1-regrade/swepro-run1-regrade/result.json
if [ -f "$R" ] && $PY -c "import json,sys;sys.exit(0 if json.load(open('$R')).get('finished_at') else 1)"; then exit 0; fi
$PY -c "import modal; d=modal.Dict.from_name('qwen36-tunnels', create_if_missing=True); d.pop('$KEY', None)"
mkdir -p logs/swe
$MODAL run --detach modal_app/serve_vllm.py::tunnel --model-dir "$DIR" --served-name "$NAME" --dict-key "$KEY" > "logs/swe/tunnel_$NAME.log" 2>&1 &
APP=""
trap '[ -n "$APP" ] && $MODAL app stop "$APP" -y >/dev/null 2>&1; echo "[$(ts)] tunnel $APP stopped"' EXIT
for _ in $(seq 120); do APP=$(grep -oE "ap-[A-Za-z0-9]+" "logs/swe/tunnel_$NAME.log" | head -1 || true); [ -n "$APP" ] && break; sleep 5; done
for _ in $(seq 240); do
  URL=$($PY -c "import modal; print(modal.Dict.from_name('qwen36-tunnels').get('$KEY') or '')")
  [ -n "$URL" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$URL/health")" = "200" ] && break; sleep 10
done
echo "[$(ts)] $NAME served at $URL; SWE-Pro starting"
VLLM_URL=$URL bash scripts/run_swepro.sh "$NAME" swepro-run1 -n 26
echo "[$(ts)] SWE-Pro $NAME done"
