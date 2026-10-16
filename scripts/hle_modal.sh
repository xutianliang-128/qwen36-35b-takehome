#!/usr/bin/env bash
# HLE for one model on Modal 2x H100 (vLLM behind a modal.forward TCP tunnel), then judge. Resumable.
# The GPU app is ALWAYS stopped on exit (success, failure or Ctrl-C).
#   bash scripts/hle_modal.sh <model_dir_in_volume> <served_name>
#   e.g. bash scripts/hle_modal.sh Qwen3.6-35B-A3B@995ad96e qwen36-base
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
MODAL=~/.conda/envs/qwen36/bin/modal
PY=~/.conda/envs/qwen36/bin/python
DIR=$1; NAME=$2
OUT=results/$NAME/hle/run1
mkdir -p "$OUT" logs
ts() { date -u +%FT%TZ; }

done_n() { [ -f "$OUT/responses.jsonl" ] && wc -l < "$OUT/responses.jsonl" || echo 0; }
if [ "$(done_n)" -lt 2500 ]; then
  $PY -c "import modal; d=modal.Dict.from_name('qwen36-tunnels', create_if_missing=True); d.pop('$NAME', None)"
  $MODAL run --detach modal_app/serve_vllm.py::tunnel --model-dir "$DIR" --served-name "$NAME" > "logs/modal_tunnel_$NAME.log" 2>&1 &
  APP=""
  trap '[ -n "$APP" ] && $MODAL app stop "$APP" -y >/dev/null 2>&1; echo "[$(ts)] modal app $APP stopped"' EXIT
  for _ in $(seq 120); do
    APP=$(grep -oE "ap-[A-Za-z0-9]+" "logs/modal_tunnel_$NAME.log" | head -1 || true)
    [ -n "$APP" ] && break; sleep 5
  done
  echo "[$(ts)] modal app $APP launched for $NAME"
  URL=""
  for _ in $(seq 240); do   # up to 40 min for image pull + weight load + compile
    URL=$($PY -c "import modal; print(modal.Dict.from_name('qwen36-tunnels').get('$NAME') or '')")
    if [ -n "$URL" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$URL/health")" = "200" ]; then break; fi
    sleep 10
  done
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$URL/health")" = "200" ] || { echo "[$(ts)] vLLM on Modal never became healthy"; exit 1; }
  echo "[$(ts)] serving $NAME at $URL"
  $PY - "$OUT/responses.jsonl" <<'EOF'
import json, sys, os
p = sys.argv[1]
if os.path.exists(p):
    good = []
    for l in open(p):
        try: json.loads(l); good.append(l)
        except Exception: pass
    open(p, "w").writelines(good)
EOF
  VLLM_URL=$URL $PY src/eval/hle_generate.py --model "$NAME" --out "$OUT" --max-tokens 8192 --concurrency 256 \
    >> "logs/eval_hle_$NAME.log" 2>&1
  [ "$(done_n)" -ge 2500 ] || { echo "[$(ts)] HLE $NAME incomplete: $(done_n)/2500"; exit 1; }
fi
echo "[$(ts)] judging $NAME"
PYTHONPATH=src/eval $PY src/eval/hle_judge.py --run "$OUT" >> "logs/judge_hle_$NAME.log" 2>&1
echo "[$(ts)] HLE $NAME done: $(python3 -c "import json;print(json.load(open('$OUT/summary.json'))['accuracy'])")"
