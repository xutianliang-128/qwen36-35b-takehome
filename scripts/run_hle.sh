#!/usr/bin/env bash
# HLE generation for one served model with the frozen settings in configs/eval/hle.yaml (detached).
#   bash scripts/run_hle.sh <served-model-name> [run_id]      -> results/<model>/hle/<run_id>/
# Writes the PID to results/<model>/hle/<run_id>/gen.pid (stop with: kill "$(cat .../gen.pid)").
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; RUN=${2:-run1}
OUT=results/$MODEL/hle/$RUN
mkdir -p "$OUT" logs
set -a; . ./.env; set +a
PYTHONNOUSERSITE=1 VLLM_URL=${VLLM_URL:-http://localhost:8100} setsid nohup "$HOME/.conda/envs/qwen36/bin/python" \
  src/eval/hle_generate.py --model "$MODEL" --out "$OUT" --max-tokens 8192 --concurrency 128 \
  > "logs/eval_hle_$MODEL.log" 2>&1 < /dev/null &
echo $! > "$OUT/gen.pid"
echo "HLE $MODEL pid $(cat "$OUT/gen.pid") -> $OUT"
