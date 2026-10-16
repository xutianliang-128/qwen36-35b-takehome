#!/usr/bin/env bash
# Run Terminal-Bench 2.0 (Harbor + Terminus-2, Modal sandboxes) against the vLLM endpoint.
# Settings come from configs/eval/terminal_bench.yaml; keep them identical for every model.
#
#   bash scripts/run_tb2.sh <served-model-name> <run_id> [extra harbor args, e.g. -i fix-git -k 1 -n 32]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; RUN_ID=$2; shift 2
HARBOR=${HARBOR:-$HOME/.conda/envs/qwen36/bin/harbor}
TASKS=${TASKS:-$HOME/interview/eval_data/terminal-bench-2.0}
VLLM_URL=${VLLM_URL:-https://tianlix--qwen36-vllm-serve.us-east.modal.run}
set -a; . ./.env; set +a
export OPENAI_API_KEY=$VLLM_API_KEY   # LiteLLM openai/* provider -> our vLLM key

OUT=results/$MODEL/terminal-bench/$RUN_ID
mkdir -p "$OUT"
cp configs/eval/terminal_bench.yaml "$OUT/eval_config.yaml"

"$HARBOR" run \
  -p "$TASKS" \
  -a terminus-2 -m "openai/$MODEL" \
  --ak "api_base=$VLLM_URL/v1" \
  --ak temperature=1.0 \
  --ak 'llm_call_kwargs={"top_p": 0.95, "max_tokens": 81920, "extra_body": {"top_k": 20}}' \
  --ak 'model_info={"max_input_tokens": 180224, "max_output_tokens": 81920, "input_cost_per_token": 0, "output_cost_per_token": 0}' \
  -e modal --override-cpus 32 --override-memory-mb 49152 --cpus limit --memory limit \
  -o "$OUT" --job-name "$RUN_ID" \
  "$@" 2>&1 | tee "$OUT/harbor.log"
