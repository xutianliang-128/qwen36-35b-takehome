#!/usr/bin/env bash
# Serve a model with vLLM on local GPUs (same flags as modal_app/serve_vllm.py).
#   bash scripts/serve_local.sh [model_path] [served_name] [gpus] [port]
# Logs: logs/serve/<served_name>.log ; stop with: pkill -f "[b]in/vllm serve"
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-$HOME/interview/models/Qwen3.6-35B-A3B@995ad96e}
NAME=${2:-qwen36-base}
GPUS=${3:-0,1}
PORT=${4:-8100}
TP=$(( $(tr -cd ',' <<<"$GPUS" | wc -c) + 1 ))
set -a; . ./.env; set +a
mkdir -p logs/serve
ENV_BIN=$HOME/.conda/envs/qwen36-vllm/bin
# ninja (and pip-installed nvcc) live in the env; flashinfer JIT-compiles sampling kernels on first use
# System /usr/bin/nvcc is CUDA 12.0 (too old for flashinfer flags); use the env's pip CUDA 13 toolkit.
export CUDA_HOME=$HOME/.conda/envs/qwen36-vllm/lib/python3.12/site-packages/nvidia/cu13
# flashinfer's JIT sampler still picks up system CUDA 12 headers -> use vLLM's PyTorch top-k/top-p sampler.
# Same setting on Modal (modal_app/serve_vllm.py) so every evaluated model uses one sampler implementation.
export VLLM_USE_FLASHINFER_SAMPLER=0
# expandable_segments breaks the CUDA IPC handles of vLLM custom all-reduce ("custom_all_reduce.cuh: invalid argument")
unset PYTORCH_CUDA_ALLOC_CONF
export PATH=$CUDA_HOME/bin:$ENV_BIN:$PATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPUS
setsid nohup "$HOME/.conda/envs/qwen36-vllm/bin/vllm" serve "$MODEL" 9>&- \
  --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --api-key "$VLLM_API_KEY" \
  --host 127.0.0.1 --port "$PORT" > "logs/serve/$NAME.log" 2>&1 &
echo "vllm pid $! -> http://localhost:$PORT (log logs/serve/$NAME.log)"
