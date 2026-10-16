#!/usr/bin/env bash
# After base HLE: free GPUs -> train E_agent -> materialize expert -> build merge variants.
#   setsid nohup bash scripts/pipeline_agent_merge.sh <hle_pid> > logs/train/pipeline_agent_merge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
ts() { date -u +%FT%TZ; }
M=/home/tianlix/interview/models
PY=~/.conda/envs/qwen36-train/bin/python
echo "[$(ts)] waiting for HLE pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(ts)] HLE finished; stopping vLLM"
pkill -f "[q]wen36-vllm/bin/vllm serve"
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; do sleep 10; done
sha256sum data/agent/sft_train.jsonl data/agent/sft_val.jsonl > data/agent/manifest.sha256
echo "[$(ts)] training E_agent"
$PY src/train/train_lora.py --config configs/train/e_agent.yaml > logs/train/e_agent.log 2>&1 || { echo "[$(ts)] E_agent training failed"; exit 1; }
echo "[$(ts)] materializing expert + merges"
R=e_reason=checkpoints/e_reason/final; A=e_agent=checkpoints/e_agent/final
$PY src/merge/merge_experts.py --method linear --lam 1.0 --expert $A --out $M/expert-e_agent > logs/merge/expert-e_agent.log 2>&1
$PY src/merge/merge_experts.py --method linear --lam 1.0 --expert $R --expert $A --out $M/merged-linear-1.0 > logs/merge/merged-linear-1.0.log 2>&1
$PY src/merge/merge_experts.py --method ties --density 0.2 --lam 1.0 --expert $R --expert $A --out $M/merged-ties-d0.2 > logs/merge/merged-ties-d0.2.log 2>&1
$PY src/merge/merge_experts.py --method dare_ties --density 0.5 --lam 1.0 --seed 0 --expert $R --expert $A --out $M/merged-dare-d0.5 > logs/merge/merged-dare-d0.5.log 2>&1
echo "[$(ts)] done"
