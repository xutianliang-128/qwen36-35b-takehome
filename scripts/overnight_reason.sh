#!/usr/bin/env bash
# Overnight E_reason pipeline: wait for rejection sampling -> build SFT set (HLE-decontaminated)
# -> stop local vLLM to free GPUs -> train LoRA on all 3 GPUs.
#   nohup bash scripts/overnight_reason.sh > logs/train/overnight_reason.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
ts() { date -u +%FT%TZ; }

if [ ! -f data/reason/gen_stats.json ]; then
  # Started here and waited on by PID: `pgrep -f gen_rft.py` also matches unrelated shells mentioning the name.
  echo "[$(ts)] starting gen_rft.py (resumes from data/reason/samples.jsonl)"
  VLLM_URL=http://localhost:8100 ~/.conda/envs/qwen36/bin/python src/data/gen_rft.py \
    --pool data/reason/pool_train.jsonl --out data/reason --max-questions "${MAX_QUESTIONS:-600}" >> logs/data/gen_reason.log 2>&1
fi
[ -f data/reason/gen_stats.json ] || { echo "[$(ts)] gen_rft.py exited without gen_stats.json; see logs/data/gen_reason.log"; exit 1; }
cat data/reason/gen_stats.json

echo "[$(ts)] building SFT set (needs HF_TOKEN for HLE decontamination)"
until grep -q "^HF_TOKEN=." .env; do
  echo "[$(ts)] HF_TOKEN missing in .env; retrying in 10 min (training will not start without decontamination)"
  sleep 600
done
set -a; . ./.env; set +a
~/.conda/envs/qwen36/bin/python src/data/build_sft_reason.py --samples data/reason/samples.jsonl --out data/reason || exit 1
sha256sum data/reason/sft_train.jsonl data/reason/sft_val.jsonl data/reason/pool_train.jsonl data/reason/pool_dev.jsonl \
  > data/reason/manifest.sha256

echo "[$(ts)] stopping local vLLM to free GPUs"
pkill -f "[q]wen36-vllm/bin/vllm serve"
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; do sleep 10; done

echo "[$(ts)] training E_reason"
mkdir -p logs/train
~/.conda/envs/qwen36-train/bin/python src/train/train_lora.py --config configs/train/e_reason.yaml \
  > logs/train/e_reason.log 2>&1
echo "[$(ts)] training exited with $?"
