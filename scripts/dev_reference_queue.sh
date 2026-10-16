#!/usr/bin/env bash
# After the Modal HLE work, fill in dev reasoning accuracy for the references (base, E_reason, E_agent).
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "[h]le_force_queue.sh" >/dev/null || pgrep -f "[h]le_modal_queue.sh" >/dev/null; do sleep 120; done
bash scripts/dev_reason_modal.sh "Qwen3.6-35B-A3B@995ad96e" qwen36-base
bash scripts/dev_reason_modal.sh expert-e_reason expert-e_reason
[ -f results/dev/expert-e_agent/reason_dev/gen_stats.json ] || ~/.conda/envs/qwen36/bin/python scripts/modal_merge.py \
  --out /models/expert-e_agent --method linear --lam 1.0 --expert e_agent=/models/adapters/e_agent
bash scripts/dev_reason_modal.sh expert-e_agent expert-e_agent
PYTHONNOUSERSITE=1 ~/.conda/envs/qwen36/bin/python scripts/plot_curves.py
