#!/usr/bin/env bash
# Budget-forcing passes in order: base, E_reason (after its plain run is judged), then the selected merge.
set -uo pipefail
cd "$(dirname "$0")/.."
bash scripts/hle_force_modal.sh "Qwen3.6-35B-A3B@995ad96e" qwen36-base
until [ -f results/expert-e_reason/hle/run1/summary.json ]; do sleep 120; done
bash scripts/hle_force_modal.sh expert-e_reason expert-e_reason
until [ -f results/dev/selection.json ]; do sleep 300; done
SEL=$(python3 -c "import json;print(json.load(open('results/dev/selection.json'))['selected'])")
until [ -f "results/$SEL/hle/run1/summary.json" ]; do sleep 120; done
bash scripts/hle_force_modal.sh "$SEL" "$SEL"
