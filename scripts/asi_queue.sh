#!/usr/bin/env bash
# ASI-Bench B4 (seed31415) for base and the selected merge, in parallel on Modal; resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
SEL=$(python3 -c "import json;print(json.load(open('results/dev/selection.json'))['selected'])")
bash scripts/asi_modal.sh "Qwen3.6-35B-A3B@995ad96e" qwen36-base b4 asi-b4-run1 &
bash scripts/asi_modal.sh "$SEL" "$SEL" b4 asi-b4-run1 &
wait
