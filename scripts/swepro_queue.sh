#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
SEL=$(python3 -c "import json;print(json.load(open('results/dev/selection.json'))['selected'])")
bash scripts/swepro_modal.sh "Qwen3.6-35B-A3B@995ad96e" qwen36-base &
bash scripts/swepro_modal.sh "$SEL" "$SEL" &
wait
