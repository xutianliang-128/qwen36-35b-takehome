#!/usr/bin/env bash
# Run Modal HLE for base and the reasoning expert, then the selected merge once results/dev/selection.json exists.
set -uo pipefail
cd "$(dirname "$0")/.."
bash scripts/hle_modal.sh "Qwen3.6-35B-A3B@995ad96e" qwen36-base
bash scripts/hle_modal.sh expert-e_reason expert-e_reason
echo "[$(date -u +%FT%TZ)] waiting for dev selection"
until [ -f results/dev/selection.json ]; do sleep 300; done
SEL=$(python3 -c "import json;print(json.load(open('results/dev/selection.json'))['selected'])")
echo "[$(date -u +%FT%TZ)] selected $SEL; building it on Modal"
until [ -f checkpoints/e_agent/final/adapter_model.safetensors ]; do sleep 60; done
export PYTHONNOUSERSITE=1
~/.conda/envs/qwen36/bin/modal volume put -f qwen36-models checkpoints/e_agent/final adapters/e_agent
ARGS=$(python3 -c "import json;c=json.load(open('/home/tianlix/interview/models/$SEL/merge_config.json'));print(' '.join(['--method',c['method'],'--density',str(c['density']),'--seed',str(c['seed']),'--lam',*map(str,c['lam'][:1] if c['method']!='linear' else c['lam'])]+sum([['--expert',f'{k}=/models/adapters/{k}'] for k in c['experts']],[])))")
~/.conda/envs/qwen36/bin/python scripts/modal_merge.py --out /models/$SEL $ARGS
bash scripts/hle_modal.sh "$SEL" "$SEL"
