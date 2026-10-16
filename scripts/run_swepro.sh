#!/usr/bin/env bash
# SWE-bench Pro V2 HARD-51 with the official host-side mini-swe-agent (LLM loop on this host, sandbox has no
# network), Modal sandboxes, then the official patch_replay re-grade on pristine images (the reported score).
#   bash scripts/run_swepro.sh <served-model-name> [run_id] [extra harbor args]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; RUN=${2:-swepro-run1}; shift; [ $# -gt 0 ] && shift
SBP=$HOME/interview/eval_data/SWE-bench_Pro-os   # pinned at 66f92766 (configs/eval/swebench_pro.yaml)
export PATH=$HOME/.conda/envs/qwen36/bin:$PATH PYTHONNOUSERSITE=1 MSWEA_COST_TRACKING=ignore_errors
set -a; . ./.env; set +a
export OPENAI_API_KEY=$VLLM_API_KEY
OUT=results/$MODEL/swebench-pro/$RUN
mkdir -p "$OUT"
cp configs/eval/swebench_pro.yaml "$OUT/"
# agent config with the endpoint of this run (local vLLM by default, or a Modal tunnel via VLLM_URL)
sed "s#api_base: http://localhost:8100/v1#api_base: ${VLLM_URL:-http://localhost:8100}/v1#" configs/eval/swepro_mini_toolcall.yaml > "$OUT/agent_config.yaml"
IDS=$(sed 's/^/-i /' configs/eval/swebench_pro_v2_hard51_ids.txt | tr '\n' ' ')
J=$OUT/$RUN
if [ -f "$J/config.json" ] && ! python3 -c "import json,sys;sys.exit(0 if json.load(open('$J/result.json')).get('finished_at') else 1)" 2>/dev/null; then
  PYTHONPATH=$SBP/v2/tooling harbor job resume -p "$J" >> "$OUT/harbor.log" 2>&1
elif [ ! -f "$J/result.json" ]; then
  PYTHONPATH=$SBP/v2/tooling harbor run -p "$SBP/v2/tasks" $IDS -e modal --cpus limit --memory limit \
    -a host_mini_swe:HostMiniSweAgent -m "openai/$MODEL" \
    --ak config_file="$PWD/$OUT/agent_config.yaml" \
    -o "$OUT" --job-name "$RUN" "$@" >> "$OUT/harbor.log" 2>&1
fi
# authoritative score: replay each captured model.patch on a pristine image with the unchanged verifier
R=$OUT/${RUN}-regrade
if ! python3 -c "import json,sys;sys.exit(0 if json.load(open('$R/$(basename $R)/result.json')).get('finished_at') else 1)" 2>/dev/null; then
  PYTHONPATH=$SBP/v2/tooling harbor run -p "$SBP/v2/tasks" $IDS -e modal --cpus limit --memory limit \
    -a patch_replay:PatchReplayAgent -m replay --ak source_job="$J" \
    -o "$R" --job-name "$(basename $R)" -n 25 >> "$OUT/harbor_regrade.log" 2>&1
fi
