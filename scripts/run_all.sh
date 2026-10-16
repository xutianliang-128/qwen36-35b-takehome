#!/usr/bin/env bash
# Idempotent experiment driver. Every stage checks a "done" marker first, so rerunning it (e.g. from
# cron @reboot after the server restarts) resumes where it stopped. A lock prevents concurrent copies.
#   setsid nohup bash scripts/run_all.sh >> logs/run_all.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
exec 9>"/tmp/qwen36_run_all.$(id -u).lock"
flock -n 9 || { echo "[$(date -u +%FT%TZ)] run_all already running"; exit 0; }
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
M=/home/tianlix/interview/models
BASE=$M/Qwen3.6-35B-A3B@995ad96e
PYT=~/.conda/envs/qwen36-train/bin/python
PYQ=~/.conda/envs/qwen36/bin/python
ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

served_name() {
  curl -s -m 5 -H "Authorization: Bearer $VLLM_API_KEY" localhost:8100/v1/models 2>/dev/null |
    $PYQ -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null
}
stop_vllm() {
  pkill -f "[q]wen36-vllm/bin/vllm serve"
  # only our serving GPUs: GPU2 may be used by other users of this shared server
  while nvidia-smi -i 0,1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; do sleep 10; done
}
serve() {  # serve <model_path> <served_name>
  [ "$(served_name)" = "$2" ] && return 0
  log "serving $2 from $1"
  stop_vllm
  bash scripts/serve_local.sh "$1" "$2" >/dev/null
  local t=0
  until [ "$(served_name)" = "$2" ]; do
    sleep 15; t=$((t + 15))
    [ $t -gt 1200 ] && { log "vLLM failed to start for $2"; exit 1; }
  done
}
hle_done() { [ -f "results/$1/hle/run1/responses.jsonl" ] && [ "$(wc -l < "results/$1/hle/run1/responses.jsonl")" -ge 2500 ]; }
hle() {  # hle <model_path> <served_name>: generate (resumable) then judge in the background
  if ! hle_done "$2"; then
    serve "$1" "$2"
    local pidf="results/$2/hle/run1/gen.pid"
    if ! { [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; }; then
      $PYQ - "results/$2/hle/run1/responses.jsonl" <<'EOF'   # drop a line cut off by a crash
import json, sys, os
p = sys.argv[1]
if os.path.exists(p):
    good = []
    for l in open(p):
        try: json.loads(l); good.append(l)
        except Exception: pass
    open(p, "w").writelines(good)
EOF
      bash scripts/run_hle.sh "$2" run1
    fi
    log "HLE $2: waiting for generation"
    while kill -0 "$(cat "$pidf")" 2>/dev/null; do sleep 60; done
    hle_done "$2" || { log "HLE $2 generation exited before 2500 responses"; exit 1; }
  fi
  if [ ! -f "results/$2/hle/run1/summary.json" ] && ! pgrep -f "[h]le_judge.py --run results/$2/" >/dev/null; then
    log "HLE $2: judging in background"
    PYTHONPATH=src/eval setsid nohup $PYQ src/eval/hle_judge.py --run "results/$2/hle/run1" \
      >> "logs/judge_hle_$2.log" 2>&1 < /dev/null &
  fi
}
train() {  # train <expert>
  [ -f "checkpoints/$1/final/adapter_model.safetensors" ] && return 0
  log "training $1"
  stop_vllm
  rm -f "checkpoints/$1/metrics.jsonl"   # an interrupted run restarts from step 0
  $PYT src/train/train_lora.py --config "configs/train/$1.yaml" > "logs/train/$1.log" 2>&1 \
    || { log "training $1 failed"; exit 1; }
}
merge() {  # merge <out_name> <merge args...>
  [ -f "$M/$1/merge_config.json" ] && return 0
  log "merge $1"
  $PYT src/merge/merge_experts.py --out "$M/$1" "${@:2}" > "logs/merge/$1.log" 2>&1 || { log "merge $1 failed"; exit 1; }
}

mkdir -p logs/train logs/merge
log "run_all start"
R=e_reason=checkpoints/e_reason/final
A=e_agent=checkpoints/e_agent/final

# 1. HLE runs on Modal H100 (scripts/hle_modal.sh), not here: frees the local GPUs for training + agent evals
# 2. agent expert
[ -f data/agent/manifest.sha256 ] || sha256sum data/agent/sft_train.jsonl data/agent/sft_val.jsonl > data/agent/manifest.sha256
train e_agent
# 2b. CONTROL: single joint LoRA on the union of both experts' data (same exposure, same hyper-parameters)
train e_joint
# 3. experts as full checkpoints + merge variants (dense dW = (alpha/r) B A)
merge expert-e_reason --method linear --lam 1.0 --expert $R
merge expert-e_agent --method linear --lam 1.0 --expert $A
merge expert-e_joint --method linear --lam 1.0 --expert e_joint=checkpoints/e_joint/final
merge merged-linear-1.0 --method linear --lam 1.0 --expert $R --expert $A
merge merged-ties-d0.2 --method ties --density 0.2 --lam 1.0 --expert $R --expert $A
merge merged-dare-d0.5 --method dare_ties --density 0.5 --lam 1.0 --seed 0 --expert $R --expert $A
log "stage 3 done"

# 4. choose the merge on dev data only: E_reason dev accuracy (served) + E_agent held-out loss (HF forward)
CANDS="merged-linear-1.0 merged-ties-d0.2 merged-dare-d0.5"
mkdir -p results/dev
for v in $CANDS expert-e_joint; do
  [ -f "results/dev/$v/reason_dev/gen_stats.json" ] && continue
  serve "$M/$v" "$v"
  log "dev: reason accuracy for $v"
  mkdir -p "results/dev/$v/reason_dev"
  VLLM_URL=http://localhost:8100 $PYQ src/data/gen_rft.py --model "$v" --pool data/reason/pool_dev.jsonl \
    --out "results/dev/$v/reason_dev" --k 1 --max-tokens 8192 --concurrency 128 >> "logs/dev_reason_$v.log" 2>&1
done
if [ ! -f results/dev/selection.json ]; then
  stop_vllm
  log "dev: held-out losses"
  $PYT src/eval/val_loss.py --out results/dev/val_loss.json --models qwen36-base=$BASE \
    expert-e_reason=$M/expert-e_reason expert-e_agent=$M/expert-e_agent expert-e_joint=$M/expert-e_joint \
    $(for v in $CANDS; do printf "%s=%s " "$v" "$M/$v"; done) >> logs/dev_val_loss.log 2>&1 || { log "val_loss failed"; exit 1; }
  $PYQ src/merge/select_merge.py --candidates $CANDS >> logs/dev_selection.log 2>&1 || { log "selection failed"; exit 1; }
fi
SEL=$($PYQ -c "import json;print(json.load(open('results/dev/selection.json'))['selected'])")
log "selected merge: $SEL"

# 5. (HLE for base / merged / E_reason runs on Modal, see scripts/hle_modal.sh)

# 6. Terminal-Bench 2.0, one run per model, identical serving (local vLLM, n=32 concurrent trials) and Modal sandboxes
tb2() {  # tb2 <model_path> <served_name>
  local job=results/$2/terminal-bench/tb2-run1/tb2-run1
  if [ -f "$job/result.json" ] && $PYQ -c "import json,sys;sys.exit(0 if json.load(open('$job/result.json')).get('finished_at') else 1)"; then
    return 0
  fi
  serve "$1" "$2"
  export PATH=$HOME/.conda/envs/qwen36/bin:$PATH
  if [ -f "$job/config.json" ]; then
    log "TB2 $2: resuming"
    # the agent's LiteLLM client must get the vLLM key (not the OpenAI judge key loaded from .env), else 401s;
    # trials that failed on auth are dropped and re-run
    OPENAI_API_KEY=$VLLM_API_KEY harbor job resume -p "$job" -f AuthenticationError >> "logs/tb2_$2.log" 2>&1
  else
    log "TB2 $2: starting"
    VLLM_URL=http://localhost:8100 bash scripts/run_tb2.sh "$2" tb2-run1 -k 1 -n 32 >> "logs/tb2_$2.log" 2>&1
  fi
}
tb2 "$BASE" qwen36-base
tb2 "$M/$SEL" "$SEL"

# 7. SWE-bench Pro V2 HARD-51 (official host-side mini-swe-agent + patch_replay re-grade); failures are non-fatal
swepro() {  # swepro <model_path> <served_name>
  local r=results/$2/swebench-pro/swepro-run1/swepro-run1-regrade/swepro-run1-regrade/result.json
  if [ -f "$r" ] && $PYQ -c "import json,sys;sys.exit(0 if json.load(open('$r')).get('finished_at') else 1)"; then return 0; fi
  serve "$1" "$2"
  log "SWE-Pro $2"
  bash scripts/run_swepro.sh "$2" swepro-run1 -n 26 || log "SWE-Pro $2 failed (continuing)"
}
swepro "$BASE" qwen36-base
swepro "$M/$SEL" "$SEL"
# E_agent SWE-Pro is NOT run locally: base/merged used Modal H100 and local A6000 inference times out under the
# 50-min budget (two voided runs, see README 6.9). Re-run only with scripts/swepro_modal.sh on the same hardware.
tb2 "$M/expert-e_agent" expert-e_agent
stop_vllm
log "all stages done"
