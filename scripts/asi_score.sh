#!/usr/bin/env bash
# Score downloaded ASI-Bench runs locally (seed31415 public references; non-official) + audit agent logs.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
# task judges call gpt-5.5 with temperature=0, which it rejects in reasoning mode; let LiteLLM drop unsupported params
export LITELLM_DROP_PARAMS=True
A=~/interview/eval_data/asi_runs/v2; INST="$A/instances_seed31415@0fa14219"
for NAME in "$@"; do
  RD=$A/out/$NAME/asi-b4-run1; OUT=results/$NAME/asi-bench/asi-b4-run1
  ( cd ~/interview/eval_data/ASI-Bench && ~/.conda/envs/qwen36-asi/bin/asibench score --repo seed31415 --results-dir "$RD" \
      --instances-dir "$INST" --tasks-dir tasks/ --judge-api-key-env OPENAI_API_KEY --parallel 8 ) >> "logs/asi/score_$NAME.log" 2>&1
  cp "$RD/local_score_seed31415.json" "$OUT/" 2>/dev/null
  grep -lE "hidden_truth|/reference/|custom_scorer|task_eval\.yaml|huggingface\.co/datasets/Apexintelligence" -r "$RD" \
    --include=*.jsonl --include=*trajectory.json > "$OUT/audit_hits.txt" 2>/dev/null
  echo "[$(date -u +%FT%TZ)] ASI $NAME: $(python3 -c "import json;d=json.load(open('$OUT/local_score_seed31415.json'));print(round(100*d['total_score']/d['total_max_score'],2),'%', 'scored',d.get('scored_instance_count'),'scorer_errors',d.get('scorer_error_count'))" 2>/dev/null) audit_hits=$(wc -l < "$OUT/audit_hits.txt")"
done
