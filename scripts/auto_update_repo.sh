#!/usr/bin/env bash
# Wait for E_agent's TB2 run, then refresh tables / README cells / report and push to the submission repo.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
PY=~/.conda/envs/qwen36/bin/python
J=results/expert-e_agent/terminal-bench/tb2-run1/tb2-run1/result.json
# finished AND no infrastructure auth failures (a resume without the vLLM key once produced 58 x 401 "finished" trials)
until [ -f "$J" ] && $PY -c "import json,sys,glob;d=json.load(open('$J'));ok=d.get('finished_at') and not any((json.load(open(f)).get('exception_info') or {}).get('exception_type')=='AuthenticationError' for f in glob.glob('$(dirname $J)/*/result.json'));sys.exit(0 if ok else 1)"; do sleep 300; done
echo "[$(date -u +%FT%TZ)] E_agent TB2 finished"
$PY scripts/summarize.py > /dev/null 2>&1
$PY - <<'PYEOF'
import csv
S = {(r["model"], r["benchmark"]): r for r in csv.DictReader(open("results/summary.csv"))}
r = S[("expert-e_agent", "terminal_bench")]
v = f"{r['score_pct']}（{float(r['diff_vs_base_pp']):+.2f} [{r['ci95_low']}, {r['ci95_high']}] vs Base）" if r.get("diff_vs_base_pp") else r["score_pct"]
ve = v.replace("（", " (").replace("）", ")")
zh = open("README.md").read(); en = open("README_en.md").read()
zh = zh.replace("| E_agent | 终端 / agent 专家 | — | 运行中（目标） |", "| E_agent | 终端 / agent 专家 | — | ✅（目标） |")
zh = zh.replace("| Terminal-Bench 2.0，89 题（成功率，1 次运行） | 41.57 | — | — |", f"| Terminal-Bench 2.0，89 题（成功率，1 次运行） | 41.57 | — | {v} |")
zh = zh.replace("E_agent 的 Terminal-Bench 目标评测正在运行（与 Base / 合版相同的本地推理条件），完成后更新", "E_agent 的 Terminal-Bench 目标评测已完成（与 Base / 合版相同的本地推理条件）")
en = en.replace("| E_agent | terminal / agent expert | — | running (target) |", "| E_agent | terminal / agent expert | — | ✅ (target) |")
en = en.replace("| Terminal-Bench 2.0, 89 tasks (success rate, 1 run) | 41.57 | — | — |", f"| Terminal-Bench 2.0, 89 tasks (success rate, 1 run) | 41.57 | — | {ve} |")
en = en.replace("E_agent's Terminal-Bench target evaluation is running (same local inference setup as Base / merged) and will be added when done", "E_agent's Terminal-Bench target evaluation is complete (same local inference setup as Base / merged)")
open("README.md", "w").write(zh); open("README_en.md", "w").write(en)
print("E_agent TB2:", v)
PYEOF
$PY scripts/plot_curves.py > /dev/null 2>&1
sed -i 's/其 TB2 仍在运行。/其 TB2 已完成（见仓库）。/' scripts/make_report.py
$PY scripts/make_report.py > /dev/null 2>&1
git ls-files -co --exclude-standard > /tmp/qwen36_files.txt
cd ~/interview/submit && rsync -a --files-from=/tmp/qwen36_files.txt ~/interview/qwen36-35b-takehome/ ./ && git add -A
set -a; . ~/interview/qwen36-35b-takehome/.env; set +a
for v in VLLM_API_KEY OPENAI_API_KEY HF_TOKEN HF_WRITE_TOKEN; do val="${!v}"; git grep -lF -- "$val" >/dev/null 2>&1 && { echo "secret found, not pushing"; exit 1; }; done
git -c user.name="xutianliang-128" -c user.email="tianlix@umich.edu" commit -q -m "Add E_agent Terminal-Bench 2.0 result; refresh tables and report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main && echo "[$(date -u +%FT%TZ)] pushed $(git log -1 --format=%H)"
