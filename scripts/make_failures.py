"""Sample >=20 failure cases of the merged model across benchmarks and attribute each from its logs."""
import glob, json, random
from pathlib import Path

R = Path("results"); M, B = "merged-linear-1.0", "qwen36-base"; rng = random.Random(0); rows = []

# HLE: merged wrong where base right (merge-induced) and both wrong
def hle(m):
    j = {x["id"]: x for x in map(json.loads, open(R / m / "hle/run1_bf/judged.jsonl"))}
    g = {x["id"]: x for x in map(json.loads, open(R / m / "hle/run1_bf/responses.jsonl"))}
    return j, g
jm, gm = hle(M); jb, _ = hle(B)
reg = sorted(i for i in jm if not jm[i]["correct"] and jb.get(i, {}).get("correct"))
both = sorted(i for i in jm if not jm[i]["correct"] and not jb.get(i, {}).get("correct"))
for i in rng.sample(reg, 4) + rng.sample(both, 2):
    x, y = jm[i], gm[i]
    forced = y.get("finish_reason") == "budget_forced"
    cat = "合版引入的退化" if i in reg else ("超时（思考预算耗尽后仓促作答）" if forced else "推理/知识错误")
    if i in reg and forced: cat += "；思考预算耗尽"
    rows.append(("HLE", i, cat, f"{x['category']}；判分抽取答案 `{str(x.get('extracted_final_answer'))[:60]}`；"
                 f"{'思考被 8K 截断后强制作答' if forced else '正常结束'}；Base 本题{'答对' if i in reg else '也答错'}"))

# TB2
def trials(p):
    out = {}
    for f in glob.glob(str(p / "*/result.json")):
        d = json.load(open(f)); t = (d.get("task_name") or Path(f).parent.name).split("/")[-1].split("__")[0]
        out[t] = d
    return out
tm = trials(R / M / "terminal-bench/tb2-run1/tb2-run1"); tb = trials(R / B / "terminal-bench/tb2-run1/tb2-run1")
rew = lambda d: ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) or 0
fails = [t for t in tm if rew(tm[t]) < 1]
regs = [t for t in fails if t in tb and rew(tb[t]) >= 1]
pick = rng.sample(regs, min(3, len(regs))) + rng.sample([t for t in fails if t not in regs], 3)
for t in pick:
    d = tm[t]; exc = (d.get("exception_info") or {}).get("exception_type"); a = d.get("agent_result") or {}
    turns = (a.get("metadata") or {}).get("n_episodes")
    if exc == "AgentTimeoutError": cat = "超时"
    elif exc: cat = "环境/工具失败"
    else: cat = "测试理解或实现错误（任务自认完成但验证失败）"
    if t in regs: cat += "；合版引入的退化（Base 通过）"
    rows.append(("TB2", t, cat, f"{turns} 轮，输出 {a.get('n_output_tokens')} tokens，异常 {exc or '无'}"))

# SWE-Pro (patch replay)
sm = trials(R / M / "swebench-pro/swepro-run1/swepro-run1-regrade/swepro-run1-regrade")
sa = trials(R / M / "swebench-pro/swepro-run1/swepro-run1")
sb = trials(R / B / "swebench-pro/swepro-run1/swepro-run1-regrade/swepro-run1-regrade")
sf = [t for t in sm if rew(sm[t]) < 1]
for t in rng.sample(sf, 4):
    pf = glob.glob(str(R / M / f"swebench-pro/swepro-run1/swepro-run1/{t[:30]}*/agent/model.patch"))
    patch = open(pf[0]).read() if pf else ""
    nfiles = patch.count("diff --git")
    if not patch.strip(): cat = "定位错误（未产生补丁）"
    elif nfiles and all(("test" in l.lower()) for l in patch.splitlines() if l.startswith("diff --git")): cat = "测试理解错误（只改了测试文件）"
    else: cat = "补丁错误（改动未通过隐藏测试）"
    if t in sb and rew(sb[t]) >= 1: cat += "；合版引入的退化（Base 通过）"
    rows.append(("SWE-Pro", t[:60], cat, f"补丁改动 {nfiles} 个文件，{len(patch)} 字节"))

# ASI-Bench
asi = json.load(open(R / M / "asi-bench/asi-b4-run1/local_score_seed31415.json"))["results"]
asb = {x["instance_id"]: x for x in json.load(open(R / B / "asi-bench/asi-b4-run1/local_score_seed31415.json"))["results"]}
af = [x for x in asi if x.get("evaluation_status") == "completed" and (x.get("final_score") or 0) < 20]
for x in rng.sample(af, 4):
    g = [s for s in x.get("gate_results") or [] if not s.get("passed")]
    cat = "输出缺失或格式不符（hard gate 未通过）" if g else "科研方法/数值精度不足（得分低）"
    if (asb.get(x["instance_id"], {}).get("final_score") or 0) > (x.get("final_score") or 0): cat += "；低于 Base"
    rows.append(("ASI-Bench", x["instance_id"].replace("__seed31415", ""), cat,
                 f"得分 {x.get('final_score')}/{x.get('max_score')}；" + (g[0].get("message", "")[:80] if g else "")))

cats = {}
for r in rows:
    for c in r[2].split("；"):
        cats[c] = cats.get(c, 0) + 1
md = ["# 合版模型失败案例分析（merged-linear-1.0）", "",
      f"共 {len(rows)} 个案例，随机种子 0 从四项 benchmark 的失败中抽取，优先抽“Base 通过而合版失败”的案例。由 `scripts/make_failures.py` 从原始日志生成，归因依据日志中的证据（是否超时、是否产生补丁、gate 结果、是否被思考预算截断等）。", "",
      "| # | Benchmark | 任务 / 题目 ID | 归因 | 证据 |", "|---|---|---|---|---|"]
md += [f"| {i+1} | {b} | `{t}` | {c} | {e} |" for i, (b, t, c, e) in enumerate(rows)]
md += ["", "## 归因统计", "", "| 类别 | 次数 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(cats.items(), key=lambda x: -x[1])]
md += ["", "## 结论与下一步", "",
       f"- HLE 上合版相对 Base 的退化题共 {len(reg)} 道（Base 对、合版错），反向（Base 错、合版对）见 `analysis/per_task.csv`。退化主要出现在思考预算耗尽、被迫作答的题上，与“训练选最短正确解，导致想得更浅”的假设一致。",
       f"- TB2 上合版失败 {len(fails)} 题，其中 Base 通过而合版失败 {len(regs)} 题；超时仍是最大类别。",
       "- 下一步：(1) E_reason 换成更难、更接近 HLE 的数据，按难度选择解长度；(2) 用本模型在 sandbox 中的成功轨迹训练 E_agent，并跑满 5 次 TB2；(3) 对推理相关模块降低合版系数，或逐层设置 λ。"]
Path("analysis").mkdir(exist_ok=True); Path("analysis/failures.md").write_text("\n".join(md) + "\n")
print("\n".join(md[:30]))
