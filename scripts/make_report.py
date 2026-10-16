"""One-page report.pdf (摘要-过程-结果) rendered with matplotlib; numbers read from results/summary.csv."""
import csv, json
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]  # Latin + CJK fallback
S = {(r["model"], r["benchmark"]): r for r in csv.DictReader(open("results/summary.csv"))}
U = json.load(open("analysis/uncertainty.json"))["macro"]
g = lambda m, b: S[(m, b)]["score_pct"] if (m, b) in S else "—"
abstract = ("摘要：以 Qwen3.6-35B-A3B 为基座，用 LoRA 训练推理与终端两个专家，在开发集上比较三种合版方法与联合训练对照，"
            "选出任务向量直接相加的合版模型。两专家任务向量近乎正交，合版在开发集上保住了双方能力且优于联合训练；"
            f"四项 benchmark 上合版相对 Base 宏平均 {U['merged-linear-1.0']['macro_diff_vs_base_pp']:+.2f} 个百分点，"
            "ASI-Bench 显著提升、HLE 显著下降，未达到 +3 的目标。")
process = [
 "过程：基座 revision 995ad96e，bf16，不量化。E_reason：WebInstruct-verified 上 Base 自采样、程序校验、保留最短正确解（207 条，HLE 13-gram 去污染 0 命中）。",
 "E_agent：TMax 成功轨迹逐轮改写为 Terminus-2 格式（1,411 条，TB2 去污染 0 命中）。LoRA r16，冻结 router 与路由专家，3×A6000 训练 40/96 分钟，峰值显存 30GB。",
 "对照 E_joint 用同样曝光联合训练。合版作用于 ΔW=(α/r)BA，按事先固定的 dev 规则选择；所有 benchmark 每个模型只评一次。",
 "评测：HLE 2500 题（含图，8K 思考 + 强制作答，o3-mini 判分）；TB2 89 题（Harbor+Terminus-2，32CPU/48GB/3h，1 次）；",
 "SWE-Pro V2 HARD-51（官方 mini-swe-agent + patch replay）；ASI-Bench seed31415 B4（官方 opencode，参考答案对 agent 隔离，审计 0 命中）。",
]
cols = ["Benchmark", "Base", "E_reason", "合版", "合版−Base [95% CI]"]
names = [("hle", "HLE"), ("terminal_bench", "Terminal-Bench 2.0"), ("swebench_pro_hard51", "SWE-Pro HARD-51"), ("asi_bench", "ASI-Bench B4")]
rows = []
for b, n in names:
    r = S.get(("merged-linear-1.0", b), {})
    ci = f"{float(r['diff_vs_base_pp']):+.2f} [{r['ci95_low']}, {r['ci95_high']}]" if r.get("diff_vs_base_pp") else "—"
    rows.append([n, g("qwen36-base", b), g("expert-e_reason", b), g("merged-linear-1.0", b), ci])
mb, mm = U["qwen36-base"]["macro_avg_pct"], U["merged-linear-1.0"]
rows.append(["四项宏平均", f"{mb}", "—", f"{mm['macro_avg_pct']}", f"{mm['macro_diff_vs_base_pp']:+.2f} {mm['macro_diff_ci95']}"])
fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.06, 0.965, "Qwen3.6-35B-A3B 多专家训练与模型合版 — 一页报告", fontsize=15, weight="bold")
import textwrap
y = 0.93
for para, size in [(abstract, 9.5)] + [(p, 8.6) for p in process]:
    for line in textwrap.wrap(para, 58):
        fig.text(0.06, y, line, fontsize=size); y -= 0.0185
    y -= 0.006
fig.text(0.06, y - 0.005, "结果（主指标，%；配对 bootstrap 95% CI）", fontsize=11, weight="bold"); y -= 0.02
ax = fig.add_axes([0.06, y - 0.17, 0.88, 0.16]); ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.35)
y -= 0.20
extra = [
 f"运行失败率：TB2 Base 2/89、合版 2/89（环境错误，计为失败）；超时 Base 31、合版 30。HLE、ASI 无运行失败；ASI 有 2 题需要 Docker 判分，两模型同等排除。",
 "开发集：推理 dev 准确率 Base 18.4 / E_reason 21.7 / 合版 23.6 / 联合训练 19.8；agent 留出 loss Base 0.286 / E_agent 0.188 / 合版 0.190 / 联合 0.198。",
 "成本：本地 RTX A6000 约 50 GPU 小时（训练 13.7）；Modal 约 $190（H100 推理 + CPU sandbox）；OpenAI 判分约 $40；主动工时约 10 小时。",
 "主要偏离：TB2 只跑 1 次（要求 5 次）；HLE 8K 思考预算；ASI 只跑 B4 档、非官方打分；E_agent 的 SWE-Pro 因推理硬件不一致（本地 A6000 vs H100）全部超时而作废，其 TB2 已完成（见仓库）。",
 "下一步：更难的推理数据并按难度选择解长度；用本模型自己的 sandbox 成功轨迹训练 E_agent；按模块 / 逐层设置合版系数以消除 HLE 退化。",
]
for para in extra:
    for line in textwrap.wrap(para, 62):
        fig.text(0.06, y, line, fontsize=8.6); y -= 0.0175
    y -= 0.005
fig.text(0.06, 0.02, "详细日志、逐题结果、20 个失败案例与置信区间见 GitHub 仓库：github.com/xutianliang-128/qwen36-35b-takehome", fontsize=7.5, color="grey")
fig.savefig("report.pdf"); fig.savefig("/tmp/claude-114281459/-home-tianlix/92487e70-038f-4b47-8a95-9606a148d5a6/scratchpad/report.png", dpi=70); import re; print("abstract chars:", len(abstract.split("：", 1)[1]), "cjk:", len(re.findall(r"[\u4e00-\u9fff]", abstract.split("：", 1)[1])), "y_end", round(y, 3))
