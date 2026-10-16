# Qwen3.6-35B-A3B 多专家训练与模型合版

[English version](README_en.md)

在 **Qwen/Qwen3.6-35B-A3B**（35B 总参数、约 3B 激活的 MoE）上训练两个专精不同能力的 LoRA 专家和一个联合训练对照，比较三种合版方法，并在 **HLE、Terminal-Bench 2.0、SWE-bench Pro V2 HARD-51、ASI-Bench** 四项 benchmark 上对比 Base、各专家与最终合版模型。

> **状态（2026-10-16 14:00 UTC）**：训练、合版、合版选择完成；**Base 与最终合版模型的四项评测全部完成**；E_agent 的 SWE-Pro 目标评测在服务器第三次重启后仍在重跑（完成后由 `scripts/summarize.py` 更新）。下文结果表均由 `scripts/summarize.py` 从原始日志生成。
> 本实验在有限算力（本地 3×RTX A6000 + 约 $200 云端额度）和固定时间内完成，**所有偏离题目设定或模型卡设置的地方都列在[第 6 节](#6-已知限制与偏离)**。

## 0. 提交信息

| 项目 | 链接 / 版本 |
|---|---|
| 代码仓库 | https://github.com/xutianliang-128/qwen36-35b-takehome |
| 基座模型 | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) @ `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| E_reason（STEM 推理专家，LoRA） | [tianlix/qwen36-35b-a3b-e-reason-lora](https://huggingface.co/tianlix/qwen36-35b-a3b-e-reason-lora) @ `5cd762da` |
| E_agent（终端 / agent 专家，LoRA） | [tianlix/qwen36-35b-a3b-e-agent-lora](https://huggingface.co/tianlix/qwen36-35b-a3b-e-agent-lora) @ `3f7904b1` |
| 最终合版模型 merged-linear-1.0 | [tianlix/qwen36-35b-a3b-merged-linear-lora](https://huggingface.co/tianlix/qwen36-35b-a3b-merged-linear-lora) @ `18e91e49`（两份 LoRA 直接相加，精确表示为 rank-32 LoRA，附合版配方） |
| 对照 E_joint（联合训练 LoRA） | [tianlix/qwen36-35b-a3b-e-joint-lora](https://huggingface.co/tianlix/qwen36-35b-a3b-e-joint-lora) @ `69e1848a` |

## 1. 研究问题

1. 对一个已经过大量后训练的 35B MoE 模型，**少量、按能力域划分的 LoRA 专家**能否在各自目标 benchmark 上带来提升？
2. 把这些专家**合并成单一模型**后，各项能力是叠加、保持还是互相干扰？在专家参数近乎正交时，简单相加和 TIES / DARE 哪种更好？
3. **"分别训练再合版"和"一份数据联合训练一个 adapter"** 相比，哪个更好？

## 2. 实验矩阵

| 模型 | 说明 | HLE | Terminal-Bench 2.0 | SWE-bench Pro HARD-51 | ASI-Bench (B4) |
|---|---|---|---|---|---|
| Base | 官方权重 | ✅ | ✅ | ✅ | ✅ |
| E_reason | STEM 推理专家 | ✅（目标） | — | — | — |
| E_agent | 终端 / agent 专家 | — | 未完成（时间） | 重跑中（目标） | — |
| **merged-linear-1.0** | E_reason + E_agent，任务向量直接相加 | ✅ | ✅ | ✅ | ✅ |
| merged-ties-d0.2 / merged-dare-d0.5 | 合版消融 | 仅 dev 集 | — | — | — |
| E_joint | 对照：两份数据联合训练 | 仅 dev 集 | — | — | — |

E_reason 的目标是 HLE；E_agent 的目标是 Terminal-Bench 和 SWE-bench Pro；ASI-Bench 是综合科研任务，不单设专家，用来考察合版能否组合多种能力。

## 3. 结果

### 3.1 四项 benchmark（主指标，百分制）

| Benchmark（主指标） | Base | E_reason | E_agent | merged-linear-1.0 | 合版 − Base（配对 bootstrap 95% CI） |
|---|---|---|---|---|---|
| HLE 2500 题，含图像（准确率，思考预算 8K + 强制作答） | 14.32 | 14.48 | — | **12.60** | **−1.72 [−3.08, −0.36]**（显著退化） |
| Terminal-Bench 2.0，89 题（成功率，1 次运行） | 41.57 | — | — | 34.83 | −6.74 [−14.61, 0.00] |
| SWE-bench Pro V2 HARD-51，51 题（resolved rate，patch replay 重判） | 31.37 | — | 重跑中 | 31.37 | 0.00 [−13.73, +13.73] |
| ASI-Bench seed31415，B4 档（本地非官方分，%；58/60 题可判分） | 7.43 | — | — | **11.50** | **+4.07 [+0.31, +8.79]**（显著提升） |
| **四项宏平均** | **23.67** | — | — | **22.57** | **−1.10 [−5.47, +2.84]**（未达到 +3 的目标） |

- E_reason 相对 Base 在 HLE 上 **+0.16 个百分点，95% CI [−1.28, +1.56]**，不显著。
- 完整数字、逐题结果和置信区间见 `results/summary.csv`、`analysis/per_task.csv`、`analysis/uncertainty.json`，可用 `python scripts/summarize.py` 从原始结果重新生成。

**HLE 补充指标**

| 模型 | 直接截断（8K）准确率 | 被截断比例 | 强制作答后准确率 | 纯文本 | 含图像 | 校准误差 |
|---|---|---|---|---|---|---|
| Base | 2.68 | 83.0% | 14.32 | 14.50 | 13.16 | 82.5 |
| E_reason | 3.12 | 78.0% | 14.48 | 14.74 | 12.87 | 82.1 |
| merged-linear-1.0 | 3.56 | 72.0% | 12.60 | 12.60 | 12.57 | 84.3 |

### 3.2 开发集（只用于选择合版配置，不使用任何 benchmark 数据）

| 模型 | E_reason dev 准确率（212 题，k=1，8K） | E_agent 留出集 loss | E_reason 留出集 loss |
|---|---|---|---|
| Base | 18.40 | 0.2863 | 0.2371 |
| E_reason | 21.70 | 0.2951 | 0.2298 |
| E_agent | 19.81 | **0.1879** | 0.2434 |
| E_joint（对照） | 19.81 | 0.1983 | **0.2278** |
| **merged-linear-1.0（选中）** | **23.58** | 0.1898 | 0.2346 |
| merged-ties-d0.2 | 23.11 | 0.1938 | 0.2357 |
| merged-dare-d0.5 | 22.64 | 0.1903 | 0.2350 |

![合版对比](curves/merge_comparison.png)

### 3.3 关键发现

1. **合版保住了两个专家的能力**（dev 集）：合版模型的 agent 留出 loss（0.190）接近 E_agent（0.188），推理 dev 准确率（23.6%）反而高于 E_reason（21.7%）和 Base（18.4%）。
2. **"分别训练再合版"优于"联合训练"**：在完全相同的数据曝光和超参下，E_joint 的推理 dev 准确率（19.8%）比合版低 3.8 个点，agent loss 也更高（0.198 vs 0.190）。
3. **两个专家的任务向量几乎正交**：逐模块余弦相似度平均约 −0.006，符号冲突率约 50%（相当于随机）。这种情况下 TIES 的裁剪和 DARE 的随机丢弃只会丢掉有用的改动，所以**简单相加最好**。
4. **合版在 ASI-Bench 上显著提升（+4.07，CI [+0.31, +8.79]），在 HLE 上显著退化（−1.72，CI [−3.08, −0.36]）**。ASI 的科研任务需要写代码、执行、处理数据，正是 E_agent 的能力，加上理科知识，体现了能力组合；而 dev 上的推理提升没有迁移到 HLE：我们的理科训练题（WebInstruct，大学难度）和 HLE（专家前沿难度）分布差距很大；合版模型想得更短（截断比例 72% vs 83%），但强制作答后准确率更低。这说明**"想得更简洁"在难题上可能损害了推理深度**，是本实验最重要的负面结果。
5. **TB2 和 SWE-Pro**：合版在 TB2 上低 6.7 个点（CI 上界为 0，边界显著），在 SWE-Pro 上持平。两者超时数相近（Base 31、合版 30），TB2 上合版丢了 9 题、多对 3 题，丢的 9 题中 6 题是"自认完成但验证失败"，3 题超时（见 `analysis/failures.md`）。**四项宏平均 −1.10 [−5.47, +2.84]，未达到 +3 的目标**：收益（ASI）和损失（HLE、TB2）相互抵消。另外，TB2 上 Base 有 31/89 题超时（3 小时上限）。本地 A6000 推理较慢，而 Terminus-2 中有 63% 的墙钟时间花在等待命令上（试跑测得），所以超时率偏高。这对所有模型的条件是一致的。

## 4. 方法

### 4.1 基座与固定版本

| 项 | 值 |
|---|---|
| 基座 | Qwen/Qwen3.6-35B-A3B @ `995ad96e`（`Qwen3_5MoeForConditionalGeneration`，35.95B 参数；40 层，其中 30 层线性注意力、10 层全注意力；每层 256 路由专家取 8 个，外加 1 个共享专家） |
| tokenizer / chat template | 与基座同一 revision（`tokenizer_config.json`、`chat_template.jinja`） |
| 精度 | bf16，所有训练和评测都不量化 |
| 推理 | vLLM 0.30.0，TP=2，`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`，max-model-len 262144，thinking 开启，`VLLM_USE_FLASHINFER_SAMPLER=0`（本地和 Modal 一致） |
| 训练依赖 | torch 2.13.0+cu130、transformers 5.19.0、peft 0.21.2、flash-linear-attention 0.5.2 |
| 评测依赖 | harbor 0.24.0、modal 1.6.1、mini-swe-agent 2.4.6、asibench 0.2.0、litellm 1.105.1 |

**没有降到 7B 的原因**：实测 bf16 LoRA 在 3×A6000 上能训 35B（最长 32K 序列，约 1,350 tokens/s，每卡峰值显存 30GB），所以保留 35B 基座。

### 4.2 专家设计

| 专家 | 能力域 | 目标 benchmark | 划分依据 |
|---|---|---|---|
| E_reason | STEM 推理（数学、物理、化学、生物、CS） | HLE | 知识和推理型问答，不需要工具 |
| E_agent | 终端 / 仓库内多步操作 | Terminal-Bench、SWE-bench Pro | 两者都是"读环境、发命令、看输出"的 agent 循环；时间有限，原计划的 SWE 专家和终端专家合并为一个 |
| （无） | — | ASI-Bench | 科研任务同时需要写代码执行和科学知识，用来考察合版能否组合能力；它只有 60 题，且是出题方自己的 benchmark，难以找到不污染的同分布训练数据 |

### 4.3 数据构造

| | E_reason | E_agent |
|---|---|---|
| 来源 | [TIGER-Lab/WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified) @ `3e8a350b3a` | [NiuNiu0110/TMax-Agent-SFT-terminus](https://huggingface.co/datasets/NiuNiu0110/TMax-Agent-SFT-terminus) @ `7d74fd38a3`（源自 allenai/tmax-sft 中经验证器判定成功的轨迹，由 Qwen3.6-27B 生成） |
| 许可 | Apache-2.0 | ODC-BY |
| 构造方式 | **自蒸馏拒绝采样**：Base 对每题采样 2 次（thinking，temperature 1.0），程序校验 `\boxed{}` 答案（数值 1% 容差、选择题字母、是非题、math-verify 表达式等价），**每题只保留最短的正确解** | 把 bash 工具调用格式**逐轮重写为 Terminus-2 JSON 格式**（官方提示模板、`analysis/plan/commands/task_complete`、两步完成确认），按命令类型设置合理的等待时间（0.1–30 秒） |
| 难度分层 | 只取大学和博士难度，按 HLE 学科分布配额（数学 495、物理 246、化学 147、生物 100、CS 84 等） | 1,062 个任务组，每组抽轨迹 |
| 清洗 | 只保留答案能被程序校验的题型；丢弃截断（16K）和没有 `\boxed{}` 的回答 | 删掉格式错误重试轮、`(exit_code=N)` 等上游框架痕迹；丢弃 73 条没有正常完成的轨迹；长度超过 16K 的样本丢弃 |
| 去重 / 防污染 | 训练题和 HLE 全部 2500 题（题干 + 答案）做 13-gram 去重，重叠 **0**；题干去重 | 和 TB2 全部 89 题的指令做 13-gram 去重 + 任务名精确匹配，重叠 **0**；**主动排除**网上所有在 TB2 考题上录制的轨迹（如 `*terminal-bench-2*-trajectories`） |
| 规模 | 600 题 × 2 次采样 → 227 题至少答对一次 → 训练 207 / 验证 20 条，1.6M tokens | 5,372 条转换成功 → 训练 1,500 条（按长度过滤后 1,411）/ 验证 60 条（按任务组划分），7.8M tokens |
| 独立开发集 | 另留 212 题，只用于合版选择 | 60 条按任务组留出的验证集 |
| 规模差异的理由 | 推理样本完全是自生成的，一次采样成本高（本地约 1,100 题/小时）；数据少，所以训 2 epoch | agent 轨迹现成可用，但单条很长（平均 5.5K tokens）；只训 1 epoch，避免过拟合 |

**训练样本格式**：只对模型自己的回答计算 loss。agent 样本**每条只训练一个目标轮**，历史轮次不带思考。这和评测时 Terminus-2 重发历史、不保留旧思考（`interleaved_thinking=false`）的情况一致。

数据清单和校验和见 `data/*/manifest.sha256`；数据构造代码在 `src/data/`。

### 4.4 训练

所有专家都用 **bf16 LoRA**（`src/train/train_lora.py`），3×A6000 逐层模型并行，gradient checkpointing，loss 分块计算（不一次性生成 248K 词表的完整 logits），MoE 专家层使用 `grouped_mm` 实现（比 eager 快 2.5 倍）。

| | E_reason | E_agent | E_joint（对照） |
|---|---|---|---|
| LoRA 目标模块 | 全注意力 q/k/v/o（10 层）、线性注意力 in_proj_qkv / in_proj_z / out_proj（30 层）、共享专家 gate/up/down（40 层） | 同左 | 同左 |
| 冻结 | router、256 个路由专家（3D 参数）、视觉塔、embedding、lm_head | 同左 | 同左 |
| rank / alpha / dropout | 16 / 32 / 0 | 16 / 32 / 0 | 16 / 32 / 0 |
| 可训练参数 | 19.17M | 19.17M | 19.17M |
| 学习率 / 调度 | 5e-5，cosine，5% warmup | 同左 | 同左 |
| batch | 16 条（单条 micro-batch，梯度累积 16） | 同左 | 同左 |
| epoch / 步数 | 2 / 26 | 1 / 89 | 1 / 115 |
| 最大长度 | 20,480 | 16,384 | 20,480（agent 部分按 16,384 过滤） |
| 随机种子 | 42 | 42 | 42 |
| 训练时长 | 40.3 分钟 | 96.0 分钟 | 138.0 分钟 |
| 吞吐 | 1,321 tokens/s | 1,358 tokens/s | 1,331 tokens/s |
| 每卡峰值显存 | 30.0 GiB | 30.0 GiB | 30.0 GiB |
| checkpoint 大小（adapter） | 76.8 MB | 76.8 MB | 76.8 MB |
| 验证 loss（起始 → 最终） | 0.2370 → 0.2296 | 0.2863 → 0.1879 | 0.2419 → 0.2248 |
| checkpoint 选择 | 训练结束时的 final（不按 benchmark 挑选） | 同左 | 同左 |

**为什么用 LoRA 而不用 QLoRA 或全参训练**：
- 全参训练的优化器状态需要 400GB 以上。
- 这个模型的路由专家是 3D 融合参数，bitsandbytes 只量化 `nn.Linear`，所以 QLoRA 省不了多少显存；而且量化基座上训出的 adapter 和 bf16 合版或推理时的基座不一致。
- **冻结 router 和路由专家**是为合版考虑：如果每个专家各自改了路由，合版后会产生路由冲突。

Loss 曲线（训练 loss 原始值 + EMA 0.8 平滑，验证 loss，标注了 checkpoint）：

| E_reason | E_agent | E_joint |
|---|---|---|
| ![](curves/e_reason_loss.png) | ![](curves/e_agent_loss.png) | ![](curves/e_joint_loss.png) |

### 4.5 对照：多专家合版 vs 单模型联合训练

E_joint 用**两份训练集的并集**训练**一个** LoRA，每条样本的曝光次数和两个专家相同（推理数据重复 2 次 = 它的 2 个 epoch，agent 数据 1 次），超参完全相同。结果见 3.2 和 3.3 第 2 条。

### 4.6 合版

工具：`src/merge/merge_experts.py`。所有方法都作用在**稠密任务向量 ΔW = (alpha/r)·B·A** 上，不分别平均 A、B 矩阵（数学上不正确）。只修改 LoRA 覆盖的 250 个模块，其余参数逐字节复制基座。

| 方法 | 原理 | 适用前提 | 配置 |
|---|---|---|---|
| Task Arithmetic（linear） | W = W₀ + Σλᵢ·ΔWᵢ | 任务向量冲突小时最保真 | λ = 1.0 |
| TIES | 每个 ΔW 只保留幅值前 density 的元素 → 按符号质量为每个位置选定方向 → 只平均同号的值 | 任务向量冲突大时能减少相互抵消 | density 0.2，λ = 1.0 |
| DARE-TIES | 随机丢弃 (1−density) 的元素并按 1/density 放大，再做 TIES | 任务向量冗余大时 | density 0.5，seed 0（CPU 随机数，本地与 Modal 结果一致） |

**参数冲突分析**（合版时自动输出，`merge_diagnostics.json`）：两个专家逐模块的余弦相似度平均为 −0.0055（按模块类型在 −0.019 到 0.002 之间），在两者都非零的位置上符号冲突率为 50.2%。也就是说，两个任务向量**近乎正交**，相加时几乎不互相抵消。

**选择规则**（在看到任何合版模型的 benchmark 结果之前就固定，见 `src/merge/select_merge.py`）：在 dev 集上，按 E_reason dev 准确率（越高越好）和 E_agent 留出 loss（越低越好）分别排名，取平均排名最小者，平局时看前者。结果选中 **merged-linear-1.0**，冻结后才进行 benchmark 评测。

最终合版模型的 TB2、SWE-Pro 和 HLE 都只评一次，**没有根据 benchmark 结果回头调整任何配置**。

## 5. 评测协议

每个 benchmark 内，所有被评模型使用**相同的任务集、相同的 agent 框架和 prompt、相同的推理服务、相同的采样参数与预算**：thinking 开启，preserve_thinking / interleaved_thinking 关闭，bf16 不量化。环境失败单独计数，**不从分母中剔除**。

| | HLE | Terminal-Bench 2.0 | SWE-bench Pro V2 HARD-51 | ASI-Bench |
|---|---|---|---|---|
| 来源 / 版本 | [cais/hle](https://huggingface.co/datasets/cais/hle) @ `5a81a4c7`，test split，2500 题（含 342 道图像题） | Harbor 注册表 `terminal-bench@2.0`，89 题；本地快照的逐题 SHA-256 见 `configs/eval/terminal_bench_2.0_manifest.json` | [scaleapi/SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os) @ `66f92766`（v2.0.0）的 `v2/hard51_ids.txt`；HF `ScaleAI/SWE-bench_Pro` @ `2d52cb3d` 的 `hard` 配置 | [apexin-ai/ASI-Bench](https://github.com/apexin-ai/ASI-Bench) @ `cf07cb89`（v0.2.0）；数据 `Apexintelligence-AI/ASI-Bench-seed31415` @ `0fa14219` |
| hard 划分依据 | — | — | 官方定义：5 个模型家族中至少 2 个在锁定协议下失败的题，去掉 3 道有歧义的 | — |
| 框架 / agent | 直接调用，无工具；官方系统提示词 | Harbor 0.24.0 + Terminus-2（json 解析器） | 官方 host 端 mini-swe-agent（`v2/tooling/host_mini_swe.py`，`mini_toolcall.yaml` 提示词不改），sandbox 不联网，**官方 patch_replay 在干净镜像上重新判分** | 官方 opencode_cli 适配器（opencode-ai@1.17.15），sandbox=task |
| 采样 | T=1.0，top_p 0.95，top_k 20，presence_penalty 1.5 | T=1.0，top_p 0.95，top_k 20，max_tokens 80K，256K context | T=1.0，top_p 0.95，top_k 20 | opencode 默认 |
| 预算 / 超时 | 思考 8K；超限时用 Qwen 官方 thinking-budget 方法强制作答，最多再写 1K tokens | 3 小时；32 CPU / 48GB 作为资源上限 | 每题 50 分钟（官方） | 每题 3 小时（官方默认） |
| 判分 | 官方判分提示词，o3-mini-2025-01-31，结构化输出 | 官方 verifier | 官方 verifier（patch replay） | `asibench score --repo seed31415`，公开参考答案，**非官方分** |
| 主指标 | 准确率（强制作答后） | 任务成功率 | resolved rate | 总分 / 满分 |
| 推理硬件 | Modal 2×H100（所有模型一致） | 本地 2×A6000，32 个 trial 并发（所有模型一致） | Modal 2×H100，26 并发 | Modal 2×H100，60 并发 |
| 运行次数 | 1 | 1（种子由 vLLM 随机） | 1 | 1 |

**和模型卡设置的差异**（均为本实验设置，**不声称复现官方成绩**）：
- HLE 模型卡未公开思考预算；本实验用 8K + 强制作答。
- TB2 使用题目指定的 32 CPU / 48GB / 3 小时，但官方任务原本只需 1–8 CPU、2–16GB、15–120 分钟；只跑了 1 次，不是 5 次。
- SWE-Pro：模型卡用的是 Qwen 内部框架和内部修订过的任务集，本实验用官方 v2 HARD-51 和官方 mini-swe-agent。
- ASI-Bench：只跑 B4 档；seed31415 本地打分非官方。

**ASI-Bench 的防作弊处理**：公开的 seed31415 在每题旁边附带参考答案，而 `task` sandbox 没有文件系统隔离。烟雾测试中我们发现：(1) agent 能读到参考答案；(2) 由于 opencode 从 `$PWD` 推断项目目录，agent 在 ASI 仓库根目录下读到了 `custom_scorer.py` 和 `task_eval.yaml`。正式运行前已修复：
- agent 只拿到**删除了所有 `reference/` 的数据副本**；
- 容器里的任务目录只保留 `task_meta.yaml`、`task.yaml` 和 `prompt_b*.md`，删除判分代码、生成脚本、示例任务和 git 历史；
- 用一个包装脚本把 opencode 的 `$PWD` 固定为任务工作区。

跑完后对所有 agent 日志做关键词审计（`hidden_truth`、`/reference/`、`custom_scorer`、`task_eval.yaml`、参考数据集 URL），**命中 0 次**（`results/*/asi-bench/*/audit_hits.txt`）。

## 6. 已知限制与偏离

本实验的算力只有实验室的 3 张 A6000（服务器没有 Docker，也没有 rootless podman 权限），外加个人 Modal 账号（中途两次触到用量上限，最终额度 $200）。在此约束下的取舍如下：

1. **专家数量**：原计划 3 个（SWE、终端、推理），时间不够，合并为 2 个。
2. **Terminal-Bench 只跑 1 次**（题目要求 5 次取均值和标准差），所以无法报告重复运行的波动，只能报告配对 bootstrap CI。E_agent 的 TB2 评测在截止前未完成。
3. **HLE 思考预算 8K**：16K 时本地一个模型要 6 小时。8K 下 Base 有 83% 的回答被截断，所以加了 Qwen 官方的 thinking-budget 强制作答。这条规则是**看到 Base 截断率之后**加的，但它在看到任何专家或合版结果之前就已固定，并且对所有模型一样。
4. **ASI-Bench 只跑 B4 档**（60/240 个实例），seed31415 本地非官方打分。有 2 个任务（TSP、Max-3-SAT）的判分需要 Docker，两个模型都无法判分，按官方规则不计入分母；比较时只用两边都成功判分的实例。
5. **SWE-bench Pro 和 ASI 的推理在 Modal H100 上**，TB2 的推理在本地 A6000 上。每个 benchmark 内所有模型的硬件一致，但不同 benchmark 之间的绝对速度不同。
6. **E_reason 的训练数据少**（207 条），分布也比 HLE 简单得多；**E_agent 的数据来自 Qwen3.6-27B 的轨迹**，不是本模型自己的成功轨迹（需要大量 sandbox，预算不允许）。
7. **服务器意外重启了三次**（10-15 03:58、14:15 UTC，10-16 12:50 UTC 内核升级），损失约 4 小时，E_agent 的 SWE-Pro 因此在截止时尚未完成；之后用 `@reboot` cron 加上可断点续跑的驱动脚本应对。
8. 早期在 A6000 上的 HLE 部分运行（16K / 8K 试跑）只作为试跑存档（`results/qwen36-base/hle/pilot-*`），不计入结果。

## 7. 成本与时间

| 项目 | 数值 |
|---|---|
| 本地 GPU（RTX A6000，免费） | 约 50 GPU 小时（训练约 14.3 GPU 小时；数据生成约 6；TB2 约 24；dev 与试跑约 6） |
| Modal（H100 推理 + CPU sandbox） | 截至 05:00 UTC 为 $154（vLLM / H100 $89.6，TB2 / SWE sandbox $48.5，ASI 容器 $16.0）；最终数字见 `logs/resources.csv` |
| OpenAI（HLE 判分 o3-mini + ASI 任务判分 gpt-5.5） | 约 $40（估算） |
| 主动工作时长 | 约 10 小时（自报；其余时间为无人值守的后台运行），见 `logs/worklog.md` |

GPU 小时 = 卡数 × 墙钟时间；Modal 费用取自 `modal.billing` 的账单导出。

## 8. 复现

```bash
# 0) 环境（三个隔离的 conda 环境；所有命令都加 PYTHONNOUSERSITE=1）
conda create -n qwen36-train python=3.12 && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 \
  && pip install transformers==5.19.0 peft==0.21.2 accelerate flash-linear-attention pyyaml
conda create -n qwen36-vllm python=3.12 && pip install vllm==0.30.0
conda create -n qwen36 python=3.12 && pip install harbor==0.24.0 modal==1.6.1 mini-swe-agent==2.4.6 dockerfile-parse math-verify httpx pandas matplotlib
# .env: VLLM_API_KEY, OPENAI_API_KEY（判分）, HF_TOKEN（cais/hle 为 gated）

# 1) 数据
python src/data/build_reason_pool.py --out data/reason                       # 选题 + 留出 dev
bash scripts/serve_local.sh                                                    # 本地 vLLM（基座）
python src/data/gen_rft.py --pool data/reason/pool_train.jsonl --out data/reason --max-questions 600
python src/data/build_sft_reason.py --samples data/reason/samples.jsonl --out data/reason   # HLE 去污染 + 最短正确解
python src/data/build_sft_agent.py --out data/agent --n-samples 1500           # TMax -> Terminus-2，TB2 去污染

# 2) 训练、合版、选择、评测（可断点续跑的驱动脚本）
bash scripts/run_all.sh          # E_agent、E_joint 训练 -> 合版 -> dev 选择 -> TB2 / SWE-Pro
bash scripts/hle_modal_queue.sh  # HLE（Modal）    bash scripts/hle_force_queue.sh  # 思考预算强制作答 + 判分
bash scripts/asi_queue.sh        # ASI-Bench B4（Modal）   bash scripts/asi_score.sh qwen36-base merged-linear-1.0
bash scripts/swepro_queue.sh     # SWE-Pro HARD-51（Modal 推理）

# 3) 汇总与画图（从原始结果重新生成全部表格和曲线）
python scripts/summarize.py && python scripts/plot_curves.py
```

单个专家训练：`python src/train/train_lora.py --config configs/train/e_reason.yaml`（其余见 `configs/train/`）。
合版：`python src/merge/merge_experts.py --method linear --lam 1.0 --expert e_reason=checkpoints/e_reason/final --expert e_agent=checkpoints/e_agent/final --out <dir>`。

## 9. 目录结构与日志

```
configs/train/        各专家与对照的训练配置（训练前固定）
configs/eval/         四项 benchmark 的固定配置、任务清单（TB2 SHA-256 清单、HARD-51 ID）
data/                 数据说明与清单（data/README.md、*/manifest.sha256）；大文件不入库
src/data/             选题、拒绝采样、答案校验、SFT 构造
src/train/            LoRA 训练、测速
src/merge/            合版（linear / TIES / DARE-TIES + 冲突诊断）与 dev 选择
src/eval/             HLE 生成 / 强制作答 / 判分，留出 loss
modal_app/            Modal 上的 vLLM 隧道、合版、ASI 运行容器
scripts/              驱动脚本、评测入口、summarize.py、plot_curves.py
checkpoints/<expert>/ metrics.jsonl（逐步训练指标）、train_metadata.json；权重在 HF
curves/               loss 曲线、合版对比图，以及 source/*.csv 源数据
results/<model>/<benchmark>/<run>/   逐题输出、判分、错误；results/dev/ 为开发集
analysis/             per_task.csv、uncertainty.json、failures.md（失败案例分析）
logs/                 训练、评测、合版、服务、错误日志；worklog.md 为工时与费用
docs/engineering_notes.md   工程细节与踩坑记录
```

日志都是纯文本或 JSONL：训练逐步指标在 `checkpoints/*/metrics.jsonl`，评测运行日志在 `logs/`，Harbor 的每个 trial 都有 `result.json` / `trial.log`。

## 10. 下一步实验

1. **E_reason 用更难、更贴近 HLE 的数据，并保留更长的思考**：当前"最短正确解"的选择可能损害了难题上的推理深度，可以改为按难度分层选长度。
2. **用本模型自己在 sandbox 中的成功轨迹训练 E_agent**（拒绝采样 / RFT），并在 TB2 上跑满 5 次以估计波动。
3. **按模块或按层调整合版系数**：两个任务向量近乎正交，可以试 λ > 1，或只对推理相关的层降权，看能否消除 HLE 上的退化。
4. **ASI-Bench 跑全 B1–B4 档**，并用 seed42 在 Docker 环境中提交官方打分。
5. 拆分 SWE 专家与终端专家（3 专家），检验专家粒度对合版的影响。
