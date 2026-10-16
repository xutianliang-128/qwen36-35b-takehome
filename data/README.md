# 数据说明 / Data

所有数据构造代码在 `src/data/`，清单与 SHA-256 校验和在 `data/*/manifest.sha256`。大文件（原始采样、SFT jsonl）不入库，可用主 README 第 8 节的命令重新生成。

## E_reason（STEM 推理）
- 来源：TIGER-Lab/WebInstruct-verified @ `3e8a350b3a`（Apache-2.0）
- 选题（`build_reason_pool.py`，种子 20261015）：只取程序可校验的答案类型（Float/Integer/Fraction/Percentage/Multiple Choice/Boolean/Expression），题干长度 40–4000 字符，题干去重；大学 / 博士难度按 HLE 学科分布配额；先留出 15% 作为独立 dev 集（212 题，只用于合版选择）。
- 生成（`gen_rft.py`）：Base 每题 2 次采样（thinking，T=1.0，top_p 0.95，top_k 20，presence_penalty 1.5，上限 16K），600 题共 1,200 个回答。
- 校验（`answer_check.py` v2）：取最后一个 `\boxed{}`；数值 1% 容差、选择题字母、是非题、math-verify 表达式等价。原则是"宁可错杀"：漏判正确只会少一条数据，误判错误则会教坏模型。
- 结果：截断 260，答对 365（pass@1 = 0.304），至少答对一次的题 227。
- 防污染：对 cais/hle @ `5a81a4c7` 全部 2500 题（题干 + 答案）做 13-gram 匹配，命中 0。
- SFT：每题只保留**最短的正确解**，训练 207 / 验证 20（按题划分）。

## E_agent（终端 / agent）
- 来源：NiuNiu0110/TMax-Agent-SFT-terminus @ `7d74fd38a3`（ODC-BY），源自 allenai/tmax-sft 中经验证器判定成功的轨迹（Qwen3.6-27B 生成）
- 转换（`build_sft_agent.py`）：逐轮改写为 Terminus-2（json 解析器）格式；使用 Harbor 0.24.0 自带的 `terminus-json-plain.txt` 模板；完成信号映射为 `task_complete` 并补上确认轮；删除 `(exit_code=N)` 等上游框架痕迹；按命令类型设置等待时间。等待时间分布：{'0.1': 40852, '1.0': 10481, '2.0': 10619, '15.0': 1065, '30.0': 3589}。
- 结果：上游 5445 条，转换成功 5372，丢弃未完成 73，任务组 1062。
- 防污染：对 Terminal-Bench 2.0 全部 89 题的 instruction 做 13-gram 匹配 + 任务名精确匹配，命中 0；并主动排除了公开的、在 TB2 考题上录制的轨迹数据集。
- SFT：每条样本只训练一个目标轮，历史不含思考；训练 1500（按 16K 长度过滤后 1,411）/ 验证 60，按任务组划分。

## E_joint（对照）
两份训练集的并集：推理数据重复 2 次（对应其 2 个 epoch）+ agent 数据 1 次，按各自的长度上限过滤后共 1,825 条；验证集 78 条。

## 评测数据（只用于评测，不参与训练或选择）
HLE（cais/hle @ `5a81a4c7`）、Terminal-Bench 2.0（逐题 SHA-256 见 `configs/eval/terminal_bench_2.0_manifest.json`）、SWE-bench Pro V2 HARD-51（`configs/eval/swebench_pro_v2_hard51_ids.txt`）、ASI-Bench seed31415 @ `0fa14219`。
