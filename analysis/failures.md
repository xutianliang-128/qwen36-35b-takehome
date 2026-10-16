# 合版模型失败案例分析（merged-linear-1.0）

共 20 个案例，随机种子 0 从四项 benchmark 的失败中抽取，优先抽“Base 通过而合版失败”的案例。由 `scripts/make_failures.py` 从原始日志生成，归因依据日志中的证据（是否超时、是否产生补丁、gate 结果、是否被思考预算截断等）。

| # | Benchmark | 任务 / 题目 ID | 归因 | 证据 |
|---|---|---|---|---|
| 1 | HLE | `67241990a1bd6b06dafb7052` | 合版引入的退化；思考预算耗尽 | Biology/Medicine；判分抽取答案 `A`；思考被 8K 截断后强制作答；Base 本题答对 |
| 2 | HLE | `6725107c97743d26179494c6` | 合版引入的退化；思考预算耗尽 | Math；判分抽取答案 `2`；思考被 8K 截断后强制作答；Base 本题答对 |
| 3 | HLE | `66eaf0ba5e31c19691fe73b1` | 合版引入的退化 | Humanities/Social Science；判分抽取答案 `B`；正常结束；Base 本题答对 |
| 4 | HLE | `671901f58b9cbfc7b2fd9723` | 合版引入的退化；思考预算耗尽 | Math；判分抽取答案 `E`；思考被 8K 截断后强制作答；Base 本题答对 |
| 5 | HLE | `677e6a6841ef4b6937c21b7c` | 超时（思考预算耗尽后仓促作答） | Math；判分抽取答案 `None`；思考被 8K 截断后强制作答；Base 本题也答错 |
| 6 | HLE | `6722072ddcce66512a82d9ed` | 超时（思考预算耗尽后仓促作答） | Humanities/Social Science；判分抽取答案 `$25`；思考被 8K 截断后强制作答；Base 本题也答错 |
| 7 | TB2 | `fix-git` | 测试理解或实现错误（任务自认完成但验证失败）；合版引入的退化（Base 通过） | 12 轮，输出 2378 tokens，异常 无 |
| 8 | TB2 | `sqlite-with-gcov` | 测试理解或实现错误（任务自认完成但验证失败）；合版引入的退化（Base 通过） | 18 轮，输出 3747 tokens，异常 无 |
| 9 | TB2 | `tune-mjcf` | 超时；合版引入的退化（Base 通过） | 223 轮，输出 94959 tokens，异常 AgentTimeoutError |
| 10 | TB2 | `polyglot-c-py` | 超时 | 4 轮，输出 5062 tokens，异常 AgentTimeoutError |
| 11 | TB2 | `overfull-hbox` | 测试理解或实现错误（任务自认完成但验证失败） | 67 轮，输出 20655 tokens，异常 无 |
| 12 | TB2 | `qemu-alpine-ssh` | 超时 | 324 轮，输出 47146 tokens，异常 AgentTimeoutError |
| 13 | SWE-Pro | `instance_flipt-io` | 补丁错误（改动未通过隐藏测试） | 补丁改动 6 个文件，13320 字节 |
| 14 | SWE-Pro | `instance_NodeBB` | 补丁错误（改动未通过隐藏测试） | 补丁改动 4 个文件，5315 字节 |
| 15 | SWE-Pro | `instance_navidrome` | 补丁错误（改动未通过隐藏测试） | 补丁改动 3 个文件，4068 字节 |
| 16 | SWE-Pro | `instance_element-hq` | 补丁错误（改动未通过隐藏测试） | 补丁改动 4 个文件，7358 字节 |
| 17 | ASI-Bench | `biostat.perturbation_convergence_testing` | 科研方法/数值精度不足（得分低） | 得分 2.8000000000000003/100.0； |
| 18 | ASI-Bench | `biology.protein_folding_go_model` | 科研方法/数值精度不足（得分低）；低于 Base | 得分 14.22911022785771/100.0； |
| 19 | ASI-Bench | `physics.ks_2d_anisotropic` | 科研方法/数值精度不足（得分低）；低于 Base | 得分 19.814055123855034/100.0； |
| 20 | ASI-Bench | `computer_science.causal_attention` | 输出缺失或格式不符（hard gate 未通过） | 得分 0.0/100.0；Target file not found: analysis.py |

## 归因统计

| 类别 | 次数 |
|---|---|
| 合版引入的退化 | 4 |
| 补丁错误（改动未通过隐藏测试） | 4 |
| 思考预算耗尽 | 3 |
| 测试理解或实现错误（任务自认完成但验证失败） | 3 |
| 合版引入的退化（Base 通过） | 3 |
| 超时 | 3 |
| 科研方法/数值精度不足（得分低） | 3 |
| 超时（思考预算耗尽后仓促作答） | 2 |
| 低于 Base | 2 |
| 输出缺失或格式不符（hard gate 未通过） | 1 |

## 结论与下一步

- HLE 上合版相对 Base 的退化题共 174 道（Base 对、合版错），反向（Base 错、合版对）见 `analysis/per_task.csv`。退化主要出现在思考预算耗尽、被迫作答的题上，与“训练选最短正确解，导致想得更浅”的假设一致。
- TB2 上合版失败 58 题，其中 Base 通过而合版失败 9 题；超时仍是最大类别。
- 下一步：(1) E_reason 换成更难、更接近 HLE 的数据，按难度选择解长度；(2) 用本模型在 sandbox 中的成功轨迹训练 E_agent，并跑满 5 次 TB2；(3) 对推理相关模块降低合版系数，或逐层设置 λ。
