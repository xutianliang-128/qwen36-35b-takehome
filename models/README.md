# 模型（Hugging Face，公开）/ Models (public on Hugging Face)

基座 / Base: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) @ `995ad96eacd98c81ed38be0c5b274b04031597b0`（tokenizer 与 chat template 同 revision；bf16，不量化）

| 模型 | 链接 | revision |
|---|---|---|
| qwen36-35b-a3b-e-reason-lora | https://huggingface.co/tianlix/qwen36-35b-a3b-e-reason-lora | `5cd762da73c846cce4711c8197df11f9da5cac03` |
| qwen36-35b-a3b-e-agent-lora | https://huggingface.co/tianlix/qwen36-35b-a3b-e-agent-lora | `3f7904b110577a6b7f41040e32895588af5bcc7f` |
| qwen36-35b-a3b-merged-linear-lora | https://huggingface.co/tianlix/qwen36-35b-a3b-merged-linear-lora | `18e91e499513a8a07b021efb079d31473f38f0ef` |
| qwen36-35b-a3b-e-joint-lora | https://huggingface.co/tianlix/qwen36-35b-a3b-e-joint-lora | `69e1848abc955058e8a140b39820062bedb78883` |

- **E_reason / E_agent / E_joint**：rank 16、alpha 32 的 LoRA adapter（目标模块见主 README 4.4）。
- **merged-linear-lora（最终合版模型）**：Task Arithmetic，λ=1，ΔW = (α/r)B·A。两个专家的 α/r 都是 2，所以两份 ΔW 之和**恰好**等于把 A 纵向拼接、B 横向拼接得到的 rank-32、alpha-64 adapter（数值验证 max|diff| = 4e-11）。合版配方见仓库中的 `merge_config.json`，冲突诊断见 `merge_diagnostics.json`。
- 许可：基座 Apache-2.0；训练数据 WebInstruct-verified（Apache-2.0）、TMax-Agent-SFT-terminus（ODC-BY）。

加载 / Load:
```python
from transformers import AutoModelForImageTextToText
from peft import PeftModel
base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "tianlix/qwen36-35b-a3b-merged-linear-lora",
        revision="18e91e499513a8a07b021efb079d31473f38f0ef").merge_and_unload()
```
vLLM：用 `peft` 合并后保存为完整权重再 `vllm serve`，或直接用 `src/merge/merge_experts.py` 从 adapter 生成完整权重。
