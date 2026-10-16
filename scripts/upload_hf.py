"""Upload adapters (+ model cards, merge recipe) to public HF repos under the token's account; prints URLs + revisions."""
import json, os, shutil, tempfile
from pathlib import Path
from huggingface_hub import HfApi

BASE = "Qwen/Qwen3.6-35B-A3B"; REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
GH = "https://github.com/xutianliang-128/qwen36-35b-takehome"
REPOS = {
    "qwen36-35b-a3b-e-reason-lora": ("checkpoints/e_reason/final", "E_reason: STEM-reasoning expert (self-distilled rejection-sampling SFT on WebInstruct-verified, HLE-decontaminated)"),
    "qwen36-35b-a3b-e-agent-lora": ("checkpoints/e_agent/final", "E_agent: terminal/agent expert (TMax successful trajectories re-rendered as Terminus-2 dialogues, TB2-decontaminated)"),
    "qwen36-35b-a3b-merged-linear-lora": ("checkpoints/merged-linear-1.0-lora", "Final merged model: Task Arithmetic (lambda=1) of E_reason + E_agent, stored EXACTLY as one rank-32 LoRA (A stacked, B concatenated)"),
    "qwen36-35b-a3b-e-joint-lora": ("checkpoints/e_joint/final", "Control: one LoRA trained jointly on the union of both experts' data"),
}
CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
tags: [lora, qwen3.6, moe, model-merging]
---
# {name}

{desc}.

- Base: [{base}](https://huggingface.co/{base}) @ `{rev}` (bf16, no quantization)
- LoRA: rank {r}, alpha {alpha}, targets = full-attention q/k/v/o, linear-attention in_proj_qkv/in_proj_z/out_proj, shared-expert gate/up/down; router and 256 routed experts frozen
- Training data, configs, logs, evaluation and limitations: {gh}
- Licenses: base Apache-2.0; training data WebInstruct-verified (Apache-2.0) / TMax-Agent-SFT-terminus (ODC-BY)

## Load
```python
from transformers import AutoModelForImageTextToText
from peft import PeftModel
m = AutoModelForImageTextToText.from_pretrained("{base}", revision="{rev}", dtype="bfloat16", device_map="auto")
m = PeftModel.from_pretrained(m, "{repo}")
m = m.merge_and_unload()   # optional: dense weights
```
{extra}
"""
api = HfApi(token=os.environ["HF_WRITE_TOKEN"]); user = api.whoami()["name"]; out = {}
for name, (src, desc) in REPOS.items():
    repo = f"{user}/{name}"
    api.create_repo(repo, private=False, exist_ok=True)
    cfg = json.load(open(Path(src) / "adapter_config.json"))
    extra = ""
    if "merged" in name:
        extra = ("\n## Merge recipe\n`merge_config.json` / `merge_diagnostics.json`: Task Arithmetic, lambda=1.0 over "
                 "dW_i=(alpha/r)B_iA_i of E_reason and E_agent. Because lambda=1 and both use alpha/r=2, the sum is exactly "
                 "this rank-32 adapter (verified max |diff| 4e-11). Selected on dev data among linear / TIES(d=0.2) / DARE-TIES(d=0.5).\n")
    with tempfile.TemporaryDirectory() as td:
        for f in Path(src).iterdir():
            if f.is_file() and f.name != "README.md":
                shutil.copy(f, td)
        Path(td, "README.md").write_text(CARD.format(name=name, desc=desc, base=BASE, rev=REV, r=cfg["r"], alpha=cfg["lora_alpha"], gh=GH, repo=repo, extra=extra))
        c = api.upload_folder(folder_path=td, repo_id=repo, commit_message="upload adapter + model card")
    out[name] = {"url": f"https://huggingface.co/{repo}", "revision": c.oid}
    print(name, out[name], flush=True)
json.dump(out, open("models/hf_links.json", "w"), indent=1)
