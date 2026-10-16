"""Measure local LoRA training throughput / peak memory for Qwen3.6-35B-A3B (bf16, no quantization).

Loads the model once, split layer-wise over all visible GPUs (naive model parallel, no NVLink needed),
then runs a few optimizer steps on random tokens for each (experts_implementation, seq_len) combo.

    PYTHONNOUSERSITE=1 python src/train/speed_test.py --out logs/setup/train_speed.json
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText

MODEL = "/home/tianlix/interview/models/Qwen3.6-35B-A3B@995ad96e"

# Language-model linears only. Router (mlp.gate, not nn.Linear) and the 256 routed experts
# (3D params) stay frozen; vision tower is frozen as well.
LORA_TARGETS = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))"
)


def chunked_lm_loss(lm_head, hidden, labels, chunk=1024):
    """Next-token CE without materializing [seq, 248K-vocab] logits (8K tokens -> ~8GB in fp32)."""
    hidden, labels = hidden[:, :-1], labels[:, 1:]
    n = (labels != -100).sum()
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[1], chunk):
        h, y = hidden[:, i:i + chunk], labels[:, i:i + chunk]
        total = total + checkpoint(
            lambda h, y: F.cross_entropy(lm_head(h).float().flatten(0, 1), y.flatten(), reduction="sum", ignore_index=-100),
            h, y, use_reentrant=False)
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default="grouped_mm")
    ap.add_argument("--seq-lens", default="4096,8192,16384,32768")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_gpu = torch.cuda.device_count()
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        # Even split (~23GB weights/GPU) so every GPU keeps room for activations.
        max_memory={i: "24GiB" for i in range(n_gpu)},
        attn_implementation="sdpa",
    )
    load_s = time.time() - t0
    model.config.use_cache = False
    model.train()  # from_pretrained returns eval mode; gradient checkpointing only applies in train mode
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=LORA_TARGETS,
    ))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"loaded in {load_s:.0f}s; trainable {trainable/1e6:.1f}M / {total/1e9:.2f}B")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    first_dev = next(model.parameters()).device
    inner = model.base_model.model  # Qwen3_5MoeForConditionalGeneration

    results = {"load_s": load_s, "trainable_params": trainable, "n_gpu": n_gpu,
               "gpu": torch.cuda.get_device_name(0), "rank": args.rank, "runs": []}
    for impl in args.impls.split(","):
        model.base_model.model.set_experts_implementation(impl)
        for L in map(int, args.seq_lens.split(",")):
            for i in range(n_gpu):
                torch.cuda.reset_peak_memory_stats(i)
            times, err = [], None
            try:
                for step in range(args.steps + 1):  # first step = warmup
                    ids = torch.randint(0, 200_000, (1, L), device=first_dev)
                    torch.cuda.synchronize()
                    t = time.time()
                    hidden = inner.model(input_ids=ids).last_hidden_state
                    loss = chunked_lm_loss(inner.lm_head, hidden.to(inner.lm_head.weight.device), ids.to(inner.lm_head.weight.device))
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    if step > 0:
                        times.append(time.time() - t)
            except Exception as e:  # OOM or unsupported kernel: record and continue
                err = f"{type(e).__name__}: {str(e)[:300]}"
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            run = {
                "experts_impl": impl, "seq_len": L, "error": err,
                "step_s": sum(times) / len(times) if times else None,
                "tokens_per_s": L * len(times) / sum(times) if times else None,
                "peak_mem_gib": [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(n_gpu)],
            }
            print(json.dumps(run), flush=True)
            results["runs"].append(run)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
