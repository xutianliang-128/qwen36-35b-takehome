"""bf16 LoRA SFT of Qwen3.6-35B-A3B on local GPUs (layer-wise model parallel over all visible GPUs).

Input: jsonl with {"messages": [...]} (assistant turns may carry "reasoning_content").
Loss is computed only on assistant tokens. Logs every optimizer step to <out>/metrics.jsonl,
evaluates validation loss periodically, saves the adapter at the end (and every --save-every steps).

    PYTHONNOUSERSITE=1 python src/train/train_lora.py --config configs/train/e_reason.yaml
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, AutoTokenizer

LORA_TARGETS = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))"
)


def encode(tok, messages, max_len):
    """Token ids + labels; only the LAST assistant turn is a target (everything before it is context).

    prompt = history rendered with the generation prompt (ends in "<|im_start|>assistant\n<think>\n"), exactly what
    the model sees at inference; completion = the rest of the full rendering. Tokenizing the two parts separately
    keeps the boundary exact. Multi-turn agent samples are built one target turn per example, so history turns
    are rendered the way the chat template renders them at inference time (no earlier reasoning).
    """
    full = tok.apply_chat_template(messages, tokenize=False)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    assert full.startswith(prompt), "chat template rendering is not prefix-stable"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_len:
        return None
    return ids, [-100] * len(p_ids) + c_ids


def chunked_lm_loss(lm_head, hidden, labels, chunk=1024):
    """Mean next-token CE over target tokens without materializing full [seq, 248K] logits."""
    hidden, labels = hidden[:, :-1], labels[:, 1:]
    n = (labels != -100).sum().clamp(min=1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[1], chunk):
        y = labels[:, i:i + chunk]
        if (y != -100).any():
            total = total + checkpoint(
                lambda h, y: F.cross_entropy(lm_head(h).float().flatten(0, 1), y.flatten(),
                                             reduction="sum", ignore_index=-100),
                hidden[:, i:i + chunk], y, use_reentrant=False)
    return total / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    data = {}
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(cfg[f"{split}_file"])]
        enc = [encode(tok, r["messages"], cfg["max_len"]) for r in rows]
        data[split] = [e for e in enc if e is not None]
        print(f"{split}: {len(data[split])}/{len(rows)} examples kept (<= {cfg['max_len']} tokens), "
              f"{sum(len(e[0]) for e in data[split])/1e6:.2f}M tokens", flush=True)

    n_gpu = torch.cuda.device_count()
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["base_model"], dtype=torch.bfloat16, device_map="auto",
        max_memory={i: cfg.get("weights_per_gpu", "24GiB") for i in range(n_gpu)}, attn_implementation="sdpa")
    model.config.use_cache = False
    model.set_experts_implementation("grouped_mm")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, target_modules=LORA_TARGETS))
    model.train()  # gradient checkpointing only applies in train mode
    inner = model.base_model.model
    head_dev = inner.lm_head.weight.device
    first_dev = next(model.parameters()).device

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0, betas=(0.9, 0.999))
    accum = cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(data["train"]) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(cfg["warmup_ratio"] * total_steps))
    lr_at = lambda s: cfg["lr"] * (s / warmup if s < warmup else
                                   0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    def fwd(ids, labels):
        x = torch.tensor([ids], device=first_dev)
        y = torch.tensor([labels], device=head_dev)
        hidden = inner.model(input_ids=x).last_hidden_state.to(head_dev)
        return chunked_lm_loss(inner.lm_head, hidden, y)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for ids, labels in data["val"]:
            loss, k = fwd(ids, labels)
            tot += loss.item() * k.item()
            n += k.item()
        model.train()
        return tot / max(1, n)

    meta = {k: v for k, v in cfg.items()}
    meta.update(n_gpu=n_gpu, gpu=torch.cuda.get_device_name(0), total_steps=total_steps,
                trainable_params=model.get_nb_trainable_parameters()[0], torch=torch.__version__)
    json.dump(meta, open(out / "train_metadata.json", "w"), indent=1)
    mf = open(out / "metrics.jsonl", "a")
    log = lambda rec: (mf.write(json.dumps(rec) + "\n"), mf.flush(), print(json.dumps(rec), flush=True))

    val0 = evaluate()
    log({"step": 0, "tokens_seen": 0, "val_loss": val0})
    step, tokens_seen, t0 = 0, 0, time.time()
    for epoch in range(cfg["epochs"]):
        order = list(range(len(data["train"])))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        for b in range(0, len(order), accum):
            batch = [data["train"][i] for i in order[b:b + accum]]
            n_tgt = sum(sum(l != -100 for l in lab) for _, lab in batch)
            loss_sum = 0.0
            for ids, labels in batch:
                loss, k = fwd(ids, labels)
                (loss * k / n_tgt).backward()  # token-weighted mean over the whole batch
                loss_sum += loss.item() * k.item()
                tokens_seen += len(ids)
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]).item()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch, "tokens_seen": tokens_seen, "train_loss": loss_sum / n_tgt,
                   "lr": lr_at(step - 1), "grad_norm": gnorm, "elapsed_s": round(time.time() - t0, 1),
                   "peak_mem_gib": [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(n_gpu)]}
            if step % cfg["eval_every"] == 0 or step == total_steps:
                rec["val_loss"] = evaluate()
            log(rec)
            if cfg.get("save_every") and step % cfg["save_every"] == 0:
                model.save_pretrained(out / f"checkpoint-{step}")
    model.save_pretrained(out / "final")
    json.dump({"total_steps": step, "tokens_seen": tokens_seen, "wall_s": time.time() - t0},
              open(out / "train_done.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
