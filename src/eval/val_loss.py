"""Held-out SFT loss of full checkpoints (base / experts / merges) on the experts' validation sets.

Used only for choosing the merge configuration on dev data (never on benchmark data).

    python src/eval/val_loss.py --models a=/path/a b=/path/b --out results/dev/val_loss.json
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train_lora import chunked_lm_loss, encode  # noqa: E402

SETS = {"agent_val": ("data/agent/sft_val.jsonl", 16384), "reason_val": ("data/reason/sft_val.jsonl", 20480)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint_dir")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = json.load(open(a.out)) if a.out.exists() else {}
    tok = None
    for spec in a.models:
        name, path = spec.split("=", 1)
        if name in res:
            continue
        t0 = time.time()
        tok = tok or AutoTokenizer.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto",
            max_memory={i: "24GiB" for i in range(torch.cuda.device_count())}, attn_implementation="sdpa")
        model.eval()
        model.config.use_cache = False  # without this the eval forward keeps a cache for 16K-token samples -> OOM
        model.set_experts_implementation("grouped_mm")
        first, head = next(model.parameters()).device, model.lm_head.weight.device
        res[name] = {"path": path}
        for set_name, (f, max_len) in SETS.items():
            tot, n = 0.0, 0
            for line in open(f):
                e = encode(tok, json.loads(line)["messages"], max_len)
                if e is None:
                    continue
                ids, lab = e
                h = model.model(input_ids=torch.tensor([ids], device=first), use_cache=False).last_hidden_state.to(head)
                loss, k = chunked_lm_loss(model.lm_head, h, torch.tensor([lab], device=head))
                tot += loss.item() * k.item()
                n += k.item()
            res[name][set_name] = round(tot / n, 5)
        res[name]["wall_s"] = round(time.time() - t0, 1)
        print(name, res[name], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del model, h, loss
        gc.collect()  # dangling references otherwise keep the previous checkpoint resident -> OOM on the next load
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
