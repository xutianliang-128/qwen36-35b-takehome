"""Merge LoRA experts into the base model (or materialize a single expert) as full bf16 safetensors.

All methods operate on the dense task vectors dW_i = (alpha/r) * B_i @ A_i (never on A/B separately,
which is not a valid way to combine low-rank factors). Only LoRA-targeted modules change; everything else
(routed experts, router, embeddings, vision tower) is copied byte-for-byte from the base.

Methods (per target tensor):
  linear    W = W0 + sum_i lam_i * dW_i                               (task arithmetic; one expert = plain merge)
  ties      trim each dW_i to its top-`density` magnitudes, elect a sign per entry by summed mass,
            average only the values agreeing with the elected sign, scale by lam
  dare_ties randomly drop (1-density) of each dW_i, rescale by 1/density, then TIES sign election/averaging

Also writes conflict diagnostics: per-module cosine similarity between experts' dW and the sign-conflict
rate among entries where both are non-zero.

    python src/merge/merge_experts.py --method ties --density 0.2 --lam 1.0 \
        --expert e_reason=checkpoints/e_reason/final --expert e_agent=checkpoints/e_agent/final \
        --out /home/tianlix/interview/models/merged-ties-d0.2
"""

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/home/tianlix/interview/models/Qwen3.6-35B-A3B@995ad96e"


def load_deltas(adapter_dir: Path, device) -> dict[str, torch.Tensor]:
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    a, b = {}, {}
    with safe_open(adapter_dir / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            m = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", k)
            assert m, f"unexpected adapter key {k}"
            (a if m.group(2) == "A" else b)[m.group(1) + ".weight"] = f.get_tensor(k)
    assert a.keys() == b.keys()
    return {k: scale * (b[k].to(device, torch.float32) @ a[k].to(device, torch.float32)) for k in a}


def ties_combine(deltas: list[torch.Tensor], density: float, dare: bool, gen: torch.Generator) -> torch.Tensor:
    kept = []
    for d in deltas:
        if dare:
            # CPU generator: identical DARE masks whether the merge runs locally (GPU) or on Modal (CPU)
            mask = (torch.rand(d.shape, generator=gen) < density).to(d.device)
            d = d * mask / density
        else:
            k = max(1, int(density * d.numel()))
            thresh = d.abs().flatten().kthvalue(d.numel() - k + 1).values
            d = d * (d.abs() >= thresh)
        kept.append(d)
    stack = torch.stack(kept)
    elected = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == elected) & (stack != 0)
    return (stack * agree).sum(0) / agree.sum(0).clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", action="append", required=True, help="name=adapter_dir (repeatable)")
    ap.add_argument("--method", choices=["linear", "ties", "dare_ties"], default="linear")
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0],
                    help="linear: one weight per expert (or one shared); ties/dare_ties: single global scale")
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")  # shared server: never grab a GPU implicitly (pass --device cuda:N explicitly)
    args = ap.parse_args()

    t0 = time.time()
    names, dirs = zip(*(e.split("=", 1) for e in args.expert))
    deltas = [load_deltas(Path(d), args.device) for d in dirs]
    keys = sorted(deltas[0])
    assert all(set(d) == set(keys) for d in deltas), "experts must share LoRA target modules"
    lams = args.lam * len(names) if len(args.lam) == 1 else args.lam
    gen = torch.Generator().manual_seed(args.seed)

    merged, diag = {}, defaultdict(dict)
    for k in keys:
        ds = [d[k] for d in deltas]
        if args.method == "linear":
            merged[k] = sum(l * d for l, d in zip(lams, ds))
        else:
            merged[k] = lams[0] * ties_combine(ds, args.density, args.method == "dare_ties", gen)
        diag[k]["norms"] = [round(d.norm().item(), 6) for d in ds]
        diag[k]["merged_norm"] = round(merged[k].norm().item(), 6)
        if len(ds) >= 2:
            a, b = ds[0].flatten(), ds[1].flatten()
            diag[k]["cosine_0_1"] = round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 4)
            both = (a != 0) & (b != 0)
            diag[k]["sign_conflict_0_1"] = round(((torch.sign(a) != torch.sign(b)) & both).sum().item()
                                                 / max(1, both.sum().item()), 4)

    # write: copy base shards, rewriting only those that contain target tensors
    base = Path(args.base)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, args.out / f.name)
    weight_map = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    by_shard = defaultdict(list)
    for k in keys:
        by_shard[weight_map[k]].append(k)
    for shard in sorted(set(weight_map.values())):
        src, dst = base / shard, args.out / shard
        if shard not in by_shard:
            shutil.copy2(src, dst)
            continue
        with safe_open(src, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
            meta = f.metadata()
        for k in by_shard[shard]:
            w = tensors[k]
            tensors[k] = (w.to(args.device, torch.float32) + merged[k]).to(w.dtype).cpu()
        save_file(tensors, dst, metadata=meta)

    summary = {
        "method": args.method, "experts": dict(zip(names, dirs)), "lam": lams, "density": args.density,
        "seed": args.seed, "base": str(base), "n_modules": len(keys), "wall_s": round(time.time() - t0, 1),
    }
    if len(deltas) >= 2:
        cos = [v["cosine_0_1"] for v in diag.values()]
        sc = [v["sign_conflict_0_1"] for v in diag.values()]
        summary["mean_cosine_0_1"] = round(sum(cos) / len(cos), 4)
        summary["mean_sign_conflict_0_1"] = round(sum(sc) / len(sc), 4)
        by_type = defaultdict(list)
        for k, v in diag.items():
            by_type[re.sub(r"layers\.\d+\.", "", k).replace("model.language_model.", "")].append(v["cosine_0_1"])
        summary["cosine_by_module_type"] = {t: round(sum(c) / len(c), 4) for t, c in sorted(by_type.items())}
    json.dump(summary, open(args.out / "merge_config.json", "w"), indent=1)
    json.dump(diag, open(args.out / "merge_diagnostics.json", "w"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
