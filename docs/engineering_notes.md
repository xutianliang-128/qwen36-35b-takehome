# Engineering notes

Decisions and pitfalls hit while setting up the experiment, with the evidence behind each.
Logs referenced here are in `logs/setup/`.

## 1. Compute platform

| Resource | Use | Why |
|---|---|---|
| Local 3x RTX A6000 48GB (Ampere, PCIe, no NVLink; GPU2 on another NUMA node), 96-core EPYC 9274F, 1TB RAM | LoRA training, HLE inference, dev-set merge ablations, data generation | Free; total budget is USD 200 |
| Modal 2x H100-80GB (vLLM 0.30.0, TP=2) | Burst inference for agent benchmarks | Only when local throughput is insufficient |
| Modal sandboxes (CPU) | Terminal-Bench 2.0 / SWE-bench Pro task containers | No Docker or sudo on the local machine; rootless podman fails (no `/etc/subuid` entry) |

Model kept at 35B (no 7B fallback): with ~3B active parameters, 35B-A3B decodes about as fast as a 7B dense model,
and both inference and LoRA training fit in local memory (measured below).

## 2. Base model facts that drive design (from `config.json`, revision `995ad96e`)

- `Qwen3_5MoeForConditionalGeneration`, 35.95B total params, bf16 weights ~67GB on disk / ~72GB in memory.
- 40 layers: 30 Gated-DeltaNet linear-attention + 10 full-attention (2 KV heads x 256 dim).
  KV cache ~20KB/token, so 256K-context agent runs are cheap in memory.
- MoE: 256 routed experts (top-8, intermediate 512) + 1 shared expert per layer.
  Routed experts are stored as fused **3D parameters** (`experts.gate_up_proj [256,1024,2048]`, `experts.down_proj`).
- Multimodal (vision tower), so HLE image questions can be answered natively.

## 3. Training setup (bf16 LoRA, no QLoRA)

**LoRA targets** (language model only; 19.2M trainable params at rank 16):

| Module | Layers | Shape |
|---|---|---|
| `self_attn.{q,k,v,o}_proj` | 10 full-attention layers | q 8192x2048, k/v 512x2048, o 2048x4096 |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | 30 linear-attention layers | 8192x2048, 4096x2048, 2048x4096 |
| `mlp.shared_expert.{gate,up,down}_proj` | 40 layers | 512x2048 / 2048x512 |

**Frozen**: router (`mlp.gate`, not an `nn.Linear`), all 256 routed experts (3D params), vision tower,
`linear_attn.in_proj_{a,b}` (32x2048 gating projections), embeddings, `lm_head`.

Rationale:
- Router frozen: if each expert changed routing, merged models would see conflicting routing decisions.
- Routed experts frozen: 256 experts x 40 layers would multiply adapter size and cost; merging them is also harder.
- **No QLoRA**: bitsandbytes only quantizes `nn.Linear`; the fused 3D expert tensors (the bulk of the weights)
  would stay bf16, so memory savings are small. A QLoRA adapter is also trained against a quantized base,
  which mismatches the bf16 base used for merging and serving.

## 4. Local training throughput (measured, `logs/setup/train_speed.json`)

Script: `src/train/speed_test.py`. 3x A6000, bf16, LoRA r=16, layer-wise model parallel (`device_map`),
gradient checkpointing, batch size 1, random tokens, 5 timed steps after 1 warm-up.

| seq len | tokens/s | step (s) | peak mem per GPU (GiB) |
|---|---|---|---|
| 4096 | 1349 | 3.0 | 23.8 / 25.2 / 22.9 |
| 8192 | 1468 | 5.6 | 25.4 / 26.8 / 23.1 |
| 16384 | 1468 | 11.2 | 28.6 / 30.0 / 26.1 |
| 32768 | 1363 | 24.0 | 35.0 / 36.4 / 32.3 |

About 5M tokens/hour, so three experts at ~8M tokens each take ~5h on local GPUs.
Random tokens give a different expert-load distribution from real data; expect about ±20% on real data.

### Pitfalls fixed to get here

1. **Uneven `device_map="auto"`** put ~40GB of weights on GPU0, leaving no room for activations, so it OOMed at 8K.
   Fix: `max_memory={i: "24GiB"}` gives an even ~23GB-per-GPU split.
2. **248K vocabulary**: full logits for 8K tokens are ~8GB in fp32. Fix: chunked cross-entropy
   (1024-token chunks, each checkpointed) so full logits are never materialized.
3. **Gradient checkpointing silently off**: `from_pretrained` returns the model in eval mode, and
   `GradientCheckpointingLayer` only recomputes in train mode. Peak memory at 4K was 34-38GB. With
   `model.train()` it is 23-25GB, and 32K sequences fit.
4. **MoE expert kernel**: transformers' default `eager` experts loop runs 766 tok/s vs 1878 tok/s for
   `grouped_mm` (4K, without checkpointing), so `set_experts_implementation("grouped_mm")` is 2.5x faster on Ampere.
5. **Environment isolation**: pip treated packages in `~/.local` as installed, so new conda envs silently depended
   on user site-packages. All envs are installed and run with `PYTHONNOUSERSITE=1`.

Environments:
- `qwen36-train`: torch 2.13.0+cu130, transformers 5.19.0, peft 0.21.2, flash-linear-attention 0.5.2
- `qwen36-vllm`: vllm 0.30.0, torch 2.13.0+cu130
- `qwen36`: modal 1.6.1, harbor 0.24.0 (orchestration)

## 5. Serving (vLLM on Modal)

- Image: official `vllm/vllm-openai:v0.30.0-x86_64`. A pip-only `debian_slim` image crashed at engine start with
  `Could not find nvcc` (vLLM JIT-compiles kernels), and Modal kept retrying it while billing GPU time.
  Symlink `python -> python3` added, because Modal could not detect the image's Python otherwise.
- Flags follow the model card: `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`,
  `--max-model-len 262144`, prefix caching on, bf16, TP=2.
- Endpoint protected by `VLLM_API_KEY` (Modal secret); unauthenticated requests return 401.
- Measured: cold start ~273-316s; single-stream decode ~240 tok/s on 2x H100.
- Cost control: `SCALEDOWN_S` (default 120s idle) and scripts that always `modal app stop` on exit
  (`scripts/smoke_modal.sh`). Validate images on CPU first (`check_image`) before paying for GPUs.

## 6. Terminal-Bench 2.0 harness

- Dataset `terminal-bench@2.0` (Harbor legacy registry, 89 tasks), frozen locally by `scripts/prepare_tb2.py`
  with per-task SHA-256 in `configs/eval/terminal_bench_2.0_manifest.json`.
- Official tasks request 1-8 CPU / 2-16GB and 3-120 min agent timeouts. The task spec (model-card setting) uses
  32 CPU / 48GB / 3h, so the frozen copy sets every `[agent] timeout_sec = 10800` and the run passes
  `--override-cpus 32 --override-memory-mb 49152`. Verifier and build timeouts stay official.
  Scores are therefore "this experiment's setting", not the official leaderboard setting.
- **Cost pitfall (measured):** with Harbor's default resource mode, the Modal backend *reserves* the overridden
  32 CPU (`cpu=(32,32)`) and requests 48GB, so every sandbox is billed for the full reservation: the 9-task pilot cost
  $13.77 of sandbox time (~$1.84 per sandbox-hour, about $150 per 89-task run). `--cpus limit --memory limit`
  passes `(small request, 32/48GB limit)`, so billing follows actual usage while the task can still use 32 CPU / 48GB.
- `prompt + max_tokens` must fit in 256K, so Terminus-2 gets `max_input_tokens = 262144 - 81920 = 180224`
  and summarizes before that point.
- Harbor's Modal backend needs `dockerfile-parse`, which is not a declared dependency (the first run failed after 4s).

## 7. Benchmark sources found

| Benchmark | Source | Notes |
|---|---|---|
| SWE-bench Pro v2-hard | Scale Labs, 51 hardest tasks of the 642-task public v2 split | Official task list and evaluator commit still to be pinned; Qwen's model card used an internal scaffold on an internally corrected set, so their score is not reproducible here |
| HLE | official 2500 questions (~14% with images) | full multimodal as primary metric; text-only reported separately |
| ASI-Bench | `asibench==0.2.0`, arXiv 2608.17271, GitHub apexin-ai/ASI-Bench (Apache-2.0) | 60 research tasks, 11 domains, guidance levels B1-B4 |
| Terminal-Bench 2.0 | `terminal-bench@2.0` via Harbor 0.24.0 | 89 tasks x 5 runs |
