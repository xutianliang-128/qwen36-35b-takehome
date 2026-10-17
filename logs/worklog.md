# Work log (active hours, training/eval wall time, GPU hours, cost)

| date | start-end (UTC) | activity | GPU type x count | GPU hours | cost (USD) |
|---|---|---|---|---|---|
| 2026-10-13 | | task analysis, hardware check, platform decision (Modal) | - | 0 | 0 |
| 2026-10-14 | | env setup, pin base revision, model download to Modal Volume | CPU only | 0 | |
| 2026-10-14 | | Modal vLLM smoke test (failed: nvcc; fixed: official image; passed: 316s cold start, 240 tok/s) | H100 x2 | ~0.5 | see Modal usage |
| 2026-10-14 | | TB2 snapshot frozen; Harbor+Terminus-2 wiring; first TB2 run failed (missing dockerfile-parse) | H100 x2 | ~0.1 | |
| 2026-10-14 | | local envs (vllm / train), local model copy, LoRA speed test (1.4k tok/s, 32K fits) | A6000 x3 (local) | ~0.5 | 0 |
| 2026-10-13..16 | total | **active human work ≈ 20 h** (self-reported); machine runs unattended otherwise | | | |
