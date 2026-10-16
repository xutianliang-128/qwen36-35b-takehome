"""Serve Qwen3.6-35B-A3B (base or merged) with vLLM on Modal.

Usage (run from repo root):
    modal run modal_app/serve_vllm.py::download      # one-off: weights -> Modal Volume
    modal run modal_app/serve_vllm.py::check_image   # CPU-only: build image, check nvcc/vllm/weights
    modal deploy modal_app/serve_vllm.py             # OpenAI-compatible endpoint
    python scripts/smoke_test.py                     # check the endpoint
"""

import os
import subprocess

import modal

BASE_REPO = "Qwen/Qwen3.6-35B-A3B"
BASE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"  # pinned before any training
VLLM_VERSION = "0.30.0"

MODELS_DIR = "/models"
BASE_PATH = f"{MODELS_DIR}/Qwen3.6-35B-A3B@{BASE_REVISION[:8]}"
# Idle seconds before the GPU container shuts down (billed while idle). Read at deploy time.
SCALEDOWN_S = int(os.environ.get("SCALEDOWN_S", "120"))
N_GPU = 2  # bf16 weights ~72GB; TP=2 leaves room for KV cache on H100-80GB

models_vol = modal.Volume.from_name("qwen36-models", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("qwen36-vllm-cache", create_if_missing=True)

# Official vLLM image: ships the CUDA toolkit (nvcc) that vLLM needs for JIT-compiled kernels.
# A pip-only debian_slim image crashed at startup with "Could not find nvcc".
image = (
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}-x86_64")
    .entrypoint([])
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")  # Modal looks for `python`
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"})  # match local serving
)

app = modal.App("qwen36-vllm")


@app.function(image=image, volumes={MODELS_DIR: models_vol}, timeout=2 * 60 * 60, cpu=8)
def download():
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASE_REPO, revision=BASE_REVISION, local_dir=BASE_PATH)
    models_vol.commit()
    print("downloaded to", path)


@app.function(image=image, volumes={MODELS_DIR: models_vol}, cpu=2)
def check_image():
    """CPU-only: build the image and verify the toolchain before paying for GPUs."""
    import os
    import shutil

    import vllm

    print("vllm", vllm.__version__)
    print("nvcc", shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc exists=" + str(os.path.exists("/usr/local/cuda/bin/nvcc")))
    print("weights", sorted(os.listdir(BASE_PATH))[:3], "...", len(os.listdir(BASE_PATH)), "files")


@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",
    volumes={MODELS_DIR: models_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("qwen36-vllm-key")],  # provides VLLM_API_KEY
    timeout=24 * 60 * 60,
    scaledown_window=SCALEDOWN_S,
    max_containers=1,  # one 2xH100 server; never autoscale (cost + identical serving for every eval)
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve():
    cmd = [
        "vllm", "serve", BASE_PATH,
        "--served-model-name", "qwen36-base",
        "--tensor-parallel-size", str(N_GPU),
        "--max-model-len", "262144",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--enable-prefix-caching",
        "--host", "0.0.0.0", "--port", "8000",
    ]
    subprocess.Popen(cmd)


# ---- Batch-eval mode: raw TCP tunnel (no web-gateway request timeout) + on-Modal merging -----------------
tunnels = modal.Dict.from_name("qwen36-tunnels", create_if_missing=True)


@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",
    volumes={MODELS_DIR: models_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("qwen36-vllm-key")],
    timeout=7 * 60 * 60,  # hard cost cap per launch (ASI runs need up to ~4-5h)
)
def tunnel(model_dir: str, served_name: str, dict_key: str = ""):
    """Serve /models/<model_dir> as <served_name>; publish the tunnel URL in the qwen36-tunnels Dict."""
    cmd = [
        "vllm", "serve", f"{MODELS_DIR}/{model_dir}", "--served-model-name", served_name,
        "--tensor-parallel-size", str(N_GPU), "--max-model-len", "262144",
        "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--enable-prefix-caching", "--host", "0.0.0.0", "--port", "8000",
    ]
    proc = subprocess.Popen(cmd)
    with modal.forward(8000) as t:
        tunnels[dict_key or served_name] = t.url
        print("TUNNEL", served_name, t.url, flush=True)
        proc.wait()


merge_image = image.add_local_file("src/merge/merge_experts.py", "/root/merge_experts.py")


@app.function(image=merge_image, volumes={MODELS_DIR: models_vol}, cpu=16, memory=98304, timeout=2 * 60 * 60)
def merge_remote(args: list[str]):
    """Run src/merge/merge_experts.py on Modal against the base copy in the volume (adapters under /models/adapters)."""
    subprocess.run(["python", "/root/merge_experts.py", "--base", BASE_PATH, "--device", "cpu", *args], check=True)
    models_vol.commit()
