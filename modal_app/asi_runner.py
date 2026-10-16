"""ASI-Bench (official repo @cf07cb89 = v0.2.0, seed31415) inside an isolated Modal CPU container.

Agent: official opencode_cli adapter (opencode-ai@1.17.15, pinned in the ASI-Bench README) against our
OpenAI-compatible vLLM endpoint; sandbox "task" (per-task Python venv) inside the container. Model-written
code therefore never runs on the shared lab server. Scoring is done afterwards on the host with
`asibench score --repo seed31415` (public references; non-official score).

    modal run --detach modal_app/asi_runner.py::asi_run --served-name X --api-base https://.../v1 --levels b4
"""

import json
import subprocess

import modal

ASI_COMMIT = "cf07cb89d8590d958ff42db7ae3d6608f5eb9dbc"
DATA_REV = "0fa14219cafdbab634d8b3cfbce238a8735a214f"  # Apexintelligence-AI/ASI-Bench-seed31415
FULL, NOREF = f"/asi/instances_seed31415@{DATA_REV[:8]}", f"/asi/instances_seed31415@{DATA_REV[:8]}_noref"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates", "rsync")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs",
        "npm install -g opencode-ai@1.17.15",
        "pip install uv",
        f"git clone https://github.com/apexin-ai/ASI-Bench.git /opt/ASI-Bench && cd /opt/ASI-Bench && git checkout {ASI_COMMIT}",
        "cd /opt/ASI-Bench && pip install -e '.[full]'",
        # The agent runs without filesystem isolation ("task" sandbox), so it must not be able to read graders.
        # Public copy of the task dirs keeps only metadata + prompts; scorers, eval contracts, GT generators and the
        # git history are removed from the image. Scoring happens later on the host with the full repo.
        "mkdir -p /opt/asi_public && cp -r /opt/ASI-Bench/config /opt/asi_public/ && cp -r /opt/ASI-Bench/tasks /opt/asi_public/tasks"
        " && find /opt/asi_public/tasks -type f ! -name task_meta.yaml ! -name task.yaml ! -name 'prompt_b*.md' -delete"
        " && rm -rf /opt/ASI-Bench/tasks /opt/ASI-Bench/examples /opt/ASI-Bench/.git",
        # opencode takes its project dir from $PWD (inherited from asibench), not from the subprocess cwd; pin it to
        # the actual cwd (= the instance workspace) so the agent works and writes inside its own workspace.
        "mv /usr/bin/opencode /usr/bin/opencode-real && printf '#!/bin/sh\\nPWD=\"$(pwd -P)\"; export PWD\\nexec /usr/bin/opencode-real \"$@\"\\n' > /usr/bin/opencode && chmod +x /usr/bin/opencode && cat /usr/bin/opencode",
    )
)
vol = modal.Volume.from_name("qwen36-asi", create_if_missing=True)
app = modal.App("qwen36-asi")


@app.function(image=image, volumes={"/asi": vol}, secrets=[modal.Secret.from_name("qwen36-vllm-key")],
              cpu=32, memory=98304, timeout=8 * 60 * 60)
def asi_run(served_name: str, api_base: str, levels: str = "b4", parallel: int = 60, tasks: str = "",
            run_id: str = "asi-run1", timeout_s: int = 10800):
    import os
    os.makedirs("/tmp/launch", exist_ok=True)
    sh = lambda c: subprocess.run(c, shell=True, check=False, cwd="/tmp/launch")  # neutral launch dir
    # seed31415 ships public reference answers next to each instance. The agent only ever sees a copy WITHOUT
    # reference/ dirs; the full copy is used for scoring on the host afterwards.
    sh(f"[ -d {FULL} ] || asibench task pull --repo seed31415 --revision {DATA_REV} --output-dir {FULL}")
    sh(f"[ -d {NOREF} ] || rsync -a --exclude reference/ {FULL}/ {NOREF}/ || "
       f"(cp -r {FULL} {NOREF} && find {NOREF} -type d -name reference -prune -exec rm -rf {{}} +)")
    sh(f"echo refs_in_agent_copy=$(find {NOREF} -type d -name reference | wc -l); "
       "echo scorer_files_visible=$(find / -name custom_scorer.py -not -path '/proc/*' 2>/dev/null | wc -l)")
    vol.commit()
    cfg = {"model": served_name, "api_base": api_base, "api_key_env": "VLLM_API_KEY", "api_protocol": "openai"}
    out = f"/asi/out/{served_name}/{run_id}"
    resume = f"--resume {out}" if subprocess.run(f"[ -d {out} ]", shell=True).returncode == 0 else ""
    cmd = (f"asibench run --instances-dir {NOREF} --agent opencode_cli "
           f"--agent-config '{json.dumps(cfg)}' --sandbox task --prompt-levels {levels} --parallel {parallel} "
           f"--timeout {timeout_s} --output-dir {out} --tasks-dir /opt/asi_public/tasks {resume} " + (f"--tasks {tasks}" if tasks else ""))
    print(cmd, flush=True)
    rc = sh(cmd).returncode
    vol.commit()
    print("asibench run exit", rc, flush=True)
