"""Freeze a local snapshot of Terminal-Bench 2.0 with the experiment's fixed agent timeout.

Downloads `terminal-bench@2.0` via Harbor, sets every task's agent timeout to 3h (model-card
setting, see configs/eval/terminal_bench.yaml), and writes a manifest with per-task hashes so
every evaluated model runs on byte-identical tasks.

    python scripts/prepare_tb2.py --out ~/interview/eval_data/terminal-bench-2.0
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AGENT_TIMEOUT_SEC = 3 * 60 * 60
REPO = Path(__file__).resolve().parents[1]


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--harbor", default=str(Path(sys.executable).parent / "harbor"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([args.harbor, "download", "terminal-bench@2.0", "-o", tmp], check=True)
        src = Path(tmp) / "terminal-bench"
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.copytree(src, args.out)

    manifest = {"dataset": "terminal-bench@2.0", "agent_timeout_sec": AGENT_TIMEOUT_SEC, "tasks": []}
    for task in sorted(p for p in args.out.iterdir() if p.is_dir()):
        toml = task / "task.toml"
        text = toml.read_text()
        orig_sha = tree_sha256(task)
        # Only the [agent] section's timeout is changed; verifier/build timeouts stay official.
        text, n = re.subn(r"(\[agent\]\s*\ntimeout_sec\s*=\s*)[0-9.]+", rf"\g<1>{AGENT_TIMEOUT_SEC}.0", text)
        assert n == 1, f"{task.name}: expected one [agent] timeout_sec, found {n}"
        toml.write_text(text)
        meta = re.search(r'difficulty\s*=\s*"(\w+)"', text)
        image = re.search(r'docker_image\s*=\s*"([^"]+)"', text)
        manifest["tasks"].append({
            "name": task.name,
            "difficulty": meta.group(1) if meta else None,
            "docker_image": image.group(1) if image else None,
            "sha256_official": orig_sha,
            "sha256_frozen": tree_sha256(task),
        })

    out_manifest = REPO / "configs/eval/terminal_bench_2.0_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest['tasks'])} tasks -> {args.out}\nmanifest -> {out_manifest}")


if __name__ == "__main__":
    main()
