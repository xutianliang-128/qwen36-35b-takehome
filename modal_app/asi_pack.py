"""Pack ASI-Bench directories inside the Modal volume into single tarballs (reliable download)."""
import subprocess

import modal

vol = modal.Volume.from_name("qwen36-asi")
app = modal.App("qwen36-asi-pack")


@app.function(image=modal.Image.debian_slim(), volumes={"/asi": vol}, timeout=1800)
def pack(src: str, dst: str):
    subprocess.run(f"cd /asi && tar czf {dst} '{src}' && ls -la {dst} && tar tzf {dst} | wc -l", shell=True, check=True)
    vol.commit()
