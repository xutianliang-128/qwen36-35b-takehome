"""Materialize an expert / merged checkpoint inside the Modal volume (adapters must be uploaded to /models/adapters).

    python scripts/modal_merge.py --out /models/expert-e_reason --method linear --lam 1.0 --expert e_reason=/models/adapters/e_reason
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))
from serve_vllm import app, merge_remote  # noqa: E402

with app.run():
    merge_remote.remote(sys.argv[1:])
