"""Send one thinking-mode request to the vLLM endpoint and report latency / token usage.

    VLLM_URL=https://<workspace>--qwen36-vllm-serve.modal.run python scripts/smoke_test.py
"""

import json
import os
import time
import urllib.request

url = os.environ["VLLM_URL"].rstrip("/") + "/v1/chat/completions"
key = os.environ["VLLM_API_KEY"]

body = {
    "model": "qwen36-base",
    "messages": [{"role": "user", "content": "How many prime numbers are there below 100? Answer with a number."}],
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 8192,
}
req = urllib.request.Request(
    url, data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    out = json.load(r)
dt = time.time() - t0

msg = out["choices"][0]["message"]
usage = out["usage"]
print("answer:", msg["content"].strip()[:200])
print("reasoning chars:", len(msg.get("reasoning_content") or msg.get("reasoning") or ""))
print(f"latency {dt:.1f}s, completion tokens {usage['completion_tokens']}, "
      f"{usage['completion_tokens'] / dt:.1f} tok/s (single stream, incl. queue)")
