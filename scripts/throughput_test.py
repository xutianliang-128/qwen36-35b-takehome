"""Aggregate decode throughput of an OpenAI-compatible endpoint at several concurrency levels.

    VLLM_URL=http://localhost:8100 python scripts/throughput_test.py --concurrency 1,16,64 --out logs/setup/throughput_local.json
"""

import argparse
import asyncio
import json
import os
import time

import httpx

PROMPTS = [
    "Prove that there are infinitely many primes of the form 4k+3.",
    "Write a bash one-liner that finds the 10 largest files under /var/log, then explain each flag.",
    "A ball is thrown upward at 20 m/s from a 45 m cliff. When does it hit the ground? Show work.",
    "Explain the difference between a mutex and a semaphore with a Python example.",
]


async def one(client, url, key, i, max_tokens):
    body = {"model": "qwen36-base", "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}
    r = await client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


async def run(conc, url, key, max_tokens):
    async with httpx.AsyncClient(timeout=3600) as client:
        t = time.time()
        toks = await asyncio.gather(*[one(client, url, key, i, max_tokens) for i in range(conc)])
        dt = time.time() - t
    return {"concurrency": conc, "completion_tokens": sum(toks), "wall_s": round(dt, 1),
            "aggregate_tok_s": round(sum(toks) / dt, 1), "per_stream_tok_s": round(sum(toks) / dt / conc, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", default="1,16,64")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = os.environ["VLLM_URL"].rstrip("/") + "/v1/chat/completions"
    key = os.environ["VLLM_API_KEY"]
    res = []
    for c in map(int, a.concurrency.split(",")):
        r = asyncio.run(run(c, url, key, a.max_tokens))
        print(json.dumps(r), flush=True)
        res.append(r)
    json.dump({"endpoint": os.environ["VLLM_URL"], "max_tokens": a.max_tokens, "runs": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
