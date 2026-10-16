"""Rejection-sampling data generation (self-distillation) for E_reason.

The base model answers each pool question K times (thinking mode); a programmatic checker compares
the \\boxed{} answer with the gold answer. Raw samples are appended to <out>/samples.jsonl (resumable).

    VLLM_URL=http://localhost:8100 python src/data/gen_rft.py --pool data/reason/pool_train.jsonl --out data/reason
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from answer_check import is_correct, last_boxed

PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
# Model-card "thinking mode, general tasks" sampling.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


async def solve(client, sem, url, key, item, k, max_tokens, fout, stats, model):
    async with sem:
        body = {"model": model, "n": k, "max_tokens": max_tokens, **SAMPLING,
                "messages": [{"role": "user", "content": item["question"] + PROMPT_SUFFIX}]}
        try:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            stats["errors"] += 1
            print("request failed", item["id"], type(e).__name__, flush=True)
            return
    for i, ch in enumerate(out["choices"]):
        msg = ch["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        pred = last_boxed(content)
        ok = ch.get("finish_reason") == "stop" and is_correct(pred, item["answer"], item["answer_type"])
        rec = {**item, "sample_idx": i, "content": content, "reasoning": reasoning, "pred": pred,
               "finish_reason": ch.get("finish_reason"), "correct": ok,
               "completion_tokens": None}
        fout.write(json.dumps(rec) + "\n")
        stats["samples"] += 1
        stats["correct"] += ok
    stats["done"] += 1
    stats["completion_tokens"] += out.get("usage", {}).get("completion_tokens", 0)
    fout.flush()


async def main_async(args):
    url = os.environ["VLLM_URL"].rstrip("/") + "/v1/chat/completions"
    key = os.environ["VLLM_API_KEY"]
    out_path = args.out / "samples.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path)}
    items = [json.loads(l) for l in open(args.pool)]
    if args.max_questions:  # pool is already shuffled, so a prefix is a random subset
        items = items[: args.max_questions]
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} questions, {len(done)} already done, {len(todo)} to go", flush=True)
    stats = {"done": 0, "samples": 0, "correct": 0, "errors": 0, "completion_tokens": 0}
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async def progress():
        while True:
            await asyncio.sleep(120)
            dt = time.time() - t0
            print(f"[{dt/60:.0f} min] {stats['done']}/{len(todo)} questions, acc {stats['correct']}/{stats['samples']}, "
                  f"{stats['completion_tokens']/dt:.0f} tok/s, errors {stats['errors']}", flush=True)

    with open(out_path, "a") as fout:
        async with httpx.AsyncClient(timeout=httpx.Timeout(7200)) as client:
            p = asyncio.create_task(progress())
            await asyncio.gather(*[solve(client, sem, url, key, it, args.k, args.max_tokens, fout, stats, args.model) for it in todo])
            p.cancel()
    json.dump({**stats, "wall_s": time.time() - t0, "k": args.k, "max_tokens": args.max_tokens, "sampling": SAMPLING},
              open(args.out / "gen_stats.json", "w"), indent=1)
    print("finished", stats, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--model", default="qwen36-base", help="served model name")
    ap.add_argument("--max-questions", type=int, default=0, help="0 = whole pool")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
