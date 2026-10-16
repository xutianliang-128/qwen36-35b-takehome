"""HLE inference (full multimodal set, no tools) against an OpenAI-compatible endpoint. Resumable.

System prompts are the official ones from the HLE repo (run_model_predictions.py). Image questions send
the image as a data URL. Judging is a separate step (src/eval/hle_judge.py).

    VLLM_URL=http://localhost:8100 python src/eval/hle_generate.py --model qwen36-base --out results/qwen36-base/hle/run1
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from huggingface_hub import hf_hub_download

HLE_REPO, HLE_REVISION = "cais/hle", "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
SYSTEM_EXACT = ("Your response should be in the following format:\nExplanation: {your explanation for your final answer}\n"
                "Exact Answer: {your succinct, final answer}\n"
                "Confidence: {your confidence score between 0% and 100% for your answer}")
SYSTEM_MC = ("Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\n"
             "Answer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}")
# Model-card thinking-mode sampling; identical for every evaluated model (configs/eval/hle.yaml).
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def load_hle():
    p = hf_hub_download(HLE_REPO, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=HLE_REVISION)
    return pd.read_parquet(p, columns=["id", "question", "image", "answer", "answer_type", "category"])


def build_messages(row):
    system = SYSTEM_EXACT if row.answer_type == "exactMatch" else SYSTEM_MC
    content = [{"type": "text", "text": row.question}]
    if row.image:
        content.append({"type": "image_url", "image_url": {"url": row.image}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


async def one(client, sem, url, key, model, row, max_tokens, seed, fout, stats):
    async with sem:
        body = {"model": model, "messages": build_messages(row), "max_tokens": max_tokens, "seed": seed, **SAMPLING}
        t = time.time()
        rec = {"id": row.id, "answer_type": row.answer_type, "category": row.category, "has_image": bool(row.image)}
        try:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            out = r.json()
            ch = out["choices"][0]
            rec.update(status="ok", response=ch["message"].get("content") or "",
                       reasoning=ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or "",
                       finish_reason=ch.get("finish_reason"), usage=out.get("usage"))
            stats["tokens"] += out.get("usage", {}).get("completion_tokens", 0)
        except Exception as e:  # recorded, never dropped: counts as a run failure in the summary
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
            stats["errors"] += 1
        rec["latency_s"] = round(time.time() - t, 1)
    fout.write(json.dumps(rec) + "\n")
    fout.flush()
    stats["done"] += 1


async def main_async(a):
    url = os.environ["VLLM_URL"].rstrip("/") + "/v1/chat/completions"
    key = os.environ["VLLM_API_KEY"]
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "responses.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    df = load_hle()
    todo = [r for r in df.itertuples() if r.id not in done]
    print(f"HLE {len(df)} questions, {len(done)} done, {len(todo)} to go", flush=True)
    json.dump({"dataset": f"{HLE_REPO}@{HLE_REVISION}", "split": "test", "n": len(df), "model": a.model,
               "max_tokens": a.max_tokens, "seed": a.seed, "sampling": SAMPLING, "tools": "none",
               "images": "included (data URL)", "system_prompts": {"exactMatch": SYSTEM_EXACT, "multipleChoice": SYSTEM_MC}},
              open(a.out / "config.json", "w"), indent=1)
    stats = {"done": 0, "errors": 0, "tokens": 0}
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()

    async def progress():
        while True:
            await asyncio.sleep(300)
            dt = time.time() - t0
            print(f"[{dt/60:.0f} min] {stats['done']}/{len(todo)}, errors {stats['errors']}, {stats['tokens']/dt:.0f} tok/s", flush=True)

    with open(path, "a") as fout:
        async with httpx.AsyncClient(timeout=httpx.Timeout(7200)) as client:
            p = asyncio.create_task(progress())
            await asyncio.gather(*[one(client, sem, url, key, a.model, r, a.max_tokens, a.seed, fout, stats) for r in todo])
            p.cancel()
    print("finished", stats, f"{(time.time()-t0)/60:.0f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=96)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
