"""Thinking-budget forcing for HLE (Qwen's documented thinking-budget recipe), applied identically to every model.

For every response that hit the 8K thinking budget (finish_reason == "length"), the thinking is closed with
Qwen's early-stop sentence and the model continues for at most 1024 tokens to write its final answer in the
official HLE format. Non-truncated responses are copied unchanged (and their judgements reused if present).
Output: <run>_bf/responses.jsonl (+ judged.jsonl seeds) -> judge with hle_judge.py --run <run>_bf.

    VLLM_URL=... python src/eval/hle_budget_force.py --model qwen36-base --run results/qwen36-base/hle/run1
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from hle_generate import SAMPLING, build_messages, load_hle

EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
              "directly now.\n</think>\n\n")
ANSWER_TOKENS = 1024


async def force(client, sem, url, key, model, rec, q, fout):
    msgs = build_messages(q) + [{"role": "assistant", "content": "<think>\n" + rec["reasoning"].rstrip() + EARLY_STOP}]
    body = {"model": model, "messages": msgs, "max_tokens": ANSWER_TOKENS, "seed": 0, **SAMPLING,
            "add_generation_prompt": False, "continue_final_message": True}
    out = dict(rec)
    async with sem:
        try:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            d = r.json()
            m = d["choices"][0]["message"]
            # the reasoning parser files continuation text under reasoning (no </think> in the generated part)
            out.update(response=(m.get("content") or "") + (m.get("reasoning_content") or m.get("reasoning") or ""),
                       finish_reason="budget_forced", forced_finish=d["choices"][0]["finish_reason"],
                       forced_usage=d.get("usage"))
        except Exception as e:
            out.update(status="error", error=f"budget forcing failed: {type(e).__name__}: {str(e)[:200]}")
    fout.write(json.dumps(out) + "\n")
    fout.flush()


async def main_async(a):
    url = os.environ["VLLM_URL"].rstrip("/") + "/v1/chat/completions"
    key = os.environ["VLLM_API_KEY"]
    src, dst = Path(a.run), Path(str(a.run) + "_bf")
    dst.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(src / "responses.jsonl")]
    assert len(recs) == 2500, f"{src}: generation incomplete ({len(recs)})"
    done = {json.loads(l)["id"] for l in open(dst / "responses.jsonl")} if (dst / "responses.jsonl").exists() else set()
    judged = {}
    if (src / "judged.jsonl").exists():
        judged = {j["id"]: j for j in map(json.loads, open(src / "judged.jsonl")) if j.get("correct") is not None}
    cfg = json.load(open(src / "config.json"))
    cfg.update(budget_forcing={"early_stop": EARLY_STOP, "answer_tokens": ANSWER_TOKENS,
                               "applies_to": "responses with finish_reason == length"})
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    hle = {r.id: r for r in load_hle().itertuples()}
    todo = []
    with open(dst / "responses.jsonl", "a") as fout, open(dst / "judged.jsonl", "a") as jout:
        for r in recs:
            if r["id"] in done:
                continue
            if r.get("status") == "ok" and r.get("finish_reason") == "length" and r.get("reasoning"):
                todo.append(r)
            else:
                fout.write(json.dumps(r) + "\n")
                if r["id"] in judged:  # unchanged response -> reuse its judgement
                    jout.write(json.dumps(judged[r["id"]]) + "\n")
    print(f"{len(todo)} truncated responses to force", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    with open(dst / "responses.jsonl", "a") as fout:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1800)) as client:
            await asyncio.gather(*[force(client, sem, url, key, a.model, r, hle[r["id"]], fout) for r in todo])
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--concurrency", type=int, default=256)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
