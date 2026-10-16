"""Judge HLE responses with the official HLE judge prompt and structured output, then score.

Judge: OpenAI o3-mini-2025-01-31 (official HLE judge), JSON-schema structured output. Resumable.
Every question gets a judgement: errored / empty generations are scored incorrect (never dropped).
Writes judged.jsonl and summary.json (accuracy overall / text-only / image / by category, calibration error,
run-failure and truncation rates).

    OPENAI_API_KEY=... python src/eval/hle_judge.py --run results/qwen36-base/hle/run1
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from hle_generate import load_hle

JUDGE_MODEL = "o3-mini-2025-01-31"
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""
SCHEMA = {"name": "extracted_answer", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["extracted_final_answer", "reasoning", "correct", "confidence"],
    "properties": {"extracted_final_answer": {"type": "string"}, "reasoning": {"type": "string"},
                   "correct": {"type": "string", "enum": ["yes", "no"]}, "confidence": {"type": "integer"}}}}


async def judge_one(client, sem, rec, q, model, fout):
    out = {"id": rec["id"], "category": rec["category"], "answer_type": rec["answer_type"],
           "has_image": rec["has_image"], "gen_status": rec["status"], "finish_reason": rec.get("finish_reason")}
    response = rec.get("response") or ""
    if rec["status"] != "ok" or not response.strip():
        out.update(correct=False, confidence=100, extracted_final_answer="None", judge="skipped: no response")
    else:
        body = {"model": model, "response_format": {"type": "json_schema", "json_schema": SCHEMA},
                "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                    question=q.question, response=response, correct_answer=q.answer)}]}
        async with sem:
            for attempt in range(5):
                try:
                    r = await client.post("https://api.openai.com/v1/chat/completions", json=body,
                                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
                    if r.status_code == 429 or r.status_code >= 500:
                        await asyncio.sleep(2 ** attempt * 5)
                        continue
                    r.raise_for_status()
                    j = json.loads(r.json()["choices"][0]["message"]["content"])
                    out.update(correct=j["correct"] == "yes", confidence=int(j["confidence"]),
                               extracted_final_answer=j["extracted_final_answer"], judge_reasoning=j["reasoning"],
                               judge=model)
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    await asyncio.sleep(2 ** attempt * 5)
            else:
                out.update(correct=None, judge_error=locals().get("err", "retries exhausted"))
    fout.write(json.dumps(out) + "\n")
    fout.flush()


def calib_err(conf, correct, beta=100):
    """Official HLE RMS calibration error over bins of `beta` sorted examples."""
    idx = np.argsort(conf)
    conf, correct = conf[idx], correct[idx]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(conf) // beta)]
    if bins:
        bins[-1] = [bins[-1][0], len(conf)]
    err = 0.0
    for lo, hi in bins:
        if hi - lo > 0:
            err += (hi - lo) / len(conf) * (np.nanmean(conf[lo:hi]) - np.nanmean(correct[lo:hi])) ** 2
    return float(np.sqrt(err))


def summarize(run: Path):
    rows = [json.loads(l) for l in open(run / "judged.jsonl")]
    n = 2500
    ok = [r for r in rows if r["correct"] is not None]
    acc = lambda rs: round(100 * sum(r["correct"] for r in rs) / max(1, len(rs)), 2)
    by_cat = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r)
    s = {
        "n_total": n, "n_judged": len(ok), "n_judge_errors": len(rows) - len(ok),
        "accuracy": acc(ok) if len(ok) == n else None,
        "accuracy_over_judged": acc(ok),
        "accuracy_text_only": acc([r for r in ok if not r["has_image"]]),
        "n_text_only": sum(not r["has_image"] for r in ok),
        "accuracy_image": acc([r for r in ok if r["has_image"]]),
        "by_category": {c: acc(rs) for c, rs in sorted(by_cat.items())},
        "generation_errors": sum(r["gen_status"] != "ok" for r in rows),
        "truncated_rate": round(sum(r.get("finish_reason") == "length" for r in rows) / max(1, len(rows)), 4),
        "calibration_error": round(100 * calib_err(np.array([r["confidence"] / 100 for r in ok]),
                                                   np.array([float(r["correct"]) for r in ok])), 2),
    }
    json.dump(s, open(run / "summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


async def main_async(a):
    gen = {json.loads(l)["id"]: json.loads(l) for l in open(a.run / "responses.jsonl")}
    hle = {r.id: r for r in load_hle().itertuples()}
    assert len(gen) == len(hle), f"generation incomplete: {len(gen)}/{len(hle)}"
    path = a.run / "judged.jsonl"
    done = {json.loads(l)["id"] for l in open(path) if json.loads(l)["correct"] is not None} if path.exists() else set()
    if path.exists():  # drop earlier judge failures so they are retried
        keep = [l for l in open(path) if json.loads(l)["id"] in done]
        open(path, "w").writelines(keep)
    todo = [gen[i] for i in gen if i not in done]
    print(f"{len(done)} judged, {len(todo)} to go with {a.judge_model}", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    with open(path, "a") as fout:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600)) as client:
            await asyncio.gather(*[judge_one(client, sem, r, hle[r["id"]], a.judge_model, fout) for r in todo])
    summarize(a.run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
