"""Turn rejection-sampled generations into the E_reason SFT set.

1. keep samples that finished normally and whose \\boxed{} answer matches gold
2. decontaminate against HLE (13-gram overlap on normalized text); needs HF access to cais/hle
3. keep the shortest correct sample per question ("think enough, then stop")
4. split off a small validation set (by question) for the loss curve

    HF_TOKEN=... python src/data/build_sft_reason.py --samples data/reason/samples.jsonl --out data/reason
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from answer_check import CHECKER_VERSION, is_correct

HLE_REPO, HLE_REVISION = "cais/hle", "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
N = 13
SEED = 20261015


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def hle_ngrams() -> set:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HLE_REPO, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=HLE_REVISION)
    df = pd.read_parquet(p, columns=["question", "answer"])
    grams = set()
    for q, a in zip(df.question, df.answer):
        grams |= ngrams(f"{q} {a}")
    print(f"HLE: {len(df)} questions, {len(grams)} {N}-grams")
    return grams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples)]
    for r in rows:  # re-grade with the current checker (generation-time flag used checker v1)
        r["correct"] = r["finish_reason"] == "stop" and is_correct(r["pred"], r["answer"], r["answer_type"])
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r)
    stats = {"checker_version": CHECKER_VERSION, "questions": len(by_q), "samples": len(rows),
             "truncated": sum(r["finish_reason"] == "length" for r in rows),
             "correct_samples": sum(r["correct"] for r in rows)}

    banned = hle_ngrams()
    examples, contaminated = [], []
    for qid, samples in by_q.items():
        good = [s for s in samples if s["correct"] and s["reasoning"]]
        if not good:
            continue
        if ngrams(samples[0]["question"]) & banned:
            contaminated.append(qid)
            continue
        best = min(good, key=lambda s: len(s["reasoning"]) + len(s["content"]))
        examples.append({
            "id": qid, "category": best["category"], "difficulty": best["difficulty"], "source": best["source"],
            "n_correct": len(good), "n_samples": len(samples),
            "messages": [
                {"role": "user", "content": best["question"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."},
                {"role": "assistant", "content": best["content"].strip(), "reasoning_content": best["reasoning"].strip()},
            ],
        })
    random.Random(SEED).shuffle(examples)
    n_val = max(20, round(len(examples) * args.val_fraction))
    splits = {"val": examples[:n_val], "train": examples[n_val:]}
    for name, part in splits.items():
        with open(args.out / f"sft_{name}.jsonl", "w") as f:
            for e in part:
                f.write(json.dumps(e) + "\n")
    stats.update(questions_with_correct=len(examples) + len(contaminated), hle_contaminated=len(contaminated),
                 contaminated_ids=contaminated, sft_train=len(splits["train"]), sft_val=len(splits["val"]),
                 pass_at_1=stats["correct_samples"] / max(1, stats["samples"]))
    json.dump(stats, open(args.out / "sft_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "contaminated_ids"}, indent=1))


if __name__ == "__main__":
    main()
