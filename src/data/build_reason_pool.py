"""Select the E_reason problem pool from WebInstruct-verified (Apache-2.0).

Keeps university/PhD STEM problems whose gold answer can be checked by a program
(numbers, fractions, percentages, multiple choice, booleans, math expressions), stratified
toward the subject mix of HLE. Splits off a held-out dev set that is never trained on.

    python src/data/build_reason_pool.py --out data/reason
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO, REVISION = "TIGER-Lab/WebInstruct-verified", "3e8a350b3a"
CHECKABLE = {"Float", "Integer", "Fraction", "Percentage", "Multiple Choice", "Boolean", "Expression"}
# (category, difficulty levels, n_train) — math/physics weighted like HLE's subject mix.
QUOTA = [
    ("Mathematics", ["PhD"], 250),
    ("Mathematics", ["University"], 250),
    ("Physics", ["University", "PhD"], 250),
    ("Chemistry", ["University", "PhD"], 150),
    ("Biology", ["University", "PhD", "Senior High School"], 100),
    ("Computer Science", ["University", "PhD"], 80),
    ("Engineering", ["University", "PhD"], 60),
    ("Other STEM", ["University", "PhD", "Senior High School"], 60),
]
DEV_FRACTION = 0.15
SEED = 20261015


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(REPO, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(path)
    df = df[df.answer_type.isin(CHECKABLE)]
    df = df[df.question.str.len().between(40, 4000)]
    df = df.drop_duplicates(subset="question")

    rng = random.Random(SEED)
    rows = []
    for cat, levels, n in QUOTA:
        pool = df[(df.category == cat) & df.difficulty.isin(levels)]
        n_total = min(len(pool), round(n / (1 - DEV_FRACTION)))
        idx = rng.sample(sorted(pool.index), n_total)
        rows.append(pool.loc[idx])
    sel = pd.concat(rows).sample(frac=1.0, random_state=SEED)

    n_dev = round(len(sel) * DEV_FRACTION)
    splits = {"dev": sel.iloc[:n_dev], "train": sel.iloc[n_dev:]}
    for name, part in splits.items():
        with open(args.out / f"pool_{name}.jsonl", "w") as f:
            for r in part.itertuples():
                f.write(json.dumps({"id": f"wiv-{r.id}", "question": r.question, "answer": r.answer,
                                    "answer_type": r.answer_type, "category": r.category,
                                    "difficulty": r.difficulty, "source": f"{REPO}@{REVISION}"}) + "\n")
        print(name, len(part), part.category.value_counts().to_dict())


if __name__ == "__main__":
    main()
