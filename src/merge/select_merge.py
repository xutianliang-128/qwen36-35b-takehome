"""Pick the merge configuration on dev data only (rule fixed before any merged model was evaluated).

Rule: rank each candidate on (a) E_reason dev accuracy (212 held-out WebInstruct questions, higher is better)
and (b) E_agent held-out SFT loss (60 held-out TMax task groups, lower is better); choose the lowest mean rank,
tie-break on (a). Benchmark data is never used.

    python src/merge/select_merge.py --candidates merged-linear-1.0 merged-ties-d0.2 merged-dare-d0.5
"""

import argparse
import json
from pathlib import Path

DEV = Path("results/dev")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    a = ap.parse_args()
    loss = json.load(open(DEV / "val_loss.json"))
    rows = []
    for c in a.candidates:
        g = json.load(open(DEV / c / "reason_dev" / "gen_stats.json"))
        rows.append({"name": c, "reason_dev_acc": round(100 * g["correct"] / g["samples"], 2),
                     "agent_val_loss": loss[c]["agent_val"], "reason_val_loss": loss[c]["reason_val"]})
    for key, rev in (("reason_dev_acc", True), ("agent_val_loss", False)):
        for r_, row in enumerate(sorted(rows, key=lambda x: x[key], reverse=rev)):
            row[f"rank_{key}"] = r_ + 1
    for row in rows:
        row["mean_rank"] = (row["rank_reason_dev_acc"] + row["rank_agent_val_loss"]) / 2
    best = min(rows, key=lambda x: (x["mean_rank"], -x["reason_dev_acc"]))
    out = {"rule": __doc__.split("\n\n")[1].strip(), "candidates": rows, "selected": best["name"],
           "reference": {k: v for k, v in loss.items() if k not in a.candidates}}
    json.dump(out, open(DEV / "selection.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
