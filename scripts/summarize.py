"""Rebuild the result tables from raw per-task outputs (no hand-entered numbers).

Per (model, benchmark): primary metric (%), n, run-failure counts (environment/agent errors are scored as
failures, never dropped), mean wall time / tokens, and a paired bootstrap 95% CI of the difference vs base on
the identical task set. Also a macro average over the benchmarks every compared model completed.

    python scripts/summarize.py          -> results/summary.csv, analysis/uncertainty.json, analysis/per_task.csv
"""

import csv
import gzip
import datetime as dt
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BASE = "qwen36-base"
B = 10_000
SEED = 0


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def hle(model):
    """Primary: thinking-budget-forced run (run1_bf); plain truncated run reported alongside."""
    out = {}
    for run, label in (("run1_bf", "hle"), ("run1", "hle_plain_truncated")):
        d = RES / model / "hle" / run
        if not (d / "summary.json").exists():
            continue
        rf = d / "responses.jsonl"  # committed as responses.jsonl.gz
        lines = open(rf) if rf.exists() else gzip.open(str(rf) + ".gz", "rt")
        resp = {json.loads(l)["id"]: json.loads(l) for l in lines}
        per = {}
        for l in open(d / "judged.jsonl"):
            j = json.loads(l)
            r = resp.get(j["id"], {})
            u = r.get("usage") or {}
            fu = r.get("forced_usage") or {}
            per[j["id"]] = {"score": float(bool(j.get("correct"))), "run_failure": r.get("status") != "ok",
                            "tokens": (u.get("completion_tokens") or 0) + (fu.get("completion_tokens") or 0),
                            "seconds": r.get("latency_s"), "has_image": r.get("has_image"),
                            "truncated": r.get("finish_reason") in ("length", "budget_forced")}
        out[label] = per
    return out


def harbor_job(job_dir):
    per = {}
    for f in sorted(Path(job_dir).glob("*/result.json")):
        d = json.load(open(f))
        # trial dir names are truncated (SWE-Pro ids collide); use the full task name recorded by Harbor
        tid = (d.get("task_name") or f.parent.name.rsplit("__", 1)[0]).split("/")[-1]
        rew = (d.get("verifier_result") or {}).get("rewards") or {}
        exc = (d.get("exception_info") or {}).get("exception_type")
        a = d.get("agent_result") or {}
        secs = (ts(d["finished_at"]) - ts(d["started_at"])).total_seconds() if d.get("finished_at") else None
        per[tid] = {"score": float(rew.get("reward", 0.0) or 0.0), "exception": exc,
                    "run_failure": exc is not None and exc != "AgentTimeoutError",
                    "timeout": exc == "AgentTimeoutError", "tokens": a.get("n_output_tokens"), "seconds": secs}
    return per


def finished(job_dir):
    f = Path(job_dir) / "result.json"
    return f.exists() and bool(json.load(open(f)).get("finished_at"))  # partial (running) jobs are not reported


def tb2(model):
    j = RES / model / "terminal-bench" / "tb2-run1" / "tb2-run1"
    return {"terminal_bench": harbor_job(j)} if finished(j) else {}


def swepro(model):
    j = RES / model / "swebench-pro" / "swepro-run1" / "swepro-run1-regrade" / "swepro-run1-regrade"
    if not finished(j):
        return {}
    graded = harbor_job(j)  # authoritative: patch replay on pristine images
    agent = harbor_job(RES / model / "swebench-pro" / "swepro-run1" / "swepro-run1")
    for k, v in graded.items():  # time/tokens/failures come from the agent run
        a = agent.get(k, {})
        v.update(tokens=a.get("tokens"), seconds=a.get("seconds"),
                 run_failure=a.get("run_failure", v["run_failure"]), timeout=a.get("timeout", False))
    return {"swebench_pro_hard51": graded}


def asi(model):
    """ASI-Bench seed31415 B4, local public scoring (non-official). Instances whose scorer could not run
    (evaluation_status != completed, e.g. Docker-only graders) are excluded identically for every model."""
    f = RES / model / "asi-bench" / "asi-b4-run1" / "local_score_seed31415.json"
    if not f.exists():
        return {}
    per = {}
    for r in json.load(open(f))["results"]:
        if r.get("evaluation_status") != "completed":
            continue
        per[r["instance_id"]] = {"score": (r.get("final_score") or 0.0) / (r.get("max_score") or 100.0),
                                 "run_failure": False}
    return {"asi_bench": per}


def paired_ci(a, b, rng):
    ids = sorted(set(a) & set(b))
    diffs = [a[i]["score"] - b[i]["score"] for i in ids]
    n = len(diffs)
    if n == 0:
        return None
    est = 100 * sum(diffs) / n
    boots = sorted(100 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {"n_paired": n, "diff_pp": round(est, 2), "ci95": [round(boots[int(0.025 * B)], 2), round(boots[int(0.975 * B)], 2)],
            "p_le_0": round(sum(x <= 0 for x in boots) / B, 4)}


def main():
    rng = random.Random(SEED)
    models = sorted(p.name for p in RES.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "dev")
    data = {m: {**hle(m), **tb2(m), **swepro(m), **asi(m)} for m in models}
    rows, unc, per_task = [], {}, []
    for m in models:
        for bench, per in data[m].items():
            n = len(per)
            secs = [v["seconds"] for v in per.values() if v.get("seconds")]
            toks = [v["tokens"] for v in per.values() if v.get("tokens")]
            row = {"model": m, "benchmark": bench, "score_pct": round(100 * sum(v["score"] for v in per.values()) / n, 2),
                   "n_tasks": n, "run_failures": sum(bool(v.get("run_failure")) for v in per.values()),
                   "timeouts": sum(bool(v.get("timeout")) for v in per.values()),
                   "mean_seconds": round(sum(secs) / len(secs), 1) if secs else None,
                   "mean_output_tokens": round(sum(toks) / len(toks)) if toks else None}
            if bench.startswith("hle"):
                row["truncated_or_forced"] = sum(bool(v.get("truncated")) for v in per.values())
            if m != BASE and bench in data.get(BASE, {}):
                ci = paired_ci(per, data[BASE][bench], rng)
                unc[f"{m}|{bench}"] = ci
                row.update(diff_vs_base_pp=ci["diff_pp"], ci95_low=ci["ci95"][0], ci95_high=ci["ci95"][1])
            rows.append(row)
            for tid, v in per.items():
                per_task.append({"model": m, "benchmark": bench, "task_id": tid, "score": v["score"],
                                 "run_failure": v.get("run_failure"), "exception": v.get("exception")})
    # macro average over the primary benchmarks each model has (missing ones listed, never imputed)
    primary = ["hle", "terminal_bench", "swebench_pro_hard51", "asi_bench"]
    macro = {}
    for m in sorted(models, key=lambda x: x != BASE):  # base first: others are differenced against it
        have = [b for b in primary if b in data[m]]
        macro[m] = {"benchmarks": have, "missing": [b for b in primary if b not in data[m]],
                    "macro_avg_pct": round(sum(next(r["score_pct"] for r in rows if r["model"] == m and r["benchmark"] == b)
                                               for b in have) / len(have), 2) if have else None}
        if m != BASE and have and all(b in data.get(BASE, {}) for b in have):
            boots = []
            for _ in range(2000):  # resample tasks independently within each benchmark
                ds = []
                for b in have:
                    ids = sorted(set(data[m][b]) & set(data[BASE][b]))
                    smp = [ids[rng.randrange(len(ids))] for _ in ids]
                    ds.append(100 * sum(data[m][b][i]["score"] - data[BASE][b][i]["score"] for i in smp) / len(smp))
                boots.append(sum(ds) / len(ds))
            boots.sort()
            # point estimate = mean of per-benchmark paired differences (same task sets as the CI)
            macro[m]["macro_diff_vs_base_pp"] = round(sum(unc[f"{m}|{b}"]["diff_pp"] for b in have) / len(have), 2)
            macro[m]["macro_diff_ci95"] = [round(boots[50], 2), round(boots[1949], 2)]
    (ROOT / "analysis").mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "analysis" / "per_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task[0]))
        w.writeheader()
        w.writerows(per_task)
    json.dump({"paired_bootstrap_vs_base": unc, "macro": macro, "bootstrap_resamples": B, "seed": SEED},
              open(ROOT / "analysis" / "uncertainty.json", "w"), indent=1)
    for r in rows:
        print({k: r[k] for k in ("model", "benchmark", "score_pct", "n_tasks", "run_failures") if k in r},
              {k: r[k] for k in ("diff_vs_base_pp", "ci95_low", "ci95_high") if k in r})
    print(json.dumps(macro, indent=1))


if __name__ == "__main__":
    main()
