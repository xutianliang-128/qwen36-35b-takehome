"""Rebuild every figure from source data (no smoothing beyond what is labelled; raw data is saved alongside).

  curves/<expert>_loss.png      train loss (raw + EMA 0.8, labelled) and held-out val loss vs tokens processed
  curves/merge_comparison.png   dev metrics for merge variants, joint-LoRA control, experts and base
  curves/source/*.csv           the exact numbers plotted

    python scripts/plot_curves.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, SRC = ROOT / "curves", ROOT / "curves" / "source"
EXPERTS = {"e_reason": "E_reason (STEM reasoning)", "e_agent": "E_agent (terminal agent)",
           "e_joint": "Joint LoRA (control: one adapter on both datasets)"}


def ema(xs, a=0.8):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def plot_expert(name, title):
    f = ROOT / "checkpoints" / name / "metrics.jsonl"
    if not f.exists():
        return
    rows = [json.loads(l) for l in open(f)]
    meta = json.load(open(ROOT / "checkpoints" / name / "train_metadata.json"))
    tr = [r for r in rows if "train_loss" in r]
    va = [r for r in rows if "val_loss" in r]
    with open(SRC / f"{name}_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "tokens_seen", "train_loss", "val_loss", "lr", "grad_norm"])
        for r in rows:
            w.writerow([r["step"], r.get("tokens_seen"), r.get("train_loss"), r.get("val_loss"), r.get("lr"), r.get("grad_norm")])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [r["tokens_seen"] / 1e6 for r in tr]
    ax.plot(x, [r["train_loss"] for r in tr], color="#9db4d6", lw=1, label="train loss (raw, per optimizer step)")
    ax.plot(x, ema([r["train_loss"] for r in tr]), color="#2f5b9a", lw=2, label="train loss (EMA 0.8)")
    ax.plot([r["tokens_seen"] / 1e6 for r in va], [r["val_loss"] for r in va], "o-", color="#c0504d", lw=2,
            label="held-out val loss")
    saves = [s for s in range(meta.get("save_every") or 10**9, meta["total_steps"] + 1, meta.get("save_every") or 10**9)]
    for s in saves + [meta["total_steps"]]:
        hit = [r for r in tr if r["step"] == s]
        if hit:
            ax.axvline(hit[0]["tokens_seen"] / 1e6, color="grey", ls=":", lw=1)
            ax.text(hit[0]["tokens_seen"] / 1e6, ax.get_ylim()[1], f" ckpt {s}" if s != meta["total_steps"] else " final",
                    fontsize=7, va="top", color="grey")
    ax.set_xlabel("tokens processed (millions, incl. masked context)")
    ax.set_ylabel("cross-entropy on assistant tokens (nats/token)")
    ax.set_title(f"{title}\nrun_id={name}, LoRA r={meta['lora_rank']}, lr={meta['lr']}, "
                 f"{meta['epochs']} epoch(s), {meta['total_steps']} steps", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_loss.png", dpi=150)
    plt.close(fig)


def plot_merge():
    f = ROOT / "results" / "dev" / "selection.json"
    if not f.exists():
        return
    sel = json.load(open(f))
    loss = json.load(open(ROOT / "results" / "dev" / "val_loss.json"))
    names = [c["name"] for c in sel["candidates"]] + ["expert-e_joint", "expert-e_reason", "expert-e_agent", "qwen36-base"]
    acc = {}
    for n in names:
        g = ROOT / "results" / "dev" / n / "reason_dev" / "gen_stats.json"
        if g.exists():
            d = json.load(open(g))
            acc[n] = 100 * d["correct"] / d["samples"]
    with open(SRC / "merge_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "reason_dev_acc", "agent_val_loss", "reason_val_loss", "selected"])
        for n in names:
            w.writerow([n, acc.get(n), loss.get(n, {}).get("agent_val"), loss.get(n, {}).get("reason_val"), n == sel["selected"]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    present = [n for n in names if n in loss]
    colors = ["#2f5b9a" if n == sel["selected"] else ("#c0504d" if "joint" in n else "#9db4d6") for n in present]
    axes[0].barh(present, [loss[n]["agent_val"] for n in present], color=colors)
    axes[0].set_xlabel("E_agent held-out loss (lower is better)")
    pa = [n for n in present if n in acc]
    axes[1].barh(pa, [acc[n] for n in pa], color=[colors[present.index(n)] for n in pa])
    axes[1].set_xlabel("E_reason dev accuracy % (212 held-out Qs, 8K)")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Merge variants vs joint-LoRA control (dev data only). Selected: {sel['selected']} (dark blue); "
                 "control in red", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "merge_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    SRC.mkdir(parents=True, exist_ok=True)
    for n, t in EXPERTS.items():
        plot_expert(n, t)
    plot_merge()
    print(sorted(p.name for p in OUT.glob("*.png")))
