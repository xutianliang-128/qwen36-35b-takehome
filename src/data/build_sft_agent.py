"""Build the E_agent SFT set: TMax terminal trajectories (verifier-successful, Qwen3.6-27B) -> Terminus-2 format.

Source: NiuNiu0110/TMax-Agent-SFT-terminus @7d74fd38a3 (ODC-BY), derived from allenai/tmax-sft
(only_success config). Each trajectory is re-rendered as the exact Terminus-2 (json parser) dialogue used at
evaluation time: official prompt template, JSON {analysis, plan, commands[{keystrokes, duration}], task_complete},
"New Terminal Output:" observations, and the two-step task_complete confirmation.

Per-turn samples (train only the last assistant turn; history turns carry no reasoning) match inference, where
Terminus-2 resends the history without earlier thinking (interleaved_thinking=false). Durations follow the
Terminus-2 prompt's own guidance instead of the base model's habit of waiting the 60 s maximum.

Decontamination: 13-gram overlap of task text against all 89 Terminal-Bench 2.0 instructions, plus
exact-match on task names.

    python src/data/build_sft_agent.py --out data/agent --n-samples 1500
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO, REVISION = "NiuNiu0110/TMax-Agent-SFT-terminus", "7d74fd38a3"
TEMPLATE = Path.home() / ".conda/envs/qwen36/lib/python3.12/site-packages/harbor/agents/terminus_2/templates/terminus-json-plain.txt"
TB2_DIR = Path.home() / "interview/eval_data/terminal-bench-2.0"
PROMPT = "root@sandbox:/home/user# "
CONFIRM = ('Current terminal state:\n{t}\n\nAre you sure you want to mark the task as complete? This will trigger your '
           'solution to be graded and you won\'t be able to make any further corrections. If so, include '
           '"task_complete": true in your JSON response again.')
MAX_OBS_CHARS = 10_000
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SEED = 20261015

QUICK = r"^(cat|ls|echo|cd|pwd|head|tail|grep|wc|mkdir|cp|mv|rm|chmod|chown|touch|sed|awk|cut|sort|uniq|diff|file|stat|which|whoami|env|export|tr|tee|printf|ln|readlink|basename|dirname|test|\[)\b"
SLOW = r"\b(pip3? install|apt(-get)? (install|update)|conda install|npm (install|ci)|yarn|cargo (build|install)|make\b|cmake --build|git clone|wget|curl|docker|mvn|gradle|go (build|get|mod))"
MEDIUM = r"^(gcc|g\+\+|clang|rustc|javac|python3?|node|bash|sh|sqlite3|go run|java|pytest|ruby|perl|Rscript)\b"


def duration_for(cmd: str) -> float:
    c = cmd.strip()
    if re.search(SLOW, c):
        return 30.0 if re.search(r"git clone|wget|curl|apt|cargo|make|mvn|gradle", c) else 15.0
    if re.match(MEDIUM, c) or "&&" in c and re.search(MEDIUM.replace("^", r"(^|&&\s*)"), c):
        return 2.0
    if re.match(QUICK, c):
        return 0.1
    return 1.0


def ngrams(text, n=13):
    w = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def parse_assistant(content: str):
    think = re.search(r"<think>\n?(.*?)</think>", content, re.S)
    reasoning = think.group(1).strip() if think else ""
    rest = content[think.end():] if think else content
    cmd = re.search(r"<parameter=command>\n?(.*?)\n?</parameter>", rest, re.S)
    thought = re.sub(r"<tool_call>.*", "", rest, flags=re.S).strip()
    thought = re.sub(r"^THOUGHT:\s*", "", thought)
    return reasoning, thought, (cmd.group(1).rstrip("\n") if cmd else None)


def plan_text(thought: str, cmd: str | None) -> str:
    first = re.split(r"(?<=[.!?])\s", thought.strip(), maxsplit=1)[0] if thought else ""
    if cmd is None:
        return "The task requirements are satisfied; no further commands are needed."
    return f"Run `{cmd.splitlines()[0][:160]}` and check its output." if not first else first


def to_terminus(msgs) -> tuple[str, list[dict]] | None:
    """Convert one TMax dialogue into Terminus-2 messages (assistant turns carry reasoning_content)."""
    msgs = [dict(m) for m in msgs]
    user0 = next(m["content"] for m in msgs if m["role"] == "user")
    instruction = re.sub(r"^Please solve this task:\s*", "", user0).strip()
    template = TEMPLATE.read_text()
    out = [{"role": "user", "content": template.format(instruction=instruction, terminal_state=PROMPT)}]
    turns = [m for m in msgs if m["role"] in ("assistant", "user")][1:]
    i = 0
    while i < len(turns):
        m = turns[i]
        if m["role"] != "assistant":
            i += 1
            continue
        reasoning, thought, cmd = parse_assistant(m["content"])
        if cmd is not None and cmd.strip() == SUBMIT:
            cmd = None  # upstream (mini-swe-agent style) completion signal -> Terminus-2 task_complete
        if cmd is None:  # final answer: task_complete + Terminus-2 confirmation round
            body = {"analysis": thought or "The task is complete.", "plan": plan_text(thought, None),
                    "commands": [], "task_complete": True}
            out.append({"role": "assistant", "content": json.dumps(body, indent=2), "reasoning_content": reasoning})
            out.append({"role": "user", "content": CONFIRM.format(t=PROMPT)})
            out.append({"role": "assistant", "content": json.dumps(
                {"analysis": "All requirements have been verified.", "plan": "Confirm completion.",
                 "commands": [], "task_complete": True}, indent=2), "reasoning_content": ""})
            break
        body = {"analysis": thought, "plan": plan_text(thought, cmd),
                "commands": [{"keystrokes": cmd + "\n", "duration": duration_for(cmd)}], "task_complete": False}
        out.append({"role": "assistant", "content": json.dumps(body, indent=2), "reasoning_content": reasoning})
        obs = ""
        if i + 1 < len(turns) and turns[i + 1]["role"] == "user":
            obs = re.sub(r"^<tool_response>\n?|\n?</tool_response>$", "", turns[i + 1]["content"].strip())
            obs = re.sub(r"\n?\(exit_code=-?\d+\)\s*$", "", obs)  # upstream harness artifact; Terminus-2 shows no such line
            i += 1
        obs = obs if len(obs) <= MAX_OBS_CHARS else obs[: MAX_OBS_CHARS // 2] + "\n...[output truncated]...\n" + obs[-MAX_OBS_CHARS // 2:]
        out.append({"role": "user", "content": f"New Terminal Output:\n\n{PROMPT}{cmd}\n{obs}\n{PROMPT}".replace("\n\n\n", "\n\n")})
        i += 1
    if not any(m.get("content", "").find('"task_complete": true') >= 0 for m in out if m["role"] == "assistant"):
        return None  # trajectory never finished cleanly
    return instruction, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=1500)
    ap.add_argument("--val-samples", type=int, default=60)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    p = hf_hub_download(REPO, "data/messages/train.parquet", repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(p)
    tb2_text = " ".join((d / "instruction.md").read_text() for d in TB2_DIR.iterdir() if (d / "instruction.md").exists())
    banned = ngrams(tb2_text)
    tb2_names = {d.name for d in TB2_DIR.iterdir()}

    convs, contaminated, failed = [], [], 0
    for r in df.itertuples():
        res = to_terminus(r.messages)
        if res is None:
            failed += 1
            continue
        instruction, msgs = res
        if ngrams(instruction) & banned or r.task_group_id in tb2_names:
            contaminated.append(r.trajectory_id)
            continue
        convs.append({"trajectory_id": r.trajectory_id, "task_group_id": r.task_group_id, "messages": msgs})

    # one trajectory per task group first (diversity), split val by task group
    groups = {}
    for c in convs:
        groups.setdefault(c["task_group_id"], []).append(c)
    gids = sorted(groups)
    rng.shuffle(gids)
    n_val_groups = a.val_samples
    splits = {"val": gids[:n_val_groups], "train": gids[n_val_groups:]}

    def sample_turn(conv):
        """Truncate after a random assistant turn; drop reasoning from history; train only that turn."""
        msgs = conv["messages"]
        idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and m["reasoning_content"]]
        if not idx:
            return None
        t = rng.choice(idx)
        hist = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in msgs[:t]]
        return {"id": f"{conv['trajectory_id']}@{t}", "task_group_id": conv["task_group_id"],
                "messages": hist + [msgs[t]], "turn_index": t}

    stats = {"source": f"{REPO}@{REVISION}", "upstream_rows": len(df), "converted": len(convs),
             "unfinished_dropped": failed, "tb2_contaminated": len(contaminated), "task_groups": len(gids)}
    for name, g in splits.items():
        target = a.val_samples if name == "val" else a.n_samples
        pool = [groups[x] for x in g]
        out = []
        for round_ in range(10):
            for convs_g in pool:
                if len(out) >= target:
                    break
                s = sample_turn(rng.choice(convs_g))
                if s:
                    out.append(s)
            if len(out) >= target:
                break
        with open(a.out / f"sft_{name}.jsonl", "w") as f:
            for s in out:
                f.write(json.dumps(s) + "\n")
        stats[f"sft_{name}"] = len(out)
    durs = [json.loads(m["content"])["commands"][0]["duration"] for c in convs for m in c["messages"]
            if m["role"] == "assistant" and '"keystrokes"' in m["content"]]
    stats["duration_hist"] = {str(d): durs.count(d) for d in sorted(set(durs))}
    json.dump({**stats, "contaminated_ids": contaminated}, open(a.out / "sft_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
