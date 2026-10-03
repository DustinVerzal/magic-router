# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["gliner2[local,train]>=2,<3"]
# ///
"""Tune the classifier's effort answer to your own prompts.

    uv run --script scripts/tune.py [how many prompts, default 500]      (or: just tune)

1. Reads your newest prompts from ~/.claude/projects: the ones the router classifies, so no slash commands
   and no follow-ups under FOLLOW_UP_WORDS.
2. Opus at xhigh effort labels the effort each one needed, 25 per `claude -p` call (no tools, settings or
   saved session; about $0.30 per 100 prompts at API prices). Labels are kept in
   ~/.cache/model-router/tune/labels.jsonl with a reason each, so a rerun only labels new prompts; skim them there.
3. Trains a LoRA adapter for the classifier on 70% of your sessions (CPU; about 15 min for 400 prompts).
4. On the other 30%, compares the adapter's effort with the router's current one (the score in hooks/route.ts).
   If the adapter is closer to the labels, installs it to ~/.cache/model-router/tuned and restarts the daemon,
   which then answers effort from it. To undo: delete that directory and run `scripts/gliner.sh stop`.
"""
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))
import classifier  # noqa: E402  the daemon's own loading and inference, so the evaluation runs what it runs

ROUTE = (ROOT / "hooks/route.ts").read_text()
WORK = Path.home() / ".cache/model-router/tune"
LABELS = WORK / "labels.jsonl"
LEVELS = ["low", "medium", "high", "xhigh", "max"]
LIMIT = int(sys.argv[1]) if sys.argv[1:] else 500
BATCH = 25
sys.stdout.reconfigure(line_buffering=True)  # progress shows up when piped to a log, too


def block(name):
    """An object literal in hooks/route.ts, such as SIGNAL_WEIGHTS, as a dict."""
    body = re.search(rf"export const {name}\b.*?= \{{\n(.*?)\n\}}", ROUTE, re.S)[1]
    return {k: ast.literal_eval(v) for k, v in re.findall(r"^\s*(\w+): (.+?),?(?:\s*//.*)?$", body, re.M)}


TASKS, SIGNALS, WEIGHTS, BIAS = map(block, ["TASKS", "SIGNALS", "SIGNAL_WEIGHTS", "TASK_BIAS"])
EFFORTS = [(float(lo.replace("Infinity", "inf")), e) for lo, e in re.findall(r"\[(-?Infinity|[\d.]+), '(\w+)'\]", ROUTE)]
FOLLOW_UP_WORDS = int(re.search(r"FOLLOW_UP_WORDS = (\d+)", ROUTE)[1])


def current(found):
    """The effort hooks/route.ts gives this classification: score() and effortFor() there."""
    s = sum(found["flags"][k] * w for k, w in WEIGHTS.items()) + sum(found["choose"][k] * b for k, b in BIAS.items())
    return next(e for lo, e in EFFORTS if s >= lo)


def text_of(message):
    """What a person typed; None for tool results."""
    c = message.get("content")
    if isinstance(c, str):
        return c
    if any(p.get("type") == "tool_result" for p in c if isinstance(p, dict)):
        return None
    return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")


def prompts():
    """Your newest prompts the router would classify, each with the prompt before it and the tool calls it took."""
    seen, found = set(), []
    for f in sorted((Path.home() / ".claude/projects").glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        prev, cur = "", None
        for line in f.open():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("message")
            if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary") or not isinstance(m, dict):
                continue
            if d.get("type") == "assistant" and cur:
                cur["tools"] += sum(isinstance(p, dict) and p.get("type") == "tool_use" for p in m.get("content") or [])
            elif d.get("type") == "user" and (t := text_of(m)) is not None:
                t, cur = t.strip(), None
                # Slash commands, task notifications, command output, interruptions: nothing a person asked for.
                if not t or t.startswith(("/", "<", "Caveat:", "[Request interrupted")):
                    continue
                cur = {"session": f.stem, "text": t, "prev": prev, "tools": 0}
                prev = t
                if len(t.split()) >= FOLLOW_UP_WORDS and t not in seen:  # resumed sessions repeat their history
                    seen.add(t)
                    found.append(cur)
        if len(found) >= LIMIT:
            break
    return found[:LIMIT]


SYSTEM = """You label prompts a developer sent to Claude Code, an AI coding agent working in their repository, \
with the effort level Claude should have answered each one at. Effort sets how long Claude thinks and how \
thorough it is. Higher effort is slower and costs more, so the right level is the lowest one at which a strong \
model reliably does the job well.

low: reflexive. A quick factual answer, running a known command, a one-line or mechanical edit, a rename.
medium: routine, with an obvious path. A small fix or feature in one place, a short explanation, a standard \
procedure such as commit, deploy or install.
high: real engineering. Changes across several files, debugging an unknown cause, a feature that needs design \
choices, a careful review or write-up.
xhigh: hard. System design or architecture; subtle correctness, concurrency or security problems; large \
features or refactors; research across a big codebase.
max: the hardest few percent. Deep algorithmic or mathematical reasoning, or intricate high-stakes work where \
even careful thinking is likely to miss something.

Each item has the prompt, the previous prompt in its session (context for follow-ups) and how many tool calls \
Claude made answering it. Judge what the prompt needed, not what happened: the tool-call count hints at scope, \
but routine work can take many calls and a hard question none. Give each a one-line reason."""

SCHEMA = json.dumps({
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "effort": {"enum": LEVELS}, "reason": {"type": "string"}},
        "required": ["id", "effort", "reason"],
    }}},
    "required": ["labels"],
})


def clip(text, n):
    return text if len(text) <= n else f"{text[:n]} [... {len(text) - n} more characters]"


def label(batch):
    """One `claude -p` call: Opus at xhigh labels a batch. Returns ({text: label}, cost in USD)."""
    items = [{"id": i, "prompt": clip(r["text"], 3000), "previous_prompt": clip(r["prev"], 500), "tool_calls": r["tools"]}
             for i, r in enumerate(batch)]
    # No setting sources: no hooks or plugins (this router would reroute the labeller), no MCP servers.
    cmd = ["claude", "-p", "--model", "opus", "--effort", "xhigh", "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--output-format", "json",
           "--system-prompt", SYSTEM, "--json-schema", SCHEMA]
    run = None
    try:
        run = subprocess.run(cmd, input="Label these prompts:\n" + json.dumps(items, ensure_ascii=False),
                             capture_output=True, text=True, cwd=WORK, timeout=900)
        out = json.loads(run.stdout)
        got = out["structured_output"]["labels"]
    except (subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as err:
        detail = (run.stdout or run.stderr)[-300:] if run else err
        print(f"  a labelling call failed, its prompts stay unlabelled: {detail}", file=sys.stderr)
        return {}, 0
    return {batch[g["id"]]["text"]: g for g in got if 0 <= g["id"] < len(batch)}, out.get("total_cost_usd", 0)


def train(rows, out):
    from gliner2 import AutoExtractor
    from gliner2.processor import SamplingConfig
    from gliner2.training import Classification, GLiNER2Trainer, InputExample, TrainingConfig

    model = AutoExtractor.from_pretrained(classifier.MODEL_ID)
    # Pretraining's augmentation (synthetic label names, dropping the true label) would spend half of a few
    # hundred examples on schemas the router never sends.
    model.processor.sampling_config = SamplingConfig(remove_classification_label_prob=0, synthetic_label_prob=0)
    examples = [InputExample(text=r["text"][:2000], classifications=[
        Classification(task="effort", labels=LEVELS, true_label=[r["effort"]], multi_label=True)]) for r in rows]
    # ponytail: CPU only (gliner2's trainer picks CUDA or CPU); fixed epochs, no search, so held-out stays honest.
    config = TrainingConfig(output_dir=str(out), num_epochs=6, batch_size=8, use_lora=True, lora_r=16, lora_alpha=32,
                            encoder_lr=2e-4, task_lr=2e-4, eval_strategy="no", save_best=False, fp16=False,
                            num_workers=0, pin_memory=False, logging_steps=10)
    GLiNER2Trainer(model=model, config=config).train(train_data=examples)


WORK.mkdir(parents=True, exist_ok=True)
rows = prompts()
labels = {d["text"]: d for d in map(json.loads, LABELS.open())} if LABELS.exists() else {}
todo = [r for r in rows if r["text"] not in labels]
print(f"{len(rows)} prompts; labelling {len(todo)} new ones with Opus at xhigh, {BATCH} per call")
cost = 0
with ThreadPoolExecutor(4) as pool, LABELS.open("a") as f:
    for job in as_completed([pool.submit(label, todo[i:i + BATCH]) for i in range(0, len(todo), BATCH)]):
        got, c = job.result()
        cost += c
        for text, g in got.items():
            labels[text] = {"text": text, "effort": g["effort"], "reason": g["reason"]}
            f.write(json.dumps(labels[text], ensure_ascii=False) + "\n")
        f.flush()
        print(f"  {sum(r['text'] in labels for r in rows)}/{len(rows)} labelled, ${cost:.2f}", flush=True)

rows = [{**r, "effort": labels[r["text"]]["effort"]} for r in rows if r["text"] in labels]
if len(rows) < 100:
    sys.exit(f"only {len(rows)} labelled prompts; tuning needs at least 100")
held = lambda r: int(hashlib.sha1(r["session"].encode()).hexdigest(), 16) % 10 < 3  # whole sessions, so no leaks
fit, test = [r for r in rows if not held(r)], [r for r in rows if held(r)]
print(f"labels: {dict(Counter(r['effort'] for r in rows))}")
print(f"training on {len(fit)} prompts, holding out {len(test)} from other sessions")

run = WORK / "run"
shutil.rmtree(run, ignore_errors=True)
train(fit, run)

classifier.load(run / "final")
common = Counter(r["effort"] for r in fit).most_common(1)[0][0]
scored = []
for r in test:
    found = classifier.classify(r["text"][:2000], TASKS, SIGNALS, LEVELS)
    scored.append((r["effort"], current(found), max(found["effort"], key=found["effort"].get), common))

err = lambda col: sum(abs(LEVELS.index(s[0]) - LEVELS.index(s[col])) for s in scored) / len(scored)
print(f"\nheld out ({len(scored)} prompts)   matches label   mean levels off")
for col, name in [(1, "current router"), (2, "tuned adapter"), (3, f"always {common}")]:
    print(f"  {name:<18} {sum(s[0] == s[col] for s in scored) / len(scored):>13.0%} {err(col):>17.2f}")

if err(2) >= err(1):
    sys.exit("\nThe adapter is no closer to the labels than the current router: not installed.")
shutil.rmtree(classifier.TUNED, ignore_errors=True)
shutil.copytree(run / "final", classifier.TUNED)
print(f"\ninstalled to {classifier.TUNED}; restarting the daemon")
subprocess.run([ROOT / "scripts/gliner.sh", "stop"])
subprocess.run([ROOT / "scripts/gliner.sh", "start"], check=True)
