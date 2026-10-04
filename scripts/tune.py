# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["gliner2[local,train]>=2,<3"]
# ///
"""Tune the classifier's effort answer to your own prompts.

    uv run --script scripts/tune.py [how many of your newest prompts, default all]      (or: just tune)

1. Reads every prompt you have sent Claude Code or Codex: Claude Code transcripts in ~/.claude/projects, older
   prompts whose transcripts are gone from ~/.claude/history.jsonl, and your own Codex CLI and desktop threads in
   ~/.codex (no subagents, `codex exec` or orchestrators). Only the kind the router classifies: no slash commands,
   $skills or shell escapes, and no follow-ups under FOLLOW_UP_WORDS.
2. Opus at xhigh effort labels the effort each one needed, 25 per `claude -p` call (no tools, settings or
   saved session; about $0.30 per 100 prompts at API prices). Labels are kept in
   ~/.cache/model-router/tune/labels.jsonl with a reason each, so a rerun only labels new prompts; skim them there.
3. Trains a LoRA adapter for the classifier on 70% of your sessions: on this machine's CPU (about 15 min for 400
   prompts), or over ssh on a machine with an NVIDIA GPU and uv with TUNE_HOST=<ssh host> just tune, which
   scores the held-out prompts there too.
4. On the other 30%, compares the adapter's effort with the router's current one (the score in hooks/route.ts).
   If the adapter is closer to the labels, installs it to ~/.cache/model-router/tuned and restarts the daemon,
   which then answers effort from it. To undo: delete that directory and run `scripts/gliner.sh stop`.

    uv run --script scripts/tune.py --bench                                              (or: just bench)

Benchmark: skips labelling and training and compares the adapter that is already trained (or installed) with hosted
decision models on the same held-out prompts: Jev (TypeSafe, through OpenRouter) with OPENROUTER_KEY, GLiDE (Fastino)
with FASTINO_API_KEY. The first 2000 characters of each held-out prompt (about 30% of them) go to each provider you
give a key for; Jev costs $0.042 per million tokens, so cents.
"""
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))
import classifier  # noqa: E402  the daemon's own loading and inference, so the evaluation runs what it runs

ROUTE = (ROOT / "hooks/route.ts").read_text()
HOME = Path.home()
WORK = HOME / ".cache/model-router/tune"
LABELS = WORK / "labels.jsonl"
LEVELS = ["low", "medium", "high", "xhigh", "max"]
LIMIT = int(sys.argv[1]) if sys.argv[1:2] and sys.argv[1].isdigit() else None
HOST = os.environ.get("TUNE_HOST")
BENCH = "--bench" in sys.argv
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


def records(f, has=""):
    """A JSONL file's records (only lines containing `has`), skipping lines cut off mid-write."""
    with f.open() as lines:
        for line in lines:
            if has in line:
                try:
                    yield json.loads(line)
                except ValueError:
                    pass


def asked(t):
    """False for what nobody asked the agent: slash commands and Codex $skills, shell escapes, injected context,
    task notifications, command output, interruptions."""
    return bool(t) and not t.startswith(("/", "$", "!", "<", "Caveat:", "[Request interrupted"))


def claude():
    """Claude Code transcripts: each prompt with the tool calls it took."""
    for f in (HOME / ".claude/projects").glob("*/*.jsonl"):
        prev, cur = "", None
        for d in records(f):
            m = d.get("message")
            if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary") or not isinstance(m, dict):
                continue
            if d.get("type") == "assistant" and cur:
                cur["tools"] += sum(isinstance(p, dict) and p.get("type") == "tool_use" for p in m.get("content") or [])
            elif d.get("type") == "user" and (t := text_of(m)) is not None:
                t, cur = t.strip(), None
                if asked(t):
                    cur = {"session": f.stem, "text": t, "prev": prev, "tools": 0, "ts": d.get("timestamp", "")}
                    prev = t
                    yield cur


def pasted(d, n):
    """Pasted text that ~/.claude/history.jsonl shows as [Pasted text #n +k lines]: inline, or in the paste cache."""
    p = d.get("pastedContents", {}).get(n) or {}
    cache = HOME / f".claude/paste-cache/{p.get('contentHash')}.txt"
    return p.get("content") or (cache.read_text() if cache.exists() else None)


def claude_history():
    """Prompts from ~/.claude/history.jsonl whose transcripts Claude Code has cleaned up. Tool calls unknown."""
    kept = {f.stem for f in (HOME / ".claude/projects").glob("*/*.jsonl")}
    prev = {}
    for d in records(HOME / ".claude/history.jsonl") if (HOME / ".claude/history.jsonl").exists() else []:
        s = d.get("sessionId") or d.get("project", "")
        t = re.sub(r"\[Pasted text #(\d+)[^\]]*\]", lambda m: pasted(d, m[1]) or m[0], d.get("display", "")).strip()
        if s not in kept and asked(t):
            yield {"session": s, "text": t, "prev": prev.get(s, ""), "tools": None,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(d.get("timestamp", 0) / 1000))}
            prev[s] = t


CODEX_APPS = {"codex-tui", "codex_cli_rs", "Codex Desktop", "codex_vscode"}  # by hand; not exec, SDK or orchestrators
CODEX_TOOLS = {"CommandExecution", "McpToolCall", "FileChange", "WebSearch", "ImageView", "DynamicToolCall",
               "CollabAgentToolCall"}


def codex():
    """Your Codex threads (no subagents): each prompt with the tool calls it took."""
    for f in [*(HOME / ".codex/sessions").rglob("*.jsonl"), *(HOME / ".codex/archived_sessions").glob("*.jsonl")]:
        meta = next(records(f), {}).get("payload", {})
        if meta.get("source") not in ("cli", "vscode") or meta.get("originator") not in CODEX_APPS:
            continue
        prev, cur = "", None
        for d in records(f, '"item_completed"'):  # parse only thread items, not the big tool output lines
            item = d.get("payload", {}).get("item") or {}
            if item.get("type") == "UserMessage":
                t = "\n".join(c.get("text", "") for c in item.get("content", []) if c.get("type") == "text")
                t, cur = t.split("## My request for Codex:")[-1].strip(), None  # the IDE prefixes open files
                if asked(t):
                    cur = {"session": meta.get("session_id") or meta.get("id"), "text": t, "prev": prev, "tools": 0,
                           "ts": d.get("timestamp", "")}
                    prev = t
                    yield cur
            elif cur and item.get("type") in CODEX_TOOLS:
                cur["tools"] += 1


def prompts():
    """Your newest prompts the router would classify, each with the prompt before it and the tool calls it took."""
    seen, found = set(), []
    for r in sorted([*claude(), *claude_history(), *codex()], key=lambda r: r["ts"], reverse=True):
        if len(r["text"].split()) >= FOLLOW_UP_WORDS and r["text"] not in seen:  # resumed sessions repeat their history
            seen.add(r["text"])
            found.append(r)
    return found[:LIMIT]


SYSTEM = """You label prompts a developer sent to an AI coding agent (Claude Code or Codex) working in their \
repository, with the effort level Claude should answer each one at. Effort sets how long Claude thinks and how \
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
the agent made answering it (null when unknown). Judge what the prompt needed, not what happened: the \
tool-call count hints at scope, but routine work can take many calls and a hard question none. Give each a \
one-line reason."""

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


DECIDERS = {  # hosted decision models that take the same questions: name -> (url, model, env var of the key, headers)
    "jev": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13", "OPENROUTER_KEY",
            lambda key: {"Authorization": f"Bearer {key}"}),
    "glide": ("https://api.fastino.ai/v1/systemone", "fastino/GLiDE", "FASTINO_API_KEY", lambda key: {"X-API-Key": key}),
}
DEFS = dict(re.findall(rf"^({'|'.join(LEVELS)}): (.+)$", SYSTEM, re.M))  # the labeller's definition of each level


def decide(name, text):
    """A hosted decision model's answers on one prompt, in one call (the questions run in parallel and in isolation):
    the effort asked as a choice and as a score, and the router's own task and signal questions put through its weights."""
    url, model, env, headers = DECIDERS[name]
    ask = ("What effort should an AI coding agent answer this developer prompt at? Higher effort is slower and "
           "costs more, so pick the lowest level at which a strong model reliably does the job well.")
    questions = {
        "choice": {"type": "choice", "instructions": ask, "criteria": DEFS},
        "score": {"type": "score", "instructions": ask, "criteria": [DEFS[level] for level in LEVELS]},
        "task": {"type": "choice", "instructions": "What kind of work does this prompt ask an AI coding agent for?",
                 "criteria": TASKS},
        **{k: {"type": "noul", "instructions": f"Is this true of the prompt: {d}?"} for k, d in SIGNALS.items()},
    }
    req = urllib.request.Request(
        url, json.dumps({"model": model, "state": text[:2000], "questions": questions}).encode(),
        {"Content-Type": "application/json", **headers(os.environ[env])})
    for tries in (1, 2, 3):
        try:
            t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            break
        except (OSError, ValueError):  # HTTPError is an OSError. After three tries stop: a gap would skew the table
            if tries == 3:
                raise
            time.sleep(tries * 2)
    a = out["answers"]
    found = {"choose": {k: a["task"]["probabilities"].get(k, 0) for k in TASKS}, "flags": {k: a[k]["noul"] for k in SIGNALS}}
    return {"choice": a["choice"]["choice"], "score": LEVELS[round(sum(int(k) * p for k, p in a["score"]["probabilities"].items()))], "router": current(found),
            "ms": (time.perf_counter() - t) * 1000, "cost": out["usage"].get("cost", 0)}


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
    # Batches of 8 as 2 x 4, in bf16 on a GPU (the trainer drops it on CPU): a GPU that runs out of memory
    # spills to system RAM and crawls.
    config = TrainingConfig(output_dir=str(out), num_epochs=6, batch_size=4, gradient_accumulation_steps=2,
                            use_lora=True, lora_r=16, lora_alpha=32,
                            encoder_lr=2e-4, task_lr=2e-4, eval_strategy="no", save_best=False, fp16=False, bf16=True,
                            num_workers=0, pin_memory=False, logging_steps=10)
    GLiNER2Trainer(model=model, config=config).train(train_data=examples)


def answers(adapter, test):
    """For each held-out prompt, the router's effort from the score and the adapter's effort."""
    classifier.load(adapter)
    found = []
    for i, r in enumerate(test, 1):
        found.append(classifier.classify(r["text"][:2000], TASKS, SIGNALS, LEVELS))
        if i % 100 == 0:
            print(f"  scored {i}/{len(test)}")
    return [current(f) for f in found], [max(f["effort"], key=f["effort"].get) for f in found]


REMOTE = ".cache/model-router/repo"  # this checkout's copy on TUNE_HOST, under its home; gliner.sh remote uses it too


def remote(args, send, fetch=()):
    """tune.py <args> on TUNE_HOST over ssh, on its GPU: send {name: local path} into its WORK first, fetch names from
    its WORK after. The prompts sent are deleted there at the end."""
    w = WORK.relative_to(HOME)
    sh = lambda *cmd: subprocess.run(cmd, check=True)  # noqa: E731
    sh("ssh", HOST, f"mkdir -p {w} {REMOTE}")
    sh("rsync", "-a", "--delete", "--exclude", ".git", f"{ROOT}/", f"{HOST}:{REMOTE}/")
    for name, path in send.items():
        sh("rsync", "-a", "--delete", f"{path}/" if path.is_dir() else str(path), f"{HOST}:{w}/{name}")
    try:
        sh("ssh", HOST, f"PATH=$HOME/.local/bin:$PATH uv run --script {REMOTE}/scripts/tune.py {args}")
        for name in fetch:
            sh("rsync", "-a", "--delete", f"{HOST}:{w}/{name}", f"{WORK}/")
    finally:
        sh("ssh", HOST, f"rm -f {w}/fit.json {w}/test.json")


if sys.argv[1:2] == ["--train"]:  # on TUNE_HOST, from remote()
    shutil.rmtree(WORK / "run", ignore_errors=True)
    train(json.loads((WORK / "fit.json").read_text()), WORK / "run")
    sys.exit()
if sys.argv[1:2] == ["--score"]:  # on TUNE_HOST, from remote(): tune.py --score <adapter dirs in WORK>
    test = json.loads((WORK / "test.json").read_text())
    (WORK / "answers.json").write_text(json.dumps({n: answers(WORK / n, test) for n in sys.argv[2:]}))
    sys.exit()

WORK.mkdir(parents=True, exist_ok=True)
rows = prompts()
labels = {d["text"]: d for d in map(json.loads, LABELS.open())} if LABELS.exists() else {}
todo = [] if BENCH else [r for r in rows if r["text"] not in labels]  # a benchmark never spends on labels
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
if BENCH:
    asked = [n for n, d in DECIDERS.items() if d[2] in os.environ]
    if not asked:
        sys.exit("--bench asks hosted models: set OPENROUTER_KEY for jev, FASTINO_API_KEY for glide, or both")
    answered = {}
    for n in asked:  # before the slow local scoring, so a bad key fails fast
        print(f"asking {n} about the {len(test)} held-out prompts")
        with ThreadPoolExecutor(8) as pool:
            answered[n] = list(pool.map(lambda r: decide(n, r["text"]), test))
else:
    shutil.rmtree(run, ignore_errors=True)
    if HOST:
        (WORK / "fit.json").write_text(json.dumps(fit, ensure_ascii=False))
        remote("--train", {"fit.json": WORK / "fit.json"}, ["run"])
    else:
        train(fit, run)


common = Counter(r["effort"] for r in fit).most_common(1)[0][0]
new = run / "final" if (run / "final").exists() else classifier.TUNED  # a benchmark takes the trained adapter as is
if not (new / "adapter_config.json").exists():
    sys.exit("no adapter to benchmark: run just tune first")
adapters = {"tuned": new}
if (classifier.TUNED / "adapter_config.json").exists():  # from an earlier run; replace it only with a better one
    adapters["installed"] = classifier.TUNED
if HOST:
    (WORK / "test.json").write_text(json.dumps(test, ensure_ascii=False))
    remote(f"--score {' '.join(adapters)}", {"test.json": WORK / "test.json", **adapters}, ["answers.json"])
    got = json.loads((WORK / "answers.json").read_text())
else:
    got = {n: answers(path, test) for n, path in adapters.items()}
cols = {"current router": got["tuned"][0], "tuned adapter": got["tuned"][1]}
if "installed" in got:
    cols["installed adapter"] = got["installed"][1]
if BENCH:
    for n, js in answered.items():
        cols |= {f"{n} ({k})": [j[k] for j in js] for k in ("choice", "score", "router")}
cols[f"always {common}"] = [common] * len(test)

off = {n: sum(abs(LEVELS.index(r["effort"]) - LEVELS.index(e)) for r, e in zip(test, c)) / len(test)
       for n, c in cols.items()}
print(f"\nheld out ({len(test)} prompts)   matches label   mean levels off")
for n, c in cols.items():
    print(f"  {n:<18} {sum(r['effort'] == e for r, e in zip(test, c)) / len(test):>13.0%} {off[n]:>17.2f}")

if BENCH:
    for n, js in answered.items():
        ms = sorted(j["ms"] for j in js)
        print(f"\n{n}: median {ms[len(ms) // 2]:.0f} ms a call" + (f", ${sum(j['cost'] for j in js):.4f}" if js[0]["cost"] else ""))
    sys.exit()

beaten = [n for n in ("current router", "installed adapter") if n in off and off[n] <= off["tuned adapter"]]
if beaten:
    sys.exit(f"\nThe adapter is no closer to the labels than the {beaten[0]}: not installed.")
shutil.rmtree(classifier.TUNED, ignore_errors=True)
shutil.copytree(run / "final", classifier.TUNED)
print(f"\ninstalled to {classifier.TUNED}; restarting the daemon")
subprocess.run([ROOT / "scripts/gliner.sh", "stop"])
subprocess.run([ROOT / "scripts/gliner.sh", "start"], check=True)
