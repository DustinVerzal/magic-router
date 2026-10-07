# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["gliner2[local,train]>=2,<3"]
# ///
"""Tune the classifier's model and effort answers to your own prompts.

    uv run --script scripts/tune.py [how many of your newest prompts, default all]      (or: just tune)

1. Reads every prompt you have sent Claude Code or Codex: Claude Code transcripts in ~/.claude/projects, older
   prompts whose transcripts are gone from ~/.claude/history.jsonl, and your own Codex CLI and desktop threads in
   ~/.codex (no subagents, `codex exec` or orchestrators). Only the kind the router classifies: no slash commands,
   $skills or shell escapes, and no follow-ups under FOLLOW_UP_WORDS (a session's first prompt stays however short,
   since it picks the model).
2. Opus at xhigh effort labels the model (Sonnet or Opus) and effort each one needed, given the Artificial Analysis
   scores for each model at each effort (`just benchmarks` caches them), 25 per `claude -p` call (no tools, settings or
   saved session; about $0.30 per 100 prompts at API prices). Labels are kept in
   ~/.cache/magic-router/tune/labels.jsonl with a reason each, so a rerun only labels new prompts (and those
   labelled under an older labeller prompt); skim them there.
3. Trains a LoRA adapter for the classifier on 70% of your sessions, whole, so related prompts never sit on both sides
   of the split: effort on every prompt, and the model on each session's first prompt only, labelled with what the
   whole session needed (Opus if any of its prompts did), because the router picks the model there and keeps it. On
   this machine's CPU (about 15 min for 400 prompts), or over ssh on a machine with an NVIDIA GPU and uv with
   TUNE_HOST=<ssh host> just tune, which scores the held-out prompts there too.
4. On the other 30%, compares the adapter's effort (on every prompt) and model (on each session's first prompt) with
   the router's current ones (the score in hooks/route.ts). If the adapter is closer on effort and no worse on model,
   installs it to ~/.cache/magic-router/tuned and restarts the daemon, which then answers both from it. To undo:
   delete that directory and run `scripts/gliner.sh stop`.

    uv run --script scripts/tune.py --bench [clef]                                       (or: just bench [clef])

Benchmark: skips labelling and training and compares the adapter that is already trained (or installed) with other
decision models on the same held-out prompts, on effort and on the model: Jev (TypeSafe, through OpenRouter) with
OPENROUTER_KEY, GLiDE (Fastino) with FASTINO_API_KEY, and with `clef`, Clef-flash (Cloudflare's 9B, open weights) on
the GPU, through scripts/clef.py, on TUNE_HOST if you set one. The first 2000 characters of each held-out prompt (about
30% of them) go to each provider you give a key for; Jev costs $0.042 per million tokens, so cents. Clef reads whole
prompts (up to about 32,000 characters) and keeps them on your machines.
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
WORK = HOME / ".cache/magic-router/tune"
LABELS = WORK / "labels.jsonl"
LEVELS = ["low", "medium", "high", "xhigh", "max"]
LIMIT = int(sys.argv[1]) if sys.argv[1:2] and sys.argv[1].isdigit() else None
HOST = os.environ.get("TUNE_HOST")
BENCH = "--bench" in sys.argv
CLEF = BENCH and "clef" in sys.argv  # tune.py --bench clef: Clef-flash on the GPU (TUNE_HOST's, if set) too
BATCH = 25
sys.stdout.reconfigure(line_buffering=True)  # progress shows up when piped to a log, too


def block(name):
    """An object literal in hooks/route.ts, such as SIGNAL_WEIGHTS, as a dict."""
    body = re.search(rf"export const {name}\b.*?= \{{\n(.*?)\n\}}", ROUTE, re.S)[1]
    return {k: ast.literal_eval(v) for k, v in re.findall(r"^\s*(\w+): (.+?),?(?:\s*//.*)?$", body, re.M)}


TASKS, SIGNALS, WEIGHTS, BIAS = map(block, ["TASKS", "SIGNALS", "SIGNAL_WEIGHTS", "TASK_BIAS"])
EFFORTS = [(float(lo.replace("Infinity", "inf")), e) for lo, e in re.findall(r"\[(-?Infinity|[\d.]+), '(\w+)'\]", ROUTE)]
OPUS_AT = float(re.search(r"OPUS_AT = (-?[\d.]+)", ROUTE)[1])
LABELLER = 2  # bump when the labeller prompt changes meaning: older labels are redone
PICKS = ["sonnet", "opus"]  # what the adapter chooses between; Fable stays the score's (FABLE_AT) once it is on
FOLLOW_UP_WORDS = int(re.search(r"FOLLOW_UP_WORDS = (\d+)", ROUTE)[1])


def current(found):
    """The effort and model hooks/route.ts gives this classification: score(), effortFor() and modelFor() there."""
    s = sum(found["flags"][k] * w for k, w in WEIGHTS.items()) + sum(found["choose"][k] * b for k, b in BIAS.items())
    return next(e for lo, e in EFFORTS if s >= lo), "opus" if s >= OPUS_AT else "sonnet"


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
                    cur = {"session": f.stem, "text": t, "prev": prev, "first": not prev, "tools": 0,
                           "ts": d.get("timestamp", "")}
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
            yield {"session": s, "text": t, "prev": prev.get(s, ""), "first": s not in prev, "tools": None,
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
                    cur = {"session": meta.get("session_id") or meta.get("id"), "text": t, "prev": prev,
                           "first": not prev, "tools": 0, "ts": d.get("timestamp", "")}
                    prev = t
                    yield cur
            elif cur and item.get("type") in CODEX_TOOLS:
                cur["tools"] += 1


def prompts():
    """Your newest prompts the router would classify, each with the prompt before it and the tool calls it took.
    A session's first prompt is kept however short: the router classifies it, and it picks the session's model."""
    seen, found = set(), []
    for r in sorted([*claude(), *claude_history(), *codex()], key=lambda r: r["ts"], reverse=True):
        # Resumed sessions repeat their history, so a prompt is kept once; an opener once per session, so that two
        # sessions both opening on "fix the typo" each keep the prompt that picked their model.
        key = (r["session"], r["text"]) if r["first"] else r["text"]
        if (r["first"] or len(r["text"].split()) >= FOLLOW_UP_WORDS) and key not in seen:
            seen.add(key)
            found.append(r)
    return found[:LIMIT]


AA = HOME / ".cache/magic-router/benchmarks.json"  # scripts/benchmarks.py's cache of the Artificial Analysis API
AA_EVALS = {"artificial_analysis_intelligence_index": "AA index", "hle": "reasoning (HLE)",
            "terminalbench_v4_0": "agentic coding (Terminal-Bench)", "scicode": "scientific coding (SciCode)",
            "lcr": "long context (AA-LCR)"}


def aa_table():
    """The Sonnet and Opus versions the router picks between, at every effort level, one line each from AA's scores and speeds."""
    if not AA.exists():
        return ""
    ours = {n: f"{a}.{b}" for n, a, b in re.findall(r"'claude-(sonnet|opus)-(\d+)-(\d+)'", ROUTE)}
    lines = []
    for m in json.loads(AA.read_text())["data"]:
        got = re.match(r"Claude (Sonnet|Opus) ([\d.]+) \((\w+)", m["name"])
        if got and ours.get(got[1].lower()) == got[2] and got[3].lower() in LEVELS:
            ev, price = m.get("evaluations", {}), m.get("pricing", {})
            scores = ", ".join(f"{AA_EVALS[k]} {ev[k]:.3g}" for k in AA_EVALS if k in ev)
            lines.append((got[1].lower(), LEVELS.index(got[3].lower()), f"{got[1].lower()} at {got[3].lower()}: {scores}"
                          f"; first answer token after {m['median_time_to_first_answer_token']:.0f}s, "
                          f"then {m['median_output_tokens_per_second']:.0f} tokens/s"))
    return "\n\nArtificial Analysis scores (the AA index is 0-100, the evals 0-1) and speeds:\n" + "\n".join(l for *_, l in sorted(lines))


SYSTEM = """You label prompts a developer sent to an AI coding agent (Claude Code or Codex) working in their \
repository, with the model and the effort level Claude should answer each one at. Effort sets how long Claude \
thinks and how thorough it is. The developer cares about the quality of the answer and about not waiting on \
overthinking, not about cost: pick the pair with the best expected answer quality that does not make them wait \
for thinking the prompt did not need. Higher effort and Opus are slower (see the time to first answer token \
below), so spend the wait only where it buys a better answer.

Model: sonnet or opus. Use opus where the AA scores below show it ahead of Sonnet on the kind of work the \
prompt is, or where the prompt rests on judgment, design or hard reasoning. Use sonnet where the two are level, \
or where Opus's lead is not worth its longer wait. A high effort on Sonnet can beat a lower one on Opus. {aa}

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

SYSTEM = SYSTEM.replace("{aa}", aa_table())

SCHEMA = json.dumps({
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "model": {"enum": PICKS}, "effort": {"enum": LEVELS},
                       "reason": {"type": "string"}},
        "required": ["id", "model", "effort", "reason"],
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
ASK = ("What effort should an AI coding agent answer this developer prompt at? Higher effort is slower and "
       "costs more, so pick the lowest level at which a strong model reliably does the job well.")
# What a decision model is asked of each prompt: the effort as a choice and as a score, the router's own task and
# signal questions (put through its weights), and the model, as the labeller was asked it.
QUESTIONS = {
    "choice": {"type": "choice", "instructions": ASK, "criteria": DEFS},
    "score": {"type": "score", "instructions": ASK, "criteria": [DEFS[level] for level in LEVELS]},
    "task": {"type": "choice", "instructions": "What kind of work does this prompt ask an AI coding agent for?",
             "criteria": TASKS},
    **{k: {"type": "noul", "instructions": f"Is this true of the prompt: {d}?"} for k, d in SIGNALS.items()},
    "model": {"type": "choice", "criteria": {
        "sonnet": "Claude Sonnet: faster, and level with Opus on routine edits, tool use, questions and write-ups",
        "opus": "Claude Opus: slower, and ahead on hard reasoning, agentic coding, and work that rests on judgment or design"},
        "instructions": "Which model should an AI coding session that opens with this prompt run on? It keeps that "
                        "model for every later prompt, so pick the best expected answers without waits they do not need."},
}


def decision(a, ms, cost=0):
    """A decision model's answers to QUESTIONS, as the effort three ways and the model."""
    found = {"choose": {k: a["task"]["probabilities"].get(k, 0) for k in TASKS}, "flags": {k: a[k]["noul"] for k in SIGNALS}}
    return {"choice": a["choice"]["choice"], "score": LEVELS[round(sum(int(k) * p for k, p in a["score"]["probabilities"].items()))],
            "router": current(found)[0], "model": a["model"]["choice"], "ms": ms, "cost": cost}


def decide(name, text):
    """A hosted decision model's answers on one prompt, in one call (the questions run in parallel and in isolation)."""
    url, model, env, headers = DECIDERS[name]
    req = urllib.request.Request(
        url, json.dumps({"model": model, "state": text[:2000], "questions": QUESTIONS}).encode(),
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
    return decision(out["answers"], (time.perf_counter() - t) * 1000, out["usage"].get("cost", 0))


def clef(test):
    """Clef-flash's answers on the held-out prompts, from scripts/clef.py on this machine's GPU. Its own environment
    (gliner2 pins an older transformers) and process, so its VRAM is free again before the adapters load. It reads
    whole prompts, where GLiNER gets 2000 characters (its DeBERTa encoder was trained on 512 tokens): scripts/clef.py
    cuts each to 8,192 tokens, about 32,000 characters, at a median 0.5 s a prompt (1,025 took 22 minutes)."""
    job, out = WORK / "clef.in.json", WORK / "clef.json"
    gliner = lambda cmd: subprocess.run([ROOT / "scripts/gliner.sh", cmd], capture_output=True).returncode == 0  # noqa: E731
    # Clef's 11 GB, a long prompt's activations and the live daemon outgrow a 16 GB card, which then spills to system
    # RAM and crawls; so the daemon stops for the run (sessions keep their last route meanwhile) and comes back after.
    # ponytail: a Claude Code session started on this machine mid-run brings the daemon back; hold off until it ends.
    was_up = gliner("status") and gliner("stop")
    try:
        job.write_text(json.dumps({"questions": QUESTIONS, "texts": [r["text"] for r in test]}, ensure_ascii=False))
        subprocess.run(["uv", "run", "--script", str(ROOT / "scripts/clef.py"), str(job), str(out)], check=True)
    finally:
        job.unlink(missing_ok=True)  # the prompts
        if was_up:
            gliner("start")
    return [decision(o["answers"], o["ms"]) for o in json.loads(out.read_text())]


def train(rows, out):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on a 16 GB card
    import torch
    from gliner2 import AutoExtractor
    from gliner2.processor import SamplingConfig
    from gliner2.training import Classification, GLiNER2Trainer, InputExample, TrainingConfig

    model = AutoExtractor.from_pretrained(classifier.MODEL_ID)
    # Pretraining's augmentation (synthetic label names, dropping the true label) would spend half of a few
    # hundred examples on schemas the router never sends.
    model.processor.sampling_config = SamplingConfig(remove_classification_label_prob=0, synthetic_label_prob=0)
    # The model is asked only of a session's first prompt, so only first prompts teach it.
    examples = [InputExample(text=r["text"][:2000], classifications=[
        Classification(task="effort", labels=LEVELS, true_label=[r["effort"]], multi_label=True),
        *[Classification(task="model", labels=PICKS, true_label=[r["model"]], multi_label=True)] * r["first"]])
        for r in rows]
    # ponytail: CPU only (gliner2's trainer picks CUDA or CPU); fixed epochs, no search, so held-out stays honest.
    # Batches of 8 as 4 x 2, in bf16 on a GPU only. 2 x 4 ran out of memory on a 16 GB card once each prompt had
    # two tasks, and a GPU that runs out of memory spills to system RAM and crawls.
    config = TrainingConfig(output_dir=str(out), num_epochs=6, batch_size=2, gradient_accumulation_steps=4,
                            use_lora=True, lora_r=16, lora_alpha=32,
                            encoder_lr=2e-4, task_lr=2e-4, eval_strategy="no", save_best=False, fp16=False, bf16=torch.cuda.is_available(),
                            num_workers=0, pin_memory=False, logging_steps=10)
    GLiNER2Trainer(model=model, config=config).train(train_data=examples)


def answers(adapter, test):
    """For each held-out prompt, the router's effort and model from the score, then the adapter's effort and model."""
    classifier.load(adapter)
    found = []
    for i, r in enumerate(test, 1):
        found.append(classifier.classify(r["text"][:2000], TASKS, SIGNALS, LEVELS, PICKS))
        if i % 100 == 0:
            print(f"  scored {i}/{len(test)}")
    top = lambda f, k: max(f[k], key=f[k].get)  # noqa: E731
    return ([current(f)[0] for f in found], [top(f, "effort") for f in found],
            [current(f)[1] for f in found], [top(f, "pick") if "pick" in f else None for f in found])  # an older adapter has none


REMOTE = ".cache/magic-router/repo"  # this checkout's copy on TUNE_HOST, under its home; gliner.sh remote uses it too


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
        sh("ssh", HOST, f"CLEF_BITS={int(os.environ.get('CLEF_BITS', 8))} PATH=$HOME/.local/bin:$PATH "
                        f"uv run --script {REMOTE}/scripts/tune.py {args}")
        for name in fetch:
            sh("rsync", "-a", "--delete", f"{HOST}:{w}/{name}", f"{WORK}/")
    finally:
        sh("ssh", HOST, f"rm -f {w}/fit.json {w}/test.json {w}/clef.in.json")


if sys.argv[1:2] == ["--train"]:  # on TUNE_HOST, from remote()
    shutil.rmtree(WORK / "run", ignore_errors=True)
    train(json.loads((WORK / "fit.json").read_text()), WORK / "run")
    sys.exit()
if sys.argv[1:2] == ["--score"]:  # on TUNE_HOST, from remote(): tune.py --score <adapter dirs in WORK> [clef]
    test = json.loads((WORK / "test.json").read_text())
    got = {"clef": clef(test)} if "clef" in sys.argv[2:] else {}  # first, while the GPU is free
    got |= {n: answers(WORK / n, test) for n in sys.argv[2:] if n != "clef"}
    (WORK / "answers.json").write_text(json.dumps(got))
    sys.exit()

WORK.mkdir(parents=True, exist_ok=True)
rows = prompts()
labels = {d["text"]: d for d in map(json.loads, LABELS.open())} if LABELS.exists() else {}
# A benchmark never spends on labels; labels from an older labeller prompt are redone.
todo = [] if BENCH else [r for r in rows if labels.get(r["text"], {}).get("v") != LABELLER]
if todo and not AA.exists():
    sys.exit("the labeller needs the Artificial Analysis scores: run `just benchmarks` first (it needs AA_API_KEY)")
print(f"{len(rows)} prompts; labelling {len(todo)} new ones with Opus at xhigh, {BATCH} per call")
cost = 0
with ThreadPoolExecutor(4) as pool, LABELS.open("a") as f:
    for job in as_completed([pool.submit(label, todo[i:i + BATCH]) for i in range(0, len(todo), BATCH)]):
        got, c = job.result()
        cost += c
        for text, g in got.items():
            labels[text] = {"text": text, "v": LABELLER, "model": g["model"], "effort": g["effort"], "reason": g["reason"]}
            f.write(json.dumps(labels[text], ensure_ascii=False) + "\n")
        f.flush()
        print(f"  {sum(r['text'] in labels for r in rows)}/{len(rows)} labelled, ${cost:.2f}", flush=True)

rows = [{**r, **{k: labels[r["text"]][k] for k in ("effort", "model")}} for r in rows if labels.get(r["text"], {}).get("v") == LABELLER or (BENCH and "model" in labels.get(r["text"], {}))]
if len(rows) < 100:
    sys.exit(f"only {len(rows)} labelled prompts; tuning needs at least 100")
# The model is picked on a session's first prompt and kept for all of it, so it has to serve the session's hardest
# prompt: one that opens on a typo fix and turns into design work needed Opus from the start.
# ponytail: any Opus prompt makes the session Opus; a share threshold if that sends long, mostly routine sessions there.
opus = {r["session"] for r in rows if r["model"] == "opus"}
rows = [{**r, "model": "opus" if r["session"] in opus else "sonnet"} for r in rows]
held = lambda r: int(hashlib.sha1(r["session"].encode()).hexdigest(), 16) % 10 < 3  # whole sessions, so no leaks
fit, test = [r for r in rows if not held(r)], [r for r in rows if held(r)]
print(f"labels: {dict(Counter(r['effort'] for r in rows))}, sessions {dict(Counter(r['model'] for r in rows if r['first']))}")
print(f"training on {len(fit)} prompts, holding out {len(test)} from other sessions")

run = WORK / "run"
if BENCH:
    asked = [n for n, d in DECIDERS.items() if d[2] in os.environ]
    if not asked and not CLEF:
        sys.exit("--bench compares other decision models: add `clef` for Clef-flash on your GPU, or set "
                 "OPENROUTER_KEY for jev, FASTINO_API_KEY for glide")
    answered = {}
    for n in asked:  # before the slow local scoring, so a bad key fails fast
        print(f"asking {n} about the {len(test)} held-out prompts")
        with ThreadPoolExecutor(8) as pool:
            answered[n] = list(pool.map(lambda r: decide(n, r["text"]), test))
else:
    # The last run's adapter is kept as run.prev, so two runs can still be compared after the second one trains.
    shutil.rmtree(WORK / "run.prev", ignore_errors=True)
    if run.exists():
        run.rename(WORK / "run.prev")
    if HOST:
        (WORK / "fit.json").write_text(json.dumps(fit, ensure_ascii=False))
        remote("--train", {"fit.json": WORK / "fit.json"}, ["run"])
    else:
        train(fit, run)
    (run / "final/tasks.json").write_text('["effort", "model"]')  # tells the daemon this adapter's model answer is trained


common = Counter(r["effort"] for r in fit).most_common(1)[0][0]
new = run / "final" if (run / "final").exists() else classifier.TUNED  # a benchmark takes the trained adapter as is
if not (new / "adapter_config.json").exists():
    sys.exit("no adapter to benchmark: run just tune first")
adapters = {"tuned": new}
if (classifier.TUNED / "adapter_config.json").exists():  # from an earlier run; replace it only with a better one
    adapters["installed"] = classifier.TUNED
if HOST:
    (WORK / "test.json").write_text(json.dumps(test, ensure_ascii=False))
    remote(f"--score {' '.join(adapters)}{' clef' * CLEF}", {"test.json": WORK / "test.json", **adapters}, ["answers.json"])
    got = json.loads((WORK / "answers.json").read_text())
else:
    got = {"clef": clef(test)} if CLEF else {}  # first, while the GPU is free
    got |= {n: answers(path, test) for n, path in adapters.items()}
if CLEF:
    answered["clef"] = got["clef"]
cols = {"current router": got["tuned"][0], "tuned adapter": got["tuned"][1]}
picks = {"current router": got["tuned"][2], "tuned adapter": got["tuned"][3]}
if "installed" in got:
    cols["installed adapter"] = got["installed"][1]
if BENCH:
    for n, js in answered.items():
        cols |= {f"{n} ({k})": [j[k] for j in js] for k in ("choice", "score", "router")}
        picks[n] = [j["model"] for j in js]
cols[f"always {common}"] = [common] * len(test)
likely = Counter(r["model"] for r in fit if r["first"]).most_common(1)[0][0]
picks[f"always {likely}"] = [likely] * len(test)
opens = [i for i, r in enumerate(test) if r["first"]]  # the model is only ever picked on these

off = {n: sum(abs(LEVELS.index(r["effort"]) - LEVELS.index(e)) for r, e in zip(test, c)) / len(test)
       for n, c in cols.items()}
print(f"\nheld out ({len(test)} prompts)   matches label   mean levels off")
for n, c in cols.items():
    print(f"  {n:<18} {sum(r['effort'] == e for r, e in zip(test, c)) / len(test):>13.0%} {off[n]:>17.2f}")

hit = {n: sum(test[i]["model"] == c[i] for i in opens) / max(len(opens), 1) for n, c in picks.items()}
print(f"\nheld out model, {len(opens)} sessions ({dict(Counter(test[i]['model'] for i in opens))})   matches label")
for n, h in hit.items():
    print(f"  {n:<18} {h:>13.0%}")

if BENCH:
    for n, js in answered.items():
        ms = sorted(j["ms"] for j in js)
        print(f"\n{n}: median {ms[len(ms) // 2]:.0f} ms a call" + (f", ${sum(j['cost'] for j in js):.4f}" if js[0]["cost"] else ""))
    sys.exit()

beaten = [n for n in ("current router", "installed adapter") if n in off and off[n] <= off["tuned adapter"]]
if beaten:
    sys.exit(f"\nThe adapter is no closer to the labels than the {beaten[0]}: not installed.")
if hit["current router"] > hit["tuned adapter"]:
    sys.exit("\nThe adapter picks the model worse than the current router: not installed.")
shutil.rmtree(classifier.TUNED, ignore_errors=True)
shutil.copytree(run / "final", classifier.TUNED)
print(f"\ninstalled to {classifier.TUNED}; restarting the daemon")
subprocess.run([ROOT / "scripts/gliner.sh", "stop"])
subprocess.run([ROOT / "scripts/gliner.sh", "start"], check=True)
