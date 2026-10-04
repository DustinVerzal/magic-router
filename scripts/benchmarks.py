# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Pull Artificial Analysis benchmarks for the models the router picks between.

    AA_API_KEY=... uv run --script scripts/benchmarks.py [name-substring ...]

Free key: https://artificialanalysis.ai/api. Default filters: the Sonnet, Opus and Fable versions in MODELS (hooks/route.ts),
at every effort level. With no args it also rewrites TASK_BIAS in hooks/route.ts from the Opus-minus-Sonnet gaps.
Prints one row per eval, one column per model; saves the raw response to
~/.cache/magic-router/benchmarks.json (the previous copy becomes benchmarks.prev.json) so you can diff.
"""
import json
import os
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
CACHE = Path.home() / ".cache/magic-router"
ROUTE = Path(__file__).parent.parent / "hooks/route.ts"

key = os.environ.get("AA_API_KEY") or sys.exit("set AA_API_KEY (free at https://artificialanalysis.ai/api)")
req = urllib.request.Request(URL, headers={"x-api-key": key, "User-Agent": "magic-router"})
with urllib.request.urlopen(req, timeout=30) as r:
    raw = r.read()

CACHE.mkdir(parents=True, exist_ok=True)
(CACHE / "benchmarks.json").replace(CACHE / "benchmarks.prev.json") if (CACHE / "benchmarks.json").exists() else None
(CACHE / "benchmarks.json").write_bytes(raw)

route = ROUTE.read_text()
# claude-opus-5-5 -> "opus 5.5"
defaults = [f"{n} {a}.{b}" for n, a, b in re.findall(r"'claude-(\w+)-(\d+)-(\d+)'", route)]
wanted = [w.lower() for w in sys.argv[1:]] or defaults
models = [m for m in json.loads(raw)["data"] if any(w in m["name"].lower() for w in wanted)]
if not models:
    sys.exit(f"no model name matches {wanted}; saved the full response anyway")

EFFORT = ["max", "xhigh", "high", "medium", "low"]


def label(name):  # "Claude Opus 5.5 (Xhigh, Default Fallback)" -> "opus 5.5 xhigh"
    m = re.match(r"Claude (.+?) \((\w+)", name)
    return f"{m[1].lower()} {m[2].lower()}" if m else name


def order(m):
    l = label(m["name"]).split()
    return (" ".join(l[:-1]), EFFORT.index(l[-1]) if l[-1] in EFFORT else len(EFFORT))


models.sort(key=order)
rows = {}  # label -> {model name: value}
for m in models:
    for k, v in {**m.get("evaluations", {}), **{f"price/{k}": v for k, v in m.get("pricing", {}).items()}}.items():
        if isinstance(v, (int, float)):
            rows.setdefault(k, {})[label(m["name"])] = v

names = [label(m["name"]) for m in models]
width = max(len(k) for k in rows)
print(f"{'':{width}}  " + "  ".join(f"{n:>16}" for n in names))
for k, vals in sorted(rows.items()):
    print(f"{k:{width}}  " + "  ".join(f"{vals[n]:>16.3f}" if n in vals else f"{'-':>16}" for n in names))

if sys.argv[1:]:
    sys.exit(0)  # a filtered run is a look, not a retune

# TASK_BIAS: per task family, Opus's lead over Sonnet on its AA eval vs. the Intelligence Index lead, averaged over
# the effort levels both models have. Bias = (gap - overall gap) * SCALE; families with no eval get 0.
SCALE = 0.03
FAMILY_EVAL = {"reasoning": "hle", "agentic_coding": "terminalbench_v4_0", "scientific_coding": "scicode", "long_context": "lcr"}
TASKS = ["reasoning", "agentic_coding", "scientific_coding", "long_context", "tool_use", "knowledge_work", "knowledge_qa"]


def gap(ev, scale):
    by = {label(m["name"]): m["evaluations"].get(ev) for m in models}
    ds = [(by[o] - by[o.replace("opus", "sonnet")]) * scale
          for o in by if o.startswith("opus") and by[o] is not None and by.get(o.replace("opus", "sonnet")) is not None]
    return sum(ds) / len(ds)


overall = gap("artificial_analysis_intelligence_index", 1)
gaps = {f: gap(e, 100) for f, e in FAMILY_EVAL.items()}
bias = {f: round((gaps[f] - overall) * SCALE, 2) if f in gaps else 0 for f in TASKS}
bias = {f: b + 0 for f, b in bias.items()}  # -0.0 -> 0
note = ", ".join(f"{FAMILY_EVAL[f]} {gaps[f]:+.1f}" for f in sorted(gaps, key=gaps.get, reverse=True))
block = f"""// Where Opus's lead over Sonnet on the matching AA eval is wider or narrower than its overall lead.
// Gaps are averaged over the effort levels both have (the router moves effort per prompt), in points
// (scripts/benchmarks.py, {date.today()}): {note},
// against the Intelligence Index's {overall:+.1f}, which families with no eval get. Bias = (gap - {overall:.1f}) * {SCALE},
// so a family at the overall gap is 0 and the scale of OPUS_AT and EFFORTS holds.
export const TASK_BIAS: Probs = {{
""" + "".join(f"  {f}: {bias[f]:g},\n" for f in sorted(TASKS, key=lambda f: -bias[f])) + "}\n"
new, n = re.subn(r"// Where Opus.*?export const TASK_BIAS: Probs = \{.*?\n\}\n", lambda _: block, route, flags=re.S)
if n != 1:
    sys.exit("could not find the TASK_BIAS block in hooks/route.ts")
ROUTE.write_text(new)
print(f"\nrewrote TASK_BIAS in {ROUTE}:\n{block}")
