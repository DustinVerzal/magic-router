# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Pull Artificial Analysis benchmarks for the models the router picks between.

    AA_API_KEY=... uv run --script scripts/benchmarks.py [name-substring ...]

Free key: https://artificialanalysis.ai/api. Default filters: Sonnet 5.5, Opus 5.5 and Fable 5.1, at every effort level.
Prints one row per eval, one column per model; saves the raw response to
~/.cache/model-router/benchmarks.json (the previous copy becomes benchmarks.prev.json) so you can diff.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
CACHE = Path.home() / ".cache/model-router"

key = os.environ.get("AA_API_KEY") or sys.exit("set AA_API_KEY (free at https://artificialanalysis.ai/api)")
req = urllib.request.Request(URL, headers={"x-api-key": key, "User-Agent": "magic-router"})
with urllib.request.urlopen(req, timeout=30) as r:
    raw = r.read()

CACHE.mkdir(parents=True, exist_ok=True)
(CACHE / "benchmarks.json").replace(CACHE / "benchmarks.prev.json") if (CACHE / "benchmarks.json").exists() else None
(CACHE / "benchmarks.json").write_bytes(raw)

wanted = [w.lower() for w in sys.argv[1:]] or ["sonnet 5.5", "opus 5.5", "fable 5.1"]
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
