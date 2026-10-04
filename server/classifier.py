# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["gliner2[local]>=2,<3"]
# ///
"""GLiNER2.5-Decide classifier daemon for magic-router, shared by every session.

GET  /health    {"ready": bool, "model": str, "tuned": bool}
POST /classify  {"text", "choose": {label: description}, "flags": {label: description}, "effort"?: [level]}
             -> {"choose": {label: p} (one distribution), "flags": {label: p} (independent),
                 "effort": {level: p} (one distribution; only with a tuned adapter), "ms"}

Binds 127.0.0.1 only. Exits quietly when the port is taken: another session's daemon is up.
"""
import json
import math
import os
import sys
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_ID = os.environ.get("ROUTER_MODEL", "fastino/GLiNER2.5-Decide")
PORT = int(os.environ.get("ROUTER_PORT", "8765"))
TUNED = Path.home() / ".cache/magic-router/tuned"  # a LoRA adapter trained on your prompts by scripts/tune.py

model = None
tuned = None  # model with the adapter, when there is one
ready = threading.Event()
lock = threading.Lock()  # ponytail: one inference at a time; ~0.3s each, fine for a handful of sessions


def load(adapter=TUNED):
    global model, tuned
    from gliner2 import AutoExtractor

    t = time.perf_counter()
    model, tuned = AutoExtractor.from_pretrained(MODEL_ID), None  # tune.py loads one adapter after another
    if (adapter / "adapter_config.json").exists():
        from peft import PeftModel

        tuned = PeftModel.from_pretrained(model, str(adapter))
    import torch

    if torch.cuda.is_available():  # e.g. the daemon on a GPU box, reached through `scripts/gliner.sh remote`
        model.to("cuda")  # the adapter's layers live inside model, so they move too
    ready.set()
    print(f"loaded {MODEL_ID}{f' + {adapter}' if tuned else ''} in {time.perf_counter() - t:.1f}s", flush=True)


def distribution(scores):
    """Single-label confidence is a softmax over label logits; multi_label at threshold 0 gives each
    label's sigmoid instead, so invert those to logits and softmax for the whole distribution."""
    logits = {}
    for d in scores:
        p = min(max(d["confidence"], 1e-6), 1 - 1e-6)
        logits[d["label"]] = math.log(p / (1 - p))
    top = max(logits.values())
    exp = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def classify(text, choose, flags, effort=None):
    """choose and flags always come from the base model. The adapter, trained on effort alone, would shift them,
    so it answers effort in a second pass (~0.13s); the base model's own effort answer is near flat, so none without one."""
    spec = lambda labels: {"labels": labels, "multi_label": True, "cls_threshold": 0.0}
    with lock:
        with tuned.disable_adapter() if tuned else nullcontext():
            out = model.classify_text(text, {"choose": spec(choose), "flags": spec(flags)}, include_confidence=True)
        result = {"choose": distribution(out["choose"]), "flags": {d["label"]: d["confidence"] for d in out["flags"]}}
        if tuned and effort:
            result["effort"] = distribution(model.classify_text(text, {"effort": spec(effort)}, include_confidence=True)["effort"])
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "not found"})
        self.reply(200, {"ready": ready.is_set(), "model": MODEL_ID, "tuned": tuned is not None})

    def do_POST(self):
        if self.path != "/classify":
            return self.reply(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text, choose, flags = str(body["text"]), dict(body["choose"]), dict(body["flags"])
            effort = list(body.get("effort") or [])
            if len(choose) < 2 or len(flags) < 2:
                raise ValueError("choose and flags need at least two labels each")
        except (ValueError, KeyError, TypeError) as err:
            return self.reply(400, {"error": str(err)})
        if not ready.wait(30):
            return self.reply(503, {"error": "model still loading"})
        t = time.perf_counter()
        result = classify(text, choose, flags, effort)
        self.reply(200, {**result, "ms": round((time.perf_counter() - t) * 1000)})

    def reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def check():
    sig = lambda x: 1 / (1 + math.exp(-x))
    logits = {"a": 1.3, "b": -1.5, "c": -2.4}
    got = distribution([{"label": k, "confidence": sig(v)} for k, v in logits.items()])
    want = {k: math.exp(v) / sum(math.exp(u) for u in logits.values()) for k, v in logits.items()}
    assert all(abs(got[k] - want[k]) < 1e-9 for k in logits), (got, want)
    print("ok")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        sys.exit(0)
    threading.Thread(target=load, daemon=True).start()
    print(f"magic-router classifier on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
