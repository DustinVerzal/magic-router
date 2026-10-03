# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["gliner2[local]>=2,<3"]
# ///
"""GLiNER2.5-Decide classifier daemon for claude-router, shared by every session.

GET  /health    {"ready": bool, "model": str}
POST /classify  {"text", "choose": {label: description}, "flags": {label: description}}
             -> {"choose": {label: p} (one distribution), "flags": {label: p} (independent), "ms"}

Binds 127.0.0.1 only. Exits quietly when the port is taken: another session's daemon is up.
"""
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = os.environ.get("ROUTER_MODEL", "fastino/GLiNER2.5-Decide")
PORT = int(os.environ.get("ROUTER_PORT", "8765"))

model = None
ready = threading.Event()
lock = threading.Lock()  # ponytail: one inference at a time; ~0.3s each, fine for a handful of sessions


def load():
    global model
    from gliner2 import AutoExtractor

    t = time.perf_counter()
    model = AutoExtractor.from_pretrained(MODEL_ID)
    ready.set()
    print(f"loaded {MODEL_ID} in {time.perf_counter() - t:.1f}s", flush=True)


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


def classify(text, choose, flags):
    spec = lambda labels: {"labels": labels, "multi_label": True, "cls_threshold": 0.0}
    with lock:
        out = model.classify_text(text, {"choose": spec(choose), "flags": spec(flags)}, include_confidence=True)
    return {"choose": distribution(out["choose"]), "flags": {d["label"]: d["confidence"] for d in out["flags"]}}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "not found"})
        self.reply(200, {"ready": ready.is_set(), "model": MODEL_ID})

    def do_POST(self):
        if self.path != "/classify":
            return self.reply(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text, choose, flags = str(body["text"]), dict(body["choose"]), dict(body["flags"])
            if len(choose) < 2 or len(flags) < 2:
                raise ValueError("choose and flags need at least two labels each")
        except (ValueError, KeyError, TypeError) as err:
            return self.reply(400, {"error": str(err)})
        if not ready.wait(30):
            return self.reply(503, {"error": "model still loading"})
        t = time.perf_counter()
        result = classify(text, choose, flags)
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
    print(f"claude-router classifier on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
