# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch>=2.11", "transformers>=5.10,<6", "accelerate", "bitsandbytes", "pillow", "torchvision"]
# ///
"""Clef-flash (Cloudflare's 9B decision model) answering SystemOne questions on this machine's NVIDIA GPU, for
`just bench clef` (scripts/tune.py), which puts its answers next to the tuned adapter's on the held-out prompts.

    uv run --script scripts/clef.py in.json out.json

in.json is {"questions": {SystemOne questions}, "texts": [...]}; out.json gets one {"answers", "ms"} per text.
Its own environment, since gliner2 and Clef want different transformers; the first run downloads 19 GB of weights.
CLEF_BITS picks the precision: 8 (the default: 11 GB of VRAM, a median 0.5 s a prompt on an RTX 4080), 4 (8 GB and
a third faster, but it moved effort probabilities by up to 0.23 from 8 bits' and changed 2 of 10 answers in a spot check)
or 16 (bf16, as Cloudflare serves it, about 20 GB). Torchvision and Pillow are there only because the processor
Cloudflare's loader builds wants them; flash-linear-attention made no difference on short prompts, so it's left out.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig

if not torch.cuda.is_available():  # before the 19 GB download, say on a Mac that forgot TUNE_HOST
    sys.exit("Clef-flash needs an NVIDIA GPU: run `just bench clef` there, or set TUNE_HOST to an ssh host with one")
# Pinned: the repo's joint_schema_model.py is imported and run below, so it stays the version that was read.
path = snapshot_download("Cloudflare/clef-flash", revision="17f0b0ad64efb65d273590632833508766b2aae6")
sys.path.insert(0, path)
from joint_schema_model import load_release_model, systemone  # noqa: E402

BITS = int(os.environ.get("CLEF_BITS", "8"))
if BITS not in (4, 8, 16):
    sys.exit(f"CLEF_BITS is 4, 8 or 16, not {BITS}")
quant = {8: BitsAndBytesConfig(load_in_8bit=True),
         4: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)}
t = time.perf_counter()
model, processor = load_release_model(path, **({"quantization_config": quant[BITS]} if BITS in quant else {}))
print(f"loaded Clef-flash at {BITS} bits in {time.perf_counter() - t:.0f}s, "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} GB on the GPU", flush=True)

# A prompt's tail is cut so it and the questions fit this many tokens, about 32,000 characters. Clef takes 16,384, but
# on a 16 GB card at 8 bits a prompt that long spills out of VRAM (8 minutes, where 8,192 tokens took 2 seconds).
# ponytail: 4 of 1,025 bench prompts still took about 3 minutes each, probably spilling too; lower this if a run crawls.
TOKENS = 8192
job = json.loads(Path(sys.argv[1]).read_text())
out = []
for i, text in enumerate(job["texts"], 1):
    t = time.perf_counter()
    request = {"model": "clef-flash", "state": text, "questions": job["questions"]}
    answers = systemone(model, processor, request, max_length=TOKENS)["answers"]
    out.append({"answers": answers, "ms": (time.perf_counter() - t) * 1000})
    if i % 100 == 0:
        print(f"  clef answered {i}/{len(job['texts'])}", flush=True)
Path(sys.argv[2]).write_text(json.dumps(out))
