# model-router

A Claude Code mod that routes each session to **Sonnet 5.5 or Opus 5.5** and each prompt to an **effort level**, using [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) running locally.

- **The model is picked once**, from the session's first prompt. Switching models mid-conversation throws away the prompt cache, so the model stays fixed after that.
- **Effort is picked again on every prompt.** Short replies ("yes", "go ahead") keep the last effort.
- **A band above the prompt** shows the route, the task distribution, the strongest complexity signals, and last turn's cache-read %.

```
router opus 5.5 · effort max  score 2.12 · 326ms · cache 94% read  [ hide ]
task agentic_coding 42% · scientific_coding 16% · long_context 14%
signals large_scope 81% · verification 31% · multi_file 28%
```

## Run it

Needs [`uv`](https://docs.astral.sh/uv/) and Claude Code 2.1.287 or later.

```sh
claude --plugin-dir ~/repos/claude-router
```

The first session after a reboot starts the classifier daemon (`server/classifier.py`, on `127.0.0.1:8765`). The daemon keeps running after the session ends, and every later session reuses it. Startup costs:

- **First run ever:** uv installs torch and gliner2, and the weights download (~1.7 GB).
- **First session after a reboot:** the model loads in about 10 s, and the first prompt waits for it.
- **Every later prompt:** about 0.3 s of CPU.

The log is at `~/.cache/model-router/classifier.log`. To stop the daemon: `pkill -f server/classifier.py`. It holds about 2 GB of RAM while it runs.

## How a prompt is routed

All routing policy lives in `hooks/route.ts`. The daemon only answers the questions the mod sends it.

1. **Task type**: one probability distribution over seven labels. Each label maps to an eval family of the [Artificial Analysis Intelligence Index v4.1](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-1):

   | Label | Eval family |
   |---|---|
   | agentic_coding | Terminal-Bench |
   | scientific_coding | SciCode |
   | tool_use | τ³-Bench |
   | knowledge_work | GDPval-AA |
   | long_context | AA-LCR |
   | knowledge_qa | AA-Omniscience, GPQA |
   | reasoning | HLE, CritPt |

2. **Complexity**: seven independent yes/no signals, such as multi_file, planning, deep_reasoning, large_scope, and quick.
3. **Score**: `Σ weight × P(signal) + Σ bias × P(task)`. The task bias leans toward Opus where its lead on the matching evals is widest.
4. **Effort and model**: the score maps to `low < 0.4 ≤ medium < 1.0 ≤ high < 1.5 ≤ xhigh < 2.0 ≤ max`. On the first prompt, a score ≥ 1.1 picks Opus; anything lower picks Sonnet.

The weights and thresholds were tuned by eye on 15 prompts, so retune them on your own traffic.

## When it stands aside

- **Subagents**: their requests keep their own model and effort.
- **`/model`, `/effort`, or a fallback**: if any of these changes what the engine asks for, the router stops rewriting for the rest of the session.
- **Classifier unreachable on the first prompt**: the session keeps its own model, and effort routing starts once the daemon answers.
- **`/clear`**: the next prompt picks a model again.

## Check it

```sh
claude plugin validate .
claude plugin test .                        # routing, stickiness, stand-aside, band
uv run --script server/classifier.py --check
```
