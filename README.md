<div align="center">

# magic-router

**Sonnet or Opus for the session. An effort level for every prompt.**<br>
Picked by a classifier that runs on your machine.

[![version](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FDustinVerzal%2Fmagic-router%2Fmain%2F.claude-plugin%2Fplugin.json&query=%24.version&label=version&style=flat-square&color=1b1b20)](.claude-plugin/plugin.json)
[![Claude Code 2.1.287+](https://img.shields.io/badge/Claude_Code-2.1.287%2B-1b1b20?style=flat-square)](#requirements)
[![macOS · Linux](https://img.shields.io/badge/runs_on-macOS_·_Linux-1b1b20?style=flat-square)](#requirements)
[![MIT license](https://img.shields.io/badge/license-MIT-1b1b20?style=flat-square)](LICENSE)

<br>

<img src=".github/assets/session.svg" width="100%" alt="One session with the router band under each prompt. The first prompt, about designing a sharded job queue, scores 1.86 and picks Opus 5.5 at xhigh effort in 326 ms. A short 'go ahead' keeps xhigh. A prompt to add retry with backoff scores 0.49 and drops to medium, and fixing a typo scores 0.21 and drops to low, while the model stays on Opus 5.5.">

<sub>One session. Scores and latencies are real classifier output for these prompts.</sub>

<br>

[Install](#install) · [First run](#first-run) · [How it routes](#how-a-prompt-is-routed) · [Troubleshooting](#troubleshooting)

</div>

<br>

A Claude Code mod that routes each session to **Sonnet 5.5 or Opus 5.5** and each prompt to an **effort level**, using [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) running locally.

- **The model is picked once**, from the session's first prompt. Switching models mid-conversation throws away the prompt cache, so the model stays fixed after that.
- **Effort is picked again on every prompt.** Short replies ("yes", "go ahead") keep the last effort.
- **Why:** a rename or a typo fix doesn't need Opus at `xhigh`, and a distributed-systems design shouldn't get Sonnet at `low`. The router spends the capability where a prompt needs it, so you pick neither by hand, and the classifier runs on your machine, so no prompt goes to a third party to be routed.
- **A band above the prompt** shows the model, the effort, how long classifying took, and last turn's cache-read %.

## Requirements

- Claude Code **2.1.287 or later** (`claude --version`)
- macOS or Linux
- `curl`. If [`uv`](https://docs.astral.sh/uv/) isn't installed, the first session installs it to `~/.local/bin` (no shell profile edits), and uv fetches a suitable Python (3.10–3.13) by itself.
- About 2 GB of free RAM while the classifier runs, and about 2 GB of disk for torch and the weights

## Install

In Claude Code:

```
/plugin marketplace add DustinVerzal/magic-router
/plugin install magic-router@magic-router
```

Restart Claude Code. That's all. The first session installs and starts the classifier itself (see [First run](#first-run)).

<details>
<summary><b>Warm it up first</b>: skip the first-run download wait</summary>
<br>

Without this step, your first session waits while torch installs and the weights download. To do that ahead of time:

```sh
git clone https://github.com/DustinVerzal/magic-router
magic-router/scripts/gliner.sh setup
```

`setup` installs uv if it's missing, installs torch and gliner2, downloads the weights (~1.7 GB), and leaves the daemon running. The downloads land in uv's and Hugging Face's shared caches, so the installed plugin reuses them; the clone is only needed to run the script.

</details>

<details>
<summary><b>From source</b>: load a checkout to hack on it</summary>
<br>

To hack on it, load a checkout for one session instead of installing:

```sh
git clone https://github.com/DustinVerzal/magic-router
claude --plugin-dir ./magic-router
```

Edits to `hooks/` reload while the session runs.

</details>

## What it does on your machine

- **Installs `uv`** with Astral's installer (`curl … | sh`, to `~/.local/bin`, no shell profile edits) if it's missing, then lets uv fetch Python, torch and `gliner2`.
- **Runs one background daemon** on `127.0.0.1` only (port `8765`). It outlives your sessions and is shared by all of them; stop it with `scripts/gliner.sh stop`.
- **Keeps its files in `~/.cache/magic-router`** and downloads the weights from Hugging Face once.
- **Sends nothing off your machine to route.** Each prompt's first 2,000 characters go to the local daemon and nowhere else (or to your own ssh host, if you [run the daemon there](#managing-the-classifier)). Only the optional [tuning](#tune-it-to-your-prompts) (your prompts go to Claude through `claude -p`, and to your own ssh host if you set one) and [`just bench`](#against-jev) (to the hosted models whose keys you set) reach beyond it.

## Configuration

Nothing needs configuring. These environment variables change the defaults:

| Variable | Default | Effect |
|---|---|---|
| `ROUTER_PORT` | `8765` | The daemon's port. Set it for Claude Code and for `scripts/gliner.sh`. |
| `ROUTER_MODEL` | `fastino/GLiNER2.5-Decide` | The Hugging Face model the daemon loads. |
| `ROUTER_WAIT` | `900` | Seconds `gliner.sh start` waits for the model on a first run. |
| `TUNE_HOST` | none | An ssh host with an NVIDIA GPU to tune on. |
| `AA_API_KEY` | none | [Artificial Analysis](https://artificialanalysis.ai/api) key for `just benchmarks`. |
| `OPENROUTER_KEY`, `FASTINO_API_KEY` | none | Hosted-model keys for `just bench`. |

The routing policy (labels, weights, thresholds) is in [`hooks/route.ts`](hooks/route.ts).

## Updating

Updates ship when the plugin version is bumped, which happens automatically on every merge to `main`. Third-party marketplaces don't auto-update by default, so turn it on once:

1. Run `/plugin` and open the **Marketplaces** tab.
2. Select **magic-router** and choose **Enable auto-update**.

Claude Code then pulls new versions at startup and asks you to restart or run `/reload-plugins`. To update by hand instead:

```
/plugin marketplace update magic-router
/plugin update magic-router@magic-router
```

## First run

The first session after a reboot starts the classifier daemon (`server/classifier.py`, on `127.0.0.1:8765`). The daemon keeps running after the session ends, and every later session reuses it.

| When | What happens |
|---|---|
| **First run ever** | uv installs (if missing), then torch and gliner2 install, and the weights download (~1.7 GB). The band shows `classifier unreachable` until this finishes, and the session keeps its own model. |
| **First session after a reboot** | The model loads in about 10 s, and the first prompt waits for it. |
| **Every later prompt** | About 0.3 s of CPU. |

## Managing the classifier

```sh
scripts/gliner.sh setup    # install uv if missing, install torch + gliner2, download the weights, start
scripts/gliner.sh start    # start the daemon (if it isn't up) and wait until the model is loaded
scripts/gliner.sh stop     # stop the daemon
scripts/gliner.sh status   # print /health
scripts/gliner.sh check    # run the classifier's offline self-check
scripts/gliner.sh logs     # follow the daemon log
scripts/gliner.sh remote HOST   # run the daemon on an ssh host instead (macOS), such as a box with a GPU
scripts/gliner.sh local    # run it here again
```

The log is at `~/.cache/magic-router/classifier.log`. Without a checkout, stop the daemon with `pkill -f server/classifier.py`.

`remote HOST` copies this checkout and your tuned adapter to `~/.cache/magic-router` on the host, starts the daemon there (on its NVIDIA GPU if it has one), and adds a launch agent that forwards port 8765 to it over ssh. The mod keeps calling `127.0.0.1:8765` and needs no change, and your Mac no longer runs the model. After that, `setup`, `start`, `stop`, `check` and `logs` act on the host. The host needs key-based ssh and rsync. If the host is down when a session starts, the mod starts a local daemon, which then holds the port until you stop it.

## How a prompt is routed

All routing policy lives in [`hooks/route.ts`](hooks/route.ts). The daemon only answers the questions the mod sends it.

1. **Task type**: one probability distribution over seven labels. Each label maps to an eval family of the [Artificial Analysis Intelligence Index v4.1](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-1):

   | Label | Eval family |
   |---|---|
   | `agentic_coding` | Terminal-Bench |
   | `scientific_coding` | SciCode |
   | `tool_use` | τ³-Bench |
   | `knowledge_work` | GDPval-AA |
   | `long_context` | AA-LCR |
   | `knowledge_qa` | AA-Omniscience, GPQA |
   | `reasoning` | HLE, CritPt |

2. **Complexity**: seven independent yes/no signals, such as `multi_file`, `planning`, `deep_reasoning`, `large_scope`, and `quick`.
3. **Score**: `Σ weight × P(signal) + Σ bias × P(task)`. The task bias leans toward Opus where its lead on the matching evals is widest.
4. **Effort and model**: the score maps to `low < 0.4 ≤ medium < 1.0 ≤ high < 1.5 ≤ xhigh < 2.0 ≤ max`. On the first prompt, a score ≥ 1.1 picks Opus; anything lower picks Sonnet.

<img src=".github/assets/scale.svg" width="100%" alt="The score scale. Effort bands run low below 0.4, medium to 1.0, high to 1.5, xhigh to 2.0, and max above. A dashed line at 1.1 splits Sonnet from Opus. The session's prompts sit at 0.21 (fix the typo, low), 0.49 (add retry and tests, medium), and 1.86 (design the job queue, xhigh).">

> [!NOTE]
> The weights and thresholds were tuned by eye on 15 prompts. To fit the effort pick to your own prompts, see [Tune it to your prompts](#tune-it-to-your-prompts).

## Tune it to your prompts

`just tune` (or `uv run --script scripts/tune.py`) trains the classifier on how you actually work:

1. It reads every prompt you have sent Claude Code or Codex: transcripts in `~/.claude/projects`, older prompts from `~/.claude/history.jsonl`, and your own Codex CLI and desktop threads in `~/.codex` (no subagents, `codex exec` or orchestrators). Slash commands, `$skills` and short follow-ups are skipped, because the router never classifies them. Pass a number (`just tune 500`) to use only your newest prompts.
2. Opus at xhigh effort labels the effort each prompt needed. It labels 25 prompts per `claude -p` call, with no tools, no settings and no saved session. Each label and a one-line reason go to `~/.cache/magic-router/tune/labels.jsonl`, so a rerun only labels new prompts. Skim them there.
3. It trains a LoRA adapter for the classifier on 70% of your sessions, on CPU. With `TUNE_HOST=<ssh host> just tune`, it trains and scores on that host instead (one with an NVIDIA GPU, uv and rsync), and deletes the prompts it sent there afterwards.
4. On the other 30%, it compares the adapter's effort with the score's, and with the adapter already installed, if any. If the new adapter is closer to the labels than both, it installs it to `~/.cache/magic-router/tuned` and restarts the daemon. Otherwise it changes nothing.

The numbers below are from the author's own history, so yours will differ. On 3,344 prompts (Claude Code and Codex, six months), labelling cost $10.33 at API prices, and training took about 15 minutes on an RTX 4080. On CPU it is too slow, about 15 minutes per 400 prompts. On 972 held-out prompts:

| | Matches label | Mean levels off |
|---|---|---|
| Adapter trained on 3,344 prompts | 66% | 0.36 |
| Adapter trained on 404 Claude Code prompts | 54% | 0.49 |
| Always `medium` | 44% | 0.59 |
| The score | 40% | 0.69 |

With an adapter installed, the daemon answers effort from it, which adds a second pass of about 0.1 s. The score still picks the model on a session's first prompt. To undo, delete `~/.cache/magic-router/tuned` and run `scripts/gliner.sh stop`.

<details id="against-jev">
<summary><b>Against Jev</b>: the adapter versus a hosted decision model</summary>
<br>

`just bench` scores the trained adapter against hosted decision models on those same 972 held-out prompts, without retraining. It sends each prompt's first 2,000 characters to the provider: `OPENROUTER_KEY` adds [Jev](https://openrouter.ai/typesafe/jev-1.13) (TypeSafe), which needs no training. Jev got the labeller's definition of each level, and was asked for the effort directly, as a choice among the five levels and as a score on the ordered scale. A third column puts Jev's answers to the router's own task and signal questions through the score's weights.

| | Matches label | Mean levels off |
|---|---|---|
| Tuned adapter | 66% | 0.36 |
| Jev, effort as a score | 63% | 0.42 |
| Jev, effort as a choice | 56% | 0.53 |
| Always `medium` | 44% | 0.59 |
| The score | 40% | 0.69 |
| Jev through the score's weights | 27% | 1.06 |

Asked for effort directly, Jev comes within 3 points of the adapter without seeing any of your prompts, and the ordered score beats the choice. The adapter is still ahead, and it was trained on labels from the same labeller, so the margin favors it. It runs locally, while Jev is a hosted call (median 234 ms, $0.049 for all 972 prompts) that sends your prompts to a third party. Jev's answers through the score's weights do worse than guessing `medium`, because those weights were set for GLiNER. Jev does not replace the classifier inside the router as is.

</details>

## When it stands aside

- **Subagents**: their requests keep their own model and effort.
- **`/model`, `/effort`, or a fallback**: if any of these changes what the engine asks for, the router stops rewriting for the rest of the session.
- **Classifier unreachable on the first prompt**: the session keeps its own model, and effort routing starts once the daemon answers.
- **`/clear`**: the next prompt picks a model again.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Band says `classifier unreachable` | Run `scripts/gliner.sh start` and read the log it names. On a first run, it's usually still downloading. |
| Daemon never starts from Claude Code, but `gliner.sh start` works | Read `~/.cache/magic-router/classifier.log`: the uv install or a torch download likely failed (offline, proxy). |
| Band says `stood aside` | You picked a model or effort by hand (`/model`, `/effort`). `/clear` to let the router pick again. |
| Port 8765 is taken by something else | Set `ROUTER_PORT` to a free port for Claude Code and the shell you run `gliner.sh` from, then restart Claude Code. |

## Uninstall

```
/plugin uninstall magic-router@magic-router
/plugin marketplace remove magic-router
```

Then stop the daemon (`pkill -f server/classifier.py`) and, to reclaim the disk, delete `~/.cache/magic-router` and the weights under `~/.cache/huggingface/hub/models--fastino--GLiNER2.5-Decide`.

## Benchmarks

`just benchmarks` (or `AA_API_KEY=... uv run --script scripts/benchmarks.py`) pulls the [Artificial Analysis](https://artificialanalysis.ai/api) evals for Sonnet 5.5, Opus 5.5 and Fable 5.1, at every effort level, into a table (and caches the raw JSON in `~/.cache/magic-router`). Put `AA_API_KEY=...` in a git-ignored `.env`, or export it in your shell. Fable 5.1 is wired in but off: set `FABLE_AT` in `hooks/route.ts` to a score above `OPUS_AT` to route the hardest first prompts to it.

## Development

```sh
claude plugin validate .
claude plugin test .                        # routing, stickiness, stand-aside, band
uv run --script server/classifier.py --check
```

## License

[MIT](LICENSE)
