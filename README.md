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

[Install](#install) · [How it works](#how-it-works) · [Privacy](#privacy) · [Tuning](#tune-it-to-your-prompts) · [Troubleshooting](#troubleshooting)

</div>

<br>

A typo fix doesn't need Opus at `xhigh`, and a distributed-systems design shouldn't get Sonnet at `low`. magic-router is a Claude Code plugin that makes that call for you: it routes each session to **Sonnet 5.5 or Opus 5.5** and each prompt to an **effort level**, using [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) running locally.

- **The model is picked once**, from the session's first prompt, and then stays put: switching models mid-conversation throws away the prompt cache.
- **Effort is picked on every prompt.** Short follow-ups ("yes", "go ahead") keep the last effort.
- **It runs on your machine.** About 0.3 s of CPU per prompt, and no prompt goes to a third party to be routed.
- **You can see every decision.** A band above the prompt shows the model, the effort, how long classifying took, and last turn's cache-read %.
- **It routes subagents too.** Each subagent gets its own model and effort from the task it was given. A model the caller or the agent's definition named is kept.
- **It gets out of your way** for the rest of the session once you pick a model or effort by hand ([details](#when-it-stands-aside)).
- **It learns how you work.** Optionally, [train it on your own prompt history](#tune-it-to-your-prompts): on held-out prompts, a tuned adapter matched the labelled effort 66% of the time, against 40% for the default score.

## Install

### Requirements

- Claude Code **2.1.287 or later** (`claude --version`)
- macOS or Linux, with `curl`
- About 2 GB of free RAM while the classifier runs, and about 2 GB of disk for torch and the weights

Python and [`uv`](https://docs.astral.sh/uv/) are optional: if uv is missing, the first session installs it to `~/.local/bin`, and uv fetches Python 3.10–3.13 itself.

### Add the plugin

In Claude Code:

```
/plugin marketplace add DustinVerzal/magic-router
/plugin install magic-router@magic-router
```

Restart Claude Code. There's nothing to configure.

### First run

The first session starts the classifier daemon in the background. It keeps running after the session ends, and every later session reuses it.

| When | What happens |
|---|---|
| **First run ever** | uv installs (if missing), torch and gliner2 install, and the weights download (~1.7 GB). Until then the band shows `classifier unreachable` and the session keeps its own model. |
| **First session after a reboot** | The model loads in about 10 s, and the first prompt waits for it. |
| **Every later prompt** | About 0.3 s of CPU. |

<details>
<summary><b>Warm it up first</b>: skip the first-run download wait</summary>
<br>

```sh
git clone https://github.com/DustinVerzal/magic-router
magic-router/scripts/gliner.sh setup
```

`setup` installs uv if it's missing, installs torch and gliner2, downloads the weights, and leaves the daemon running. The downloads land in uv's and Hugging Face's shared caches, so the installed plugin reuses them; the clone is only needed to run the script.

</details>

<details>
<summary><b>From source</b>: load a checkout to hack on it</summary>
<br>

```sh
git clone https://github.com/DustinVerzal/magic-router
claude --plugin-dir ./magic-router
```

This loads the checkout for one session instead of installing it. Edits to `hooks/` reload while the session runs.

</details>

## How it works

All routing policy lives in [`hooks/route.ts`](hooks/route.ts). The daemon only answers the questions the plugin sends it.

1. **Task type**: one probability distribution over seven labels, each mapped to an eval family of the [Artificial Analysis Intelligence Index v4.1](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-1).
2. **Complexity**: seven independent yes/no signals, such as `multi_file`, `planning`, `deep_reasoning`, `large_scope`, and `quick`.
3. **Score**: `Σ weight × P(signal) + Σ bias × P(task)`. The task bias leans toward Opus where its lead on the matching evals is widest.
4. **Effort and model**: the score maps to `low < 0.4 ≤ medium < 1.0 ≤ high < 1.5 ≤ xhigh < 2.0 ≤ max`. On the first prompt, a score ≥ 1.1 picks Opus; anything lower picks Sonnet.

<img src=".github/assets/scale.svg" width="100%" alt="The score scale. Effort bands run low below 0.4, medium to 1.0, high to 1.5, xhigh to 2.0, and max above. A dashed line at 1.1 splits Sonnet from Opus. The session's prompts sit at 0.21 (fix the typo, low), 0.49 (add retry and tests, medium), and 1.86 (design the job queue, xhigh).">

<details>
<summary><b>Task labels</b> and the evals behind them</summary>
<br>

| Label | Eval family |
|---|---|
| `agentic_coding` | Terminal-Bench |
| `scientific_coding` | SciCode |
| `tool_use` | τ³-Bench |
| `knowledge_work` | GDPval-AA |
| `long_context` | AA-LCR |
| `knowledge_qa` | AA-Omniscience, GPQA |
| `reasoning` | HLE, CritPt |

</details>

> [!NOTE]
> The default weights and thresholds were tuned by eye on 15 prompts. To fit them to how you work, see [Tune it to your prompts](#tune-it-to-your-prompts).

### When it stands aside

- **Subagents**: each one is classified once, on its first request, and keeps that route. Only what it would inherit from the session is rewritten: a model or effort set by the Agent call or the agent's definition is kept. Forks are left alone, because they share their parent's context and prompt cache.
- **`/model`, `/effort`, or a fallback**: if any of these changes what the engine asks for, the router stops rewriting for the rest of the session.
- **Classifier unreachable on the first prompt**: the session keeps its own model, and effort routing starts once the daemon answers.
- **`/clear`**: the next prompt picks a model again.

## Privacy

- **Routing never leaves your machine.** Each prompt's first 2,000 characters go to the daemon on `127.0.0.1` and nowhere else (or to your own ssh host, if you [run the daemon there](#managing-the-classifier)).
- **One background daemon**, on port `8765`, shared by all your sessions. It outlives them; [stop it](#managing-the-classifier) any time.
- **Files stay in `~/.cache/magic-router`**, plus the weights in Hugging Face's cache, downloaded once.
- **uv comes from Astral's installer** (`curl … | sh`, to `~/.local/bin`, no shell profile edits) if it's missing.

Only the opt-in tools reach further: [`just tune`](#tune-it-to-your-prompts) reads your local transcripts and sends prompts to Claude through `claude -p` (and to your ssh host, if you set one), and [`just bench`](#against-hosted-decision-models) sends prompts to the hosted models whose keys you set.

## Configuration

Nothing needs configuring. These environment variables change the defaults:

| Variable | Default | Effect |
|---|---|---|
| `ROUTER_PORT` | `8765` | The daemon's port. Set it for Claude Code and for `scripts/gliner.sh`. |
| `ROUTER_MODEL` | `fastino/GLiNER2.5-Decide` | The Hugging Face model the daemon loads. |
| `ROUTER_WAIT` | `900` | Seconds `gliner.sh start` waits for the model on a first run. |
| `TUNE_HOST` | none | An ssh host with an NVIDIA GPU to [tune](#tune-it-to-your-prompts) on. |
| `AA_API_KEY` | none | [Artificial Analysis](https://artificialanalysis.ai/api) key for `just benchmarks`. |
| `OPENROUTER_KEY`, `FASTINO_API_KEY` | none | Hosted-model keys for `just bench`. |

Weights and thresholds are constants in [`hooks/route.ts`](hooks/route.ts). Fable 5.1 is wired in but off: set `FABLE_AT` there to a score above `OPUS_AT` to send the hardest first prompts to it.

## Managing the classifier

These need a checkout of this repo:

```sh
scripts/gliner.sh setup         # install uv if missing, install torch + gliner2, download the weights, start
scripts/gliner.sh start         # start the daemon (if it isn't up) and wait until the model is loaded
scripts/gliner.sh stop          # stop the daemon
scripts/gliner.sh status        # print /health
scripts/gliner.sh check         # run the classifier's offline self-check
scripts/gliner.sh logs          # follow the daemon log
scripts/gliner.sh remote HOST   # run the daemon on an ssh host instead (macOS), such as a box with a GPU
scripts/gliner.sh local         # run it here again
```

Without a checkout, stop the daemon with `pkill -f server/classifier.py`. The log is at `~/.cache/magic-router/classifier.log`.

<details>
<summary><b>Running the daemon on another machine</b></summary>
<br>

`remote HOST` copies this checkout and your tuned adapter to `~/.cache/magic-router` on the host, starts the daemon there (on its NVIDIA GPU if it has one), and adds a launch agent that forwards port 8765 to it over ssh. The plugin keeps calling `127.0.0.1:8765` and needs no change, and your Mac no longer runs the model. After that, `setup`, `start`, `stop`, `check` and `logs` act on the host.

The host needs key-based ssh and rsync. If it's down when a session starts, the plugin starts a local daemon, which then holds the port until you stop it.

</details>

## Tune it to your prompts

`just tune` trains the classifier on how you actually work. It's opt-in and needs a checkout. Labelling runs through `claude -p`, so it counts against your Claude subscription's usage, or bills your API key if that's how you're signed in. Run `just benchmarks` first: the labeller uses its scores.

1. **Collect.** It reads every prompt you've sent Claude Code or Codex, from `~/.claude/projects`, `~/.claude/history.jsonl` and `~/.codex`. Slash commands, `$skills` and short follow-ups are skipped, because the router never classifies them, and so are subagents, `codex exec` runs and orchestrators. A session's first prompt is kept however short, because it picks the model. `just tune 500` uses only your newest 500.
2. **Label.** Opus 5.5 at xhigh decides which model and effort each prompt needed, given both models' Artificial Analysis scores and response times at every effort, and is asked for the best answer without overthinking, not the cheapest.
3. **Train.** It trains a LoRA adapter on 70% of your sessions, on CPU, or on `TUNE_HOST` if set. Sessions are split whole, so related prompts never land on both sides. Effort is learned from every prompt. The model is learned only from each session's first prompt, labelled with what the whole session needed: a session that opens with a typo fix and turns into design work counts as Opus, because the model picked on that first prompt has to last the session.
4. **Gate.** On the other 30%, the adapter must beat both the score and any adapter already installed on effort, and pick the model no worse than the score on those sessions' first prompts. Only then does it install to `~/.cache/magic-router/tuned` and restart the daemon; otherwise nothing changes.

Results from the author's history (3,344 prompts over six months), scored on 972 held-out prompts:

| | Matches label | Mean levels off |
|---|---|---|
| Adapter trained on 3,344 prompts | **66%** | **0.36** |
| Adapter trained on 404 Claude Code prompts | 54% | 0.49 |
| Always `medium` | 44% | 0.59 |
| The default score | 40% | 0.69 |

Contrast training was tried and dropped. Opus 5.5 at xhigh rewrote each of 2,538 training prompts into the nearest prompt that needed the next effort level up or down, and the adapter trained on both. On the same 1,024 held-out prompts it matched 66% of labels and was 0.35 levels off on average, against 65% and 0.37 without contrasts, while the rewrites cost $15.94 and training took twice as long. The adapter already gets about as close to the labels as they allow, so more labelled data or better labels is the place to look, not synthetic pairs.

Labelling cost $10.33 at API prices. Training took about 15 minutes on an RTX 4080; on CPU it runs about 15 minutes per 400 prompts. Your numbers will differ.

With an adapter installed, the daemon answers effort and model from it, which adds a second pass of about 0.1 s. The model is still picked only on a session's first prompt. An adapter trained before model labelling existed answers effort only; retune to get both. To undo, delete `~/.cache/magic-router/tuned` and run `scripts/gliner.sh stop`.

<details>
<summary><b>What tuning sends and stores</b></summary>
<br>

- Labelling sends 25 prompts per `claude -p` call, with no tools, no settings and no saved session.
- Each label and a one-line reason go to `~/.cache/magic-router/tune/labels.jsonl`. A rerun only labels new prompts, and any labelled under an older labeller prompt. Skim them there.
- With `TUNE_HOST`, training and scoring run on that host (it needs an NVIDIA GPU, uv and rsync), and the prompts sent there are deleted afterwards.

</details>

<details>
<summary><b>Refreshing the benchmarks</b>: <code>just benchmarks</code></summary>
<br>

`just benchmarks` pulls the [Artificial Analysis](https://artificialanalysis.ai/api) evals for Sonnet 5.5, Opus 5.5 and Fable 5.1 at every effort level, prints them as a table, and caches the raw JSON in `~/.cache/magic-router`. With no arguments it also **rewrites `TASK_BIAS` in `hooks/route.ts`** from Opus's per-family lead over Sonnet; model names as arguments only filter the printout. Put `AA_API_KEY=...` in a git-ignored `.env`, or export it.

</details>

<details id="against-hosted-decision-models">
<summary><b>Against hosted decision models</b>: <code>just bench</code></summary>
<br>

`just bench` scores the trained adapter against hosted decision models on the same 972 held-out prompts, without retraining, sending each prompt's first 2,000 characters to the provider. `OPENROUTER_KEY` adds [Jev](https://openrouter.ai/typesafe/jev-1.13) (TypeSafe) and `FASTINO_API_KEY` adds GLiDE. Jev got the labeller's definition of each level, and was asked for the effort directly, both as a choice among the five levels and as a score on the ordered scale. A third run put Jev's answers to the router's own task and signal questions through the score's weights.

| | Matches label | Mean levels off |
|---|---|---|
| Tuned adapter | **66%** | **0.36** |
| Jev, effort as a score | 63% | 0.42 |
| Jev, effort as a choice | 56% | 0.53 |
| Always `medium` | 44% | 0.59 |
| The default score | 40% | 0.69 |
| Jev through the score's weights | 27% | 1.06 |

Asked directly, Jev comes within 3 points of the adapter without seeing any of your prompts. The adapter stays ahead, though it was trained on labels from the same labeller, so the margin favours it. It also runs locally, while Jev is a hosted call (median 234 ms, $0.049 for all 972 prompts) that sends your prompts to a third party. Through the score's weights Jev does worse than always guessing `medium`, because those weights were set for GLiNER, so it can't drop into the router as is.

</details>

## Troubleshooting

| Symptom | Fix |
|---|---|
| Band says `classifier unreachable` | Run `scripts/gliner.sh start` and read the log it names. On a first run, it's usually still downloading. |
| Daemon never starts from Claude Code, but `gliner.sh start` works | Read `~/.cache/magic-router/classifier.log`: the uv install or a torch download likely failed (offline, proxy). |
| Band says `stood aside` | You picked a model or effort by hand (`/model`, `/effort`). `/clear` to let the router pick again. |
| Port 8765 is taken by something else | Set `ROUTER_PORT` to a free port for Claude Code and the shell you run `gliner.sh` from, then restart Claude Code. |

## Updating and uninstalling

A new version ships with every merge to `main`. Third-party marketplaces don't auto-update by default, so turn it on once: run `/plugin`, open **Marketplaces**, select **magic-router** and choose **Enable auto-update**. Claude Code then pulls new versions at startup and asks you to restart or run `/reload-plugins`. To update by hand:

```
/plugin marketplace update magic-router
/plugin update magic-router@magic-router
```

To uninstall:

```
/plugin uninstall magic-router@magic-router
/plugin marketplace remove magic-router
```

Then stop the daemon (`pkill -f server/classifier.py`) and, to reclaim the disk, delete `~/.cache/magic-router` and `~/.cache/huggingface/hub/models--fastino--GLiNER2.5-Decide`.

## Contributing

Issues and pull requests are welcome. [`AGENTS.md`](AGENTS.md) covers the layout, the gotchas, and the conventions for both humans and coding agents. Before opening a PR, run what CI runs:

```sh
claude plugin validate .
claude plugin test .                           # routing, stickiness, stand-aside, band
uv run --script server/classifier.py --check   # classifier self-check
shellcheck scripts/*.sh
```

## Acknowledgements

[GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) by Fastino does the classifying. [Artificial Analysis](https://artificialanalysis.ai) supplies the evals that set the task bias and inform the labeller.

## License

[MIT](LICENSE)
