# AGENTS.md

A Claude Code plugin (`magic-router`) that picks Sonnet or Opus per session and an effort level per prompt, from a local GLiNER2.5 classifier daemon. README.md covers what it does for users; this file covers what you need to change it safely.

## Layout

- `hooks/route.ts`: all routing policy (labels, weights, thresholds). Pure functions, no I/O.
- `hooks/register.tsx`: the plugin's hooks (daemon start, prompt classification, request rewrite, the band UI). State lives in atoms typed in `types/index.d.ts`.
- `server/classifier.py`: the daemon (uv inline-script deps, stdlib HTTP server on `127.0.0.1:8765`). It only answers questions; it holds no policy.
- `scripts/gliner.sh` manages the daemon; `scripts/tune.py` trains a LoRA effort adapter; `scripts/benchmarks.py` pulls Artificial Analysis evals.

## Commands

```sh
claude plugin validate .                          # manifests
claude plugin test .                              # tests/router.test.tsx (fast, no daemon needed)
uv run --script server/classifier.py --check      # classifier self-check (installs torch on first run)
shellcheck scripts/*.sh
```

CI (`.github/workflows/ci.yml`) runs validate, test, `py_compile` on the server, shellcheck and `jq empty hooks/hooks.json`. Run the matching ones before calling a change done.

Don't run `just tune`, `just bench` or `just benchmarks` unless asked: they cost money (`claude -p` labelling, paid APIs), send prompts to third parties, read the user's private transcripts, or rewrite source.

## Gotchas

- `scripts/tune.py` regex-parses `hooks/route.ts` for `TASKS`, `SIGNALS`, `SIGNAL_WEIGHTS`, `TASK_BIAS`, `EFFORTS` and `FOLLOW_UP_WORDS`. Keep those as plain literals in their current shape (one `key: value` per line, `[number, 'effort']` tuples), or tuning crashes or silently drops entries.
- `scripts/benchmarks.py` (no args) rewrites the `TASK_BIAS` block, from its `// Where Opus` comment to the closing `}`. Hand edits there get overwritten; change the script instead.
- The port is `ROUTER_PORT` (default 8765), read by `daemon()` in `hooks/register.tsx`, `PORT` in `scripts/gliner.sh` and `server/classifier.py`. Keep the three defaults in step.
- The model is picked once per session on purpose: switching mid-conversation forfeits the prompt cache. Don't make the model re-route per prompt.
- `turn.step` must leave subagent requests (`e.agentId !== undefined`) untouched, and must stand aside once `/model` or `/effort` changes the baseline. Tests cover both; keep them passing.
- `tests/router.test.tsx` fixtures (`FIX`) are recorded daemon outputs. If you change weights or thresholds, update the expected routes in the first test deliberately, not by copying whatever the new code returns.
- The daemon is shared across sessions and outlives them. After editing `server/classifier.py`, restart it (`scripts/gliner.sh stop && scripts/gliner.sh start`) before trusting live behaviour.

## Conventions

- Comments explain why, in full sentences; match the existing density. Deliberate shortcuts carry a `ponytail:` comment naming the limit and the upgrade path.
- Prefer stdlib and what's already installed. No new runtime dependencies in the hooks; the daemon's deps stay in its inline `# /// script` block.
- Keep the README in sync when user-visible behaviour, commands or numbers change.

## Git and releases

- Branch off `main`, open a PR; `main` requires the `check` CI job and resolved review threads.
- Commit subjects: imperative, sentence case, no prefix (e.g. "Fix shellcheck findings in gliner.sh"). Merging to `main` bumps the version automatically: `feat:` → minor, `!:`/`BREAKING` → major, anything else → patch.
- Never edit `version` in `.claude-plugin/plugin.json` by hand; the release workflow owns it.
