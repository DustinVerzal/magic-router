# Loads AA_API_KEY from a git-ignored .env if present (or take it from your shell environment).
set dotenv-load

# Pull Artificial Analysis benchmarks and retune TASK_BIAS in hooks/route.ts (run when new models ship). Args just filter the printout.
benchmarks *models:
    uv run --script scripts/benchmarks.py {{models}}

# Label your Claude Code and Codex prompts with Opus and train the classifier's model and effort answers on them (run `just benchmarks` first: its AA scores go into the labeller's prompt) (scripts/tune.py). Arg: cap on how many newest prompts, default all.
tune *limit:
    uv run --script scripts/tune.py {{limit}}

# Benchmark the adapter from `just tune` against other decision models on the held-out prompts, without retraining. OPENROUTER_KEY adds Jev, FASTINO_API_KEY adds GLiDE (each gets those prompts); `just bench clef` adds Clef-flash on your GPU (TUNE_HOST's, if set).
bench *models:
    uv run --script scripts/tune.py --bench {{models}}

# Routing tests.
test:
    claude plugin test .
