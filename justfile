# Loads AA_API_KEY from a git-ignored .env if present (or take it from your shell environment).
set dotenv-load

# Pull Artificial Analysis benchmarks and retune TASK_BIAS in hooks/route.ts (run when new models ship). Args just filter the printout.
benchmarks *models:
    uv run --script scripts/benchmarks.py {{models}}

# Label your newest prompts with Opus and train the classifier's effort answer on them (scripts/tune.py). Arg: how many prompts, default 500.
tune *limit:
    uv run --script scripts/tune.py {{limit}}

# Routing tests.
test:
    claude plugin test .
