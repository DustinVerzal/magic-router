# Loads AA_API_KEY from a git-ignored .env if present (or take it from your shell environment).
set dotenv-load

# Pull Artificial Analysis benchmarks and retune TASK_BIAS in hooks/route.ts (run when new models ship). Args just filter the printout.
benchmarks *models:
    uv run --script scripts/benchmarks.py {{models}}

# Routing tests.
test:
    claude plugin test .
