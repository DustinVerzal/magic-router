# Loads AA_API_KEY from a git-ignored .env if present (or take it from your shell environment).
set dotenv-load

# Pull Artificial Analysis benchmarks; args filter model names.
benchmarks *models:
    uv run --script scripts/benchmarks.py {{models}}

# Routing tests.
test:
    claude plugin test .
