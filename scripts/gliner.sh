#!/bin/sh
# Install, run and inspect the GLiNER2.5 classifier daemon that model-router talks to.
# The mod starts the daemon by itself; this script is for warming it up ahead of time and poking at it.
#
#   scripts/gliner.sh setup    install uv if missing, install torch + gliner2, download the weights, start
#   scripts/gliner.sh start    start the daemon (if it isn't up) and wait until the model is loaded
#   scripts/gliner.sh stop     stop the daemon
#   scripts/gliner.sh status   print /health
#   scripts/gliner.sh check    run the classifier's offline self-check
#   scripts/gliner.sh logs     follow the daemon log
set -eu

# shellcheck disable=SC1007  # empty CDPATH is deliberate
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SERVER="$ROOT/server/classifier.py"
URL=http://127.0.0.1:8765 # ponytail: fixed, matches DAEMON in hooks/register.tsx
LOG="$HOME/.cache/model-router/classifier.log"
WAIT=${ROUTER_WAIT:-900} # seconds; the first run downloads ~1.7 GB of torch and weights

health() { curl -fsS -m 2 "$URL/health" 2>/dev/null; }

need_uv() {
  command -v uv >/dev/null 2>&1 && return
  [ -x "$HOME/.local/bin/uv" ] && PATH="$HOME/.local/bin:$PATH" && return
  echo "uv is not installed: run 'scripts/gliner.sh setup' or see https://docs.astral.sh/uv/" >&2
  exit 1
}

start() {
  need_uv
  if ! health >/dev/null; then
    mkdir -p "$(dirname "$LOG")"
    # Same command the mod runs at session start, so both share uv's cached environment.
    nohup uv run --script "$SERVER" </dev/null >>"$LOG" 2>&1 &
    echo "starting daemon (log: $LOG)"
  fi
  i=0
  until health | grep -q '"ready": true'; do
    i=$((i + 2))
    [ "$i" -gt "$WAIT" ] && { echo "not ready after ${WAIT}s; see $LOG" >&2; exit 1; }
    sleep 2
  done
  echo "ready: $(health)"
}

case "${1:-}" in
  setup)
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
      echo "installing uv (https://astral.sh/uv)"
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    need_uv
    echo "installing torch + gliner2 (first run only)"
    uv run --script "$SERVER" --check
    start
    if ! sh -lc 'command -v uv' >/dev/null 2>&1; then
      echo "note: add $(dirname "$(command -v uv)") to PATH in your shell profile so Claude Code can find uv"
    fi
    ;;
  start) start ;;
  # Any copy: a marketplace install runs the daemon from the plugin cache, not this checkout.
  stop) pkill -f server/classifier.py && echo stopped || echo "not running" ;;
  # shellcheck disable=SC2015  # echo cannot fail here
  status) health && echo || { echo "not running"; exit 1; } ;;
  check) need_uv; uv run --script "$SERVER" --check ;;
  logs) mkdir -p "$(dirname "$LOG")"; touch "$LOG"; tail -f "$LOG" ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
