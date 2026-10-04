#!/bin/sh
# Install, run and inspect the GLiNER2.5 classifier daemon that magic-router talks to.
# The mod starts the daemon by itself; this script is for warming it up ahead of time and poking at it.
#
#   scripts/gliner.sh setup    install uv if missing, install torch + gliner2, download the weights, start
#   scripts/gliner.sh start    start the daemon (if it isn't up) and wait until the model is loaded
#   scripts/gliner.sh stop     stop the daemon
#   scripts/gliner.sh status   print /health
#   scripts/gliner.sh check    run the classifier's offline self-check
#   scripts/gliner.sh logs     follow the daemon log
#   scripts/gliner.sh remote HOST   run the daemon on an ssh host (a GPU box) instead, through a tunnel (macOS)
#   scripts/gliner.sh local    run it here again
# After `remote`, setup/start/stop/check/logs act on that host, and start copies your tuned adapter there.
set -eu

# shellcheck disable=SC1007  # empty CDPATH is deliberate
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SERVER="$ROOT/server/classifier.py"
PORT=${ROUTER_PORT:-8765} # the same variable the daemon and the mod read
URL=http://127.0.0.1:$PORT
export ROUTER_PORT="$PORT"
LOG="$HOME/.cache/magic-router/classifier.log"
WAIT=${ROUTER_WAIT:-900} # seconds; the first run downloads ~1.7 GB of torch and weights

HOSTFILE="$HOME/.cache/magic-router/host" # written by `remote`
HOST=$(cat "$HOSTFILE" 2>/dev/null || true)
AGENT="$HOME/Library/LaunchAgents/dev.magic-router.tunnel.plist"

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

# Run this script's command on $HOST, from a copy of this checkout, with your tuned adapter (or none) copied there.
on_host() {
  ssh "$HOST" 'mkdir -p .cache/magic-router/repo'
  rsync -a --delete --exclude .git "$ROOT/" "$HOST:.cache/magic-router/repo/"
  if [ -d "$HOME/.cache/magic-router/tuned" ]; then
    rsync -a --delete "$HOME/.cache/magic-router/tuned/" "$HOST:.cache/magic-router/tuned/"
  else
    ssh "$HOST" 'rm -rf .cache/magic-router/tuned'
  fi
  # shellcheck disable=SC2029 # $1 is meant to expand locally
  ssh "$HOST" "ROUTER_PORT=$PORT sh .cache/magic-router/repo/scripts/gliner.sh $1"
}

# A launch agent keeps the daemon's port forwarded to $HOST's, so the mod reaches it at the same URL.
# ponytail: no ExitOnForwardFailure, since a DynamicForward in your ssh config may be held by another ssh. So if a
# session started a local daemon while $HOST was down, it keeps the port: stop it with `pkill -f server/classifier.py`.
tunnel() {
  mkdir -p "$(dirname "$AGENT")"
  cat >"$AGENT" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.magic-router.tunnel</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/ssh</string><string>-N</string><string>-o</string><string>BatchMode=yes</string>
    <string>-o</string><string>ServerAliveInterval=15</string><string>-o</string><string>ServerAliveCountMax=3</string>
    <string>-L</string><string>$PORT:127.0.0.1:$PORT</string><string>$HOST</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardErrorPath</key><string>$HOME/.cache/magic-router/tunnel.log</string>
</dict></plist>
EOF
  launchctl unload "$AGENT" 2>/dev/null || true
  launchctl load "$AGENT"
}

case "${1:-}" in
  setup | start | stop | check | logs) [ -z "$HOST" ] || { on_host "$1"; exit; } ;;
esac

case "${1:-}" in
  remote)
    [ -n "${2:-}" ] || { echo "usage: scripts/gliner.sh remote HOST" >&2; exit 2; }
    HOST=$2
    on_host setup # the first time, installs uv, torch and the weights there; then starts the daemon
    mkdir -p "$(dirname "$HOSTFILE")"
    echo "$HOST" >"$HOSTFILE"
    # it would hold the tunnel's port
    if pkill -f server/classifier.py; then echo "stopped the local daemon"; fi
    tunnel
    sleep 3
    echo "through the tunnel: $(health || echo 'not up yet; see ~/.cache/magic-router/tunnel.log')"
    ;;
  local)
    launchctl unload "$AGENT" 2>/dev/null || true
    rm -f "$AGENT" "$HOSTFILE"
    echo "the daemon runs here again: it starts with your next session, or now with scripts/gliner.sh start"
    ;;
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
  status) if health; then echo; else echo "not running"; exit 1; fi ;;
  check) need_uv; uv run --script "$SERVER" --check ;;
  logs) mkdir -p "$(dirname "$LOG")"; touch "$LOG"; tail -f "$LOG" ;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
