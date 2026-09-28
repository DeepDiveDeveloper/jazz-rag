#!/usr/bin/env bash
#
# Starts the whole jazzbot stack in dependency order:
#
#   1. RAG server  (:8008)  python, Chroma + embedding model  -> wait for /health
#   2. API         (:3000)  node/tsx, SQLite + forwards to RAG -> wait for /api/health
#   3. UI          (:5173)  vite, proxies /api to :3000         -> wait for HTTP
#
# Each tier is started only after the previous one is actually reachable, and
# all three log lines are prefixed and interleaved. Ctrl-C (or a failure in any
# tier) stops everything.
#
# Env: RAG_PORT, API_PORT, UI_PORT, TIMEOUT, LLM_BASE_URL, LLM_MODEL, API_KEY
#
# Each port env var drives both the process and the readiness probe for that
# tier, so overriding any of them moves the whole stack consistently (the UI's
# /api proxy follows API_PORT via web/vite.config.ts).

set -Eeuo pipefail
set -m # background jobs get their own process group, so we can kill the tree

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAG_PORT="${RAG_PORT:-8008}"
API_PORT="${API_PORT:-3000}"
UI_PORT="${UI_PORT:-5173}"
TIMEOUT="${TIMEOUT:-180}"
HOST="127.0.0.1"
PY="$ROOT/.venv/bin/python"

RAG_URL="http://$HOST:$RAG_PORT"
API_URL="http://$HOST:$API_PORT"
UI_URL="http://$HOST:$UI_PORT"

# Passed through to the RAG server and the API.
export RAG_PORT API_PORT UI_PORT
export PORT="$API_PORT"          # the API reads PORT; API_PORT is this script's knob
export RAG_BASE_URL="${RAG_BASE_URL:-$RAG_URL}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434/v1}"
export LLM_MODEL="${LLM_MODEL:-tinyllama}"
export API_KEY="${API_KEY:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

# --- lifecycle ---------------------------------------------------------------

PIDS=()

cleanup() {
  local code=$?
  trap - INT TERM EXIT
  for pid in "${PIDS[@]:-}"; do
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  if [ "$code" -ne 0 ]; then
    echo "==> stopped (exit $code)"
  else
    echo "==> stopped"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- helpers -----------------------------------------------------------------

# start <name> <cmd...>  — run in its own process group, logs prefixed
start() {
  local name=$1
  shift
  echo "==> starting $name: $*"
  "$@" 2>&1 | sed -u "s/^/[$name] /" &
  PIDS+=("$!")
}

# wait_http <name> <url> <check-regexp>  — poll until the endpoint answers the check
wait_http() {
  local name=$1 url=$2 pattern=$3
  local start=$SECONDS body
  printf '==> waiting for %s ' "$name"
  while :; do
    if body=$(curl -fsS --max-time 3 "$url" 2>/dev/null) && printf '%s' "$body" | grep -Eq "$pattern"; then
      echo " up ($url)"
      return 0
    fi
    if (( SECONDS - start > TIMEOUT )); then
      echo " TIMEOUT after ${TIMEOUT}s ($url)"
      return 1
    fi
    if [ "${#PIDS[@]}" -gt 0 ] && ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
      echo " FAILED — $name exited during startup"
      return 1
    fi
    printf '.'
    sleep 0.5
  done
}

# --- preflight ---------------------------------------------------------------

command -v curl >/dev/null || fail "curl is required"
command -v npm >/dev/null || fail "npm is required"
[ -x "$PY" ] || fail "missing venv at $ROOT/.venv (see README: pip install -r requirements.txt)"
for dir in server web; do
  if [ ! -d "$ROOT/$dir/node_modules" ]; then
    echo "==> installing $dir dependencies"
    npm --prefix "$ROOT/$dir" install --no-fund --no-audit
  fi
done

# --- 1. RAG server (must be up before the API can reach it) -------------------

start rag "$PY" scripts/rag_server.py \
  --host "$HOST" --port "$RAG_PORT" \
  --base-url "$LLM_BASE_URL" --llm-model "$LLM_MODEL" \
  ${API_KEY:+--api-key "$API_KEY"}
wait_http "rag" "$RAG_URL/health" '"service"|"status"' || exit 1

# --- 2. API (only once the RAG server answers, so health reports it) ----------

start api npm --prefix "$ROOT/server" run dev
wait_http "api" "$API_URL/api/health" '"reachable":true' || exit 1

# --- 3. UI last: it is pure client code and proxies /api to the API ----------

start ui npm --prefix "$ROOT/web" run dev
wait_http "ui" "$UI_URL" '<html|<!doctype' || exit 1

cat <<EOF

==> stack ready
    UI    $UI_URL
    API   $API_URL/api/health
    RAG   $RAG_URL/health
    LLM   $LLM_BASE_URL (model: $LLM_MODEL)

    Ctrl-C stops all three.
EOF

wait
