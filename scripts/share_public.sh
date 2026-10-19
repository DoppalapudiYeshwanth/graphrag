#!/usr/bin/env bash
# Publish the app from THIS machine through a free Cloudflare quick tunnel.
#
#   - no cloud account, no payment method, nothing is deployed anywhere
#   - the public URL exists only while this script runs; Ctrl-C closes it
#   - the URL is random and changes every run
#
# Usage:  scripts/share_public.sh
# Tune:   DAILY_ASK_BUDGET=40 DAILY_CHUNK_BUDGET=80 PORT=8010 scripts/share_public.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8010}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-neo4j-graphrag}"
LOG_DIR="${TMPDIR:-/tmp}/graphrag-share"
mkdir -p "$LOG_DIR"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1  ($2)"; exit 1; }; }
need cloudflared "install with: brew install cloudflared"
need docker "install Docker Desktop"
[ -x venv/bin/uvicorn ] || { echo "Missing venv. Run: python3.10 -m venv venv && venv/bin/pip install -r requirements.txt"; exit 1; }

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Closing the public URL..."
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "Stopped. The app is no longer reachable from the internet."
}
trap cleanup EXIT INT TERM

# 1. Database (local Docker container; it is never exposed through the tunnel).
docker start "$NEO4J_CONTAINER" >/dev/null 2>&1 || { echo "Could not start container '$NEO4J_CONTAINER'"; exit 1; }

# 2. Public-safe settings. Questions and uploads are capped per day so the free Groq quota
#    cannot be drained; /docs is hidden; the programmatic /ask needs a random key.
export API_KEY="$(openssl rand -hex 24)"
export DOCS_ENABLED=false
export DAILY_ASK_BUDGET="${DAILY_ASK_BUDGET:-60}"
export DAILY_CHUNK_BUDGET="${DAILY_CHUNK_BUDGET:-100}"

# 3. The app, listening on localhost only. cloudflared connects from localhost, so trusting
#    proxy headers from 127.0.0.1 gives per-visitor rate limits the real client address.
venv/bin/uvicorn graphrag.api:app --host 127.0.0.1 --port "$PORT" \
  --proxy-headers --forwarded-allow-ips=127.0.0.1 >"$LOG_DIR/app.log" 2>&1 &
APP_PID=$!
PIDS+=("$APP_PID")

echo -n "Starting the app"
for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$APP_PID" 2>/dev/null || { echo; echo "The app stopped. Last log lines:"; tail -n 15 "$LOG_DIR/app.log"; exit 1; }
  echo -n "."; sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo; echo "The app did not become healthy. See $LOG_DIR/app.log"; exit 1; }
echo " ready"

# Remove visitor workspaces that have been idle past the retention period.
venv/bin/python -m scripts.cleanup_workspaces >/dev/null 2>&1 || true

# 4. Keep the Mac awake while sharing (released automatically when this script exits).
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ &
  PIDS+=($!)
fi

# 5. The tunnel.
cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG_DIR/tunnel.log" 2>&1 &
TUNNEL_PID=$!
PIDS+=("$TUNNEL_PID")

URL=""
for _ in $(seq 1 45); do
  URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" 2>/dev/null | head -n 1 || true)"
  [ -n "$URL" ] && break
  kill -0 "$TUNNEL_PID" 2>/dev/null || { echo "The tunnel stopped. Last log lines:"; tail -n 15 "$LOG_DIR/tunnel.log"; exit 1; }
  sleep 1
done
[ -n "$URL" ] || { echo "No tunnel URL after 45 seconds. See $LOG_DIR/tunnel.log"; exit 1; }

cat <<EOF

  ┌──────────────────────────────────────────────────────────────┐
   Public URL:  $URL
  └──────────────────────────────────────────────────────────────┘

  Live while this window stays open. Press Ctrl-C to close it.

  Daily caps:  $DAILY_ASK_BUDGET questions, $DAILY_CHUNK_BUDGET upload chunks (shared by all visitors)
  Privacy:     documents are private per browser; their text goes to Groq for analysis
  Logs:        $LOG_DIR/app.log   $LOG_DIR/tunnel.log

EOF

wait "$TUNNEL_PID"
