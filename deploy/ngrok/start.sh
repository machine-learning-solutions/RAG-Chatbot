#!/usr/bin/env bash
# Tunnel Streamlit on port 8501.
#   bash deploy/ngrok/start.sh
#
# ═══════════════════════════════════════════════════════════════════
# ngrok-skip-browser-warning — READ THIS
# ═══════════════════════════════════════════════════════════════════
#
# CORRECT: header on the request *to ngrok* (browser → ngrok edge):
#   ngrok-skip-browser-warning: 1
#
# WRONG (does NOT skip the browser warning page):
#   ngrok http 8501 --request-header-add "ngrok-skip-browser-warning: 1"
#     ↑ adds header on ngrok → localhost only; ngrok cloud never sees it
#
#   https://xxx.ngrok-free.dev?ngrok-skip-browser-warning=1
#     ↑ query string is NOT supported by ngrok docs
#
# Browser / iPhone / iframe — pick one:
#   1. Reverse proxy on YOUR domain → ngrok with header (portfolio-proxy.nginx.conf)
#   2. Browser extension (ModHeader) on desktop Chrome
#   3. Click "Visit Site" once (direct Safari, not iframe)
#   4. Paid ngrok plan (no warning)
#
# Docs: https://ngrok.com/docs/pricing-limits/free-plan-limits
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

PORT="${PORT:-9080}"

if ! command -v ngrok >/dev/null 2>&1; then
  echo "Install ngrok: https://ngrok.com/download" >&2
  exit 1
fi

echo "Starting tunnel → http://localhost:${PORT}"
echo "Gateway :9080 → /api (FastAPI) + / (Streamlit). Portfolio: ?portfolio=true"
echo ""
echo "Verify skip header (API only, not browser navigation):"
echo "  curl -sI -H 'ngrok-skip-browser-warning: 1' https://YOUR.ngrok-free.dev/ | head -5"
echo ""

exec ngrok http "$PORT" --host-header=rewrite
