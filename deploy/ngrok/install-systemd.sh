#!/usr/bin/env bash
# Install ngrok as a systemd USER service (survives logout with linger).
#   bash deploy/ngrok/install-systemd.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
UNIT_SRC="$ROOT/deploy/ngrok/chatbot-ngrok.service"
UNIT_DST="$HOME/.config/systemd/user/chatbot-ngrok.service"

mkdir -p "$HOME/.config/systemd/user"
cp "$UNIT_SRC" "$UNIT_DST"

systemctl --user daemon-reload
systemctl --user enable chatbot-ngrok.service
systemctl --user restart chatbot-ngrok.service

if ! loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q yes; then
  echo ""
  echo "Enable linger so ngrok starts at boot (no login required):"
  echo "  sudo loginctl enable-linger $USER"
fi

echo ""
echo "ngrok user service installed."
systemctl --user status chatbot-ngrok.service --no-pager || true
echo ""
echo "Tunnel URL:"
sleep 2
curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
d = json.load(sys.stdin)
for t in d.get('tunnels', []):
    if t.get('proto') == 'https':
        print(' ', t['public_url'])
" 2>/dev/null || echo "  (open http://localhost:4040 if API not ready yet)"
