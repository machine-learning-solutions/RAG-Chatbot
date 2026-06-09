#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Removing old services..."
for svc in wefix-all stop-ngrok-on-suspend; do
  if systemctl list-unit-files "${svc}.service" &>/dev/null; then
    systemctl stop "${svc}.service" 2>/dev/null || true
    systemctl disable "${svc}.service" 2>/dev/null || true
    rm -f "/etc/systemd/system/${svc}.service"
    echo "    Removed ${svc}.service"
  fi
done

echo "==> Installing new services..."
cp "$SCRIPT_DIR/chatbot.service" /etc/systemd/system/chatbot.service
cp "$SCRIPT_DIR/chatbot-ngrok.service" /etc/systemd/system/chatbot-ngrok.service

systemctl daemon-reload

echo "==> Enabling services..."
systemctl enable chatbot.service
systemctl enable chatbot-ngrok.service

echo "==> Starting services..."
systemctl start chatbot.service
sleep 5
systemctl start chatbot-ngrok.service

echo ""
echo "Done. Status:"
systemctl status chatbot.service --no-pager -l | tail -10
systemctl status chatbot-ngrok.service --no-pager -l | tail -10

echo ""
echo "Tunnel URLs:"
sleep 3
curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
for t in d.get('tunnels', []):
    print(' ', t.get('name','?'), '->', t.get('public_url','?'))
" 2>/dev/null || echo "  (check http://localhost:4040)"
