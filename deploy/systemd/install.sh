#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RECOVER="$SCRIPT_DIR/chatbot-recover.sh"

echo "==> Removing old services..."
for svc in wefix-all stop-ngrok-on-suspend; do
  if systemctl list-unit-files "${svc}.service" &>/dev/null; then
    systemctl stop "${svc}.service" 2>/dev/null || true
    systemctl disable "${svc}.service" 2>/dev/null || true
    rm -f "/etc/systemd/system/${svc}.service"
    echo "    Removed ${svc}.service"
  fi
done

echo "==> Installing scripts..."
chmod +x \
  "$SCRIPT_DIR/chatbot-start.sh" \
  "$SCRIPT_DIR/chatbot-recover.sh" \
  "$SCRIPT_DIR/chatbot-watchdog.sh" \
  "$SCRIPT_DIR/chatbot-resume.sh" \
  "$SCRIPT_DIR/wait-for-gateway.sh"

echo "==> Installing systemd units..."
cp "$SCRIPT_DIR/chatbot.service" /etc/systemd/system/chatbot.service
cp "$SCRIPT_DIR/chatbot-ngrok.service" /etc/systemd/system/chatbot-ngrok.service
cp "$SCRIPT_DIR/chatbot-resume.service" /etc/systemd/system/chatbot-resume.service
cp "$SCRIPT_DIR/chatbot-watchdog.service" /etc/systemd/system/chatbot-watchdog.service
cp "$SCRIPT_DIR/chatbot-watchdog.timer" /etc/systemd/system/chatbot-watchdog.timer

echo "==> Installing sleep hook (post-wake recover)..."
mkdir -p /etc/systemd/system-sleep
install -m 755 "$SCRIPT_DIR/chatbot-resume.sh" /etc/systemd/system-sleep/chatbot-resume

echo "==> Installing no-suspend policy (lid close / idle)..."
mkdir -p /etc/systemd/logind.d
cp "$SCRIPT_DIR/99-chatbot-no-suspend.conf" /etc/systemd/logind.d/99-chatbot-no-suspend.conf
systemctl restart systemd-logind.service 2>/dev/null || true

systemctl --user stop chatbot-ngrok.service 2>/dev/null || true
systemctl --user disable chatbot-ngrok.service 2>/dev/null || true

systemctl daemon-reload

echo "==> Enabling services..."
systemctl enable chatbot.service
systemctl enable chatbot-ngrok.service
systemctl enable chatbot-resume.service
systemctl enable chatbot-watchdog.timer

echo "==> Starting stack + watchdog..."
systemctl start chatbot-watchdog.timer
bash "$RECOVER"

echo ""
echo "Done. Status:"
systemctl is-active chatbot.service chatbot-ngrok.service chatbot-watchdog.timer
systemctl status chatbot-ngrok.service --no-pager -l | tail -5

echo ""
echo "Tunnel URLs:"
sleep 2
curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
for t in d.get('tunnels', []):
    print(' ', t.get('name','?'), '->', t.get('public_url','?'))
" 2>/dev/null || echo "  (check http://localhost:4040)"

echo ""
echo "Watchdog runs every 90s — auto-fixes offline ngrok after suspend."
echo "Manual recover: bash $RECOVER"
echo ""
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
  echo "Skipping monitor alerts (run as your user: bash $SCRIPT_DIR/../monitor/install.sh)"
else
  echo "Installing client activity alerts..."
  bash "$SCRIPT_DIR/../monitor/install.sh" || true
fi
