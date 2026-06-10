#!/usr/bin/env bash
# Install user systemd service for chatbot client alerts.
# Runs at boot (with linger) and listens continuously.
#   bash deploy/monitor/install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
UNIT_SRC="$ROOT/deploy/monitor/chatbot-client-alert.service"
UNIT_DST="$HOME/.config/systemd/user/chatbot-client-alert.service"
DESKTOP_SRC="$ROOT/deploy/monitor/chatbot-client-alert.desktop"
DESKTOP_DST="$HOME/.config/autostart/chatbot-client-alert.desktop"
SCRIPT="$ROOT/deploy/monitor/chatbot-client-alert.sh"

chmod +x "$SCRIPT"

mkdir -p "$HOME/.config/systemd/user"
mkdir -p "$HOME/.config/autostart"

# Patch home path into unit file for this machine.
sed "s|%h|$HOME|g; s|%U|$(id -u)|g" "$UNIT_SRC" > "$UNIT_DST"
cp "$DESKTOP_SRC" "$DESKTOP_DST"

# Boot without login: user systemd must linger.
if ! loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q yes; then
  echo "==> Enabling linger (start user services at boot)..."
  if sudo loginctl enable-linger "$USER"; then
    echo "    Linger enabled for $USER"
  else
    echo "    Run manually: sudo loginctl enable-linger $USER"
  fi
fi

systemctl --user daemon-reload
systemctl --user enable chatbot-client-alert.service
systemctl --user restart chatbot-client-alert.service

echo ""
echo "Chatbot client alert monitor installed."
echo "  - Starts at boot (linger + enabled user service)"
echo "  - Restarts automatically if it crashes (Restart=always)"
echo "  - Listens forever; cooldown only limits repeat alerts"
echo ""
systemctl --user status chatbot-client-alert.service --no-pager || true
echo ""
echo "Test: bash deploy/monitor/chatbot-client-alert.sh --test"
echo "Logs: journalctl --user -u chatbot-client-alert -f"
