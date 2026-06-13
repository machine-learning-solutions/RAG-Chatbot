#!/usr/bin/env bash
# Keep chatbot + ngrok running: block suspend, allow screen off / lock.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="${SUDO_USER:-${USER:-jadaboawwad}}"

echo "==> Enabling chatbot host mode (no system suspend)..."

mkdir -p /etc/systemd/logind.d
cp "$SCRIPT_DIR/99-chatbot-no-suspend.conf" /etc/systemd/logind.d/99-chatbot-no-suspend.conf
# Do NOT restart logind here — it logs every user out to the login screen.
# systemd-inhibit (below) blocks suspend immediately; logind.conf applies after reboot.

cp "$SCRIPT_DIR/chatbot-host-mode.service" /etc/systemd/system/chatbot-host-mode.service
systemctl daemon-reload
systemctl enable --now chatbot-host-mode.service

if command -v gsettings >/dev/null && id "$RUN_USER" &>/dev/null; then
  echo "==> GNOME: disable auto-suspend (screen may still turn off)..."
  sudo -u "$RUN_USER" -H dbus-run-session gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
  sudo -u "$RUN_USER" -H dbus-run-session gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true
fi

echo ""
echo "Host mode ON."
echo "  - Chatbot + ngrok keep running"
echo "  - Screen lock / blank: OK"
echo "  - System Suspend (menu): BLOCKED while host mode is on"
echo "  - Lid-close rules: fully active after next reboot (logind not restarted)"
echo ""
echo "To suspend the laptop later:"
echo "  sudo bash $SCRIPT_DIR/disable-host-mode.sh"
systemctl is-active chatbot-host-mode.service
