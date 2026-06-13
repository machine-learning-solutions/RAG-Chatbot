#!/usr/bin/env bash
# Allow normal suspend again (chatbot will stop until wake + recover).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="${SUDO_USER:-${USER:-jadaboawwad}}"

echo "==> Disabling chatbot host mode..."
systemctl disable --now chatbot-host-mode.service 2>/dev/null || true
rm -f /etc/systemd/logind.d/99-chatbot-no-suspend.conf
# Do not restart logind — avoids forced logout. Reboot to clear lid-close overrides.

if command -v gsettings >/dev/null && id "$RUN_USER" &>/dev/null; then
  sudo -u "$RUN_USER" -H dbus-run-session gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'suspend' 2>/dev/null || true
  sudo -u "$RUN_USER" -H dbus-run-session gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'suspend' 2>/dev/null || true
fi

echo "Host mode OFF — you can Suspend from the power menu."
echo "After wake, chatbot recovers via watchdog (≤90s) or: bash $SCRIPT_DIR/chatbot-recover.sh"
