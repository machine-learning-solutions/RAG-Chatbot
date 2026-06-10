#!/usr/bin/env bash
# Watch chatbot Docker CPU (mainly Ollama) and alert on Ubuntu when a visitor
# is likely chatting. Desktop notification + loud sound.
set -euo pipefail

COMPOSE_PROJECT="${COMPOSE_PROJECT:-chatbot}"
INTERVAL_SEC="${INTERVAL_SEC:-5}"
SUSTAINED_ROUNDS="${SUSTAINED_ROUNDS:-2}"
COOLDOWN_SEC="${COOLDOWN_SEC:-60}"
LOG_LOOKBACK_SEC="${LOG_LOOKBACK_SEC:-8}"

# CPU % thresholds (docker stats CPUPerc)
OLLAMA_CPU_THRESHOLD="${OLLAMA_CPU_THRESHOLD:-20}"
BACKEND_CPU_THRESHOLD="${BACKEND_CPU_THRESHOLD:-35}"
COMBINED_CPU_THRESHOLD="${COMBINED_CPU_THRESHOLD:-25}"

HEARTBEAT_SEC="${HEARTBEAT_SEC:-300}"

NOTIFY_TITLE="${NOTIFY_TITLE:-Portfolio Chatbot}"
NOTIFY_BODY="${NOTIFY_BODY:-Possible visitor — chatbot is processing a request.}"

HOT_DETAIL=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

container_names() {
  docker ps --filter "name=${COMPOSE_PROJECT}-" --format '{{.Names}}' 2>/dev/null \
    | grep -E 'ollama|backend|gateway' || true
}

cpu_for() {
  local name="$1"
  docker stats --no-stream --format '{{.CPUPerc}}' "$name" 2>/dev/null \
    | tr -d ' %' || echo "0"
}

float_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit (a + 0 >= b + 0) ? 0 : 1 }'
}

backend_container() {
  docker ps --filter "name=${COMPOSE_PROJECT}-backend" --format '{{.Names}}' 2>/dev/null \
    | head -1
}

recent_chat_request() {
  local backend
  backend="$(backend_container)"
  [[ -z "$backend" ]] && return 1

  docker logs --since "${LOG_LOOKBACK_SEC}s" "$backend" 2>&1 \
    | grep -qE 'POST /api/chat' || return 1
  return 0
}

play_alert_sound() {
  local sound=""
  for candidate in \
    "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga" \
    "/usr/share/sounds/freedesktop/stereo/phone-incoming-call.oga" \
    "/usr/share/sounds/ubuntu/stereo/dialog-warning.ogg" \
    "/usr/share/sounds/gnome/default/alerts/bark.ogg"; do
    if [[ -f "$candidate" ]]; then
      sound="$candidate"
      break
    fi
  done

  if command -v paplay >/dev/null 2>&1 && [[ -n "$sound" ]]; then
    for _ in 1 2 3; do
      paplay "$sound" >/dev/null 2>&1 &
    done
    wait
    return
  fi

  if command -v canberra-gtk-play >/dev/null 2>&1; then
    canberra-gtk-play -i alarm-clock-elapsed -d "$NOTIFY_TITLE" >/dev/null 2>&1 &
    canberra-gtk-play -i alarm-clock-elapsed -d "$NOTIFY_TITLE" >/dev/null 2>&1 &
    wait
    return
  fi

  if command -v speaker-test >/dev/null 2>&1; then
    speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 &
    wait
  fi
}

send_desktop_alert() {
  local body="$1"
  if ! command -v notify-send >/dev/null 2>&1; then
    log "notify-send not found; alert body: $body"
    return
  fi

  notify-send \
    -u critical \
    -a "chatbot-client-alert" \
    -i dialog-information \
    -t 15000 \
    "$NOTIFY_TITLE" \
    "$body" 2>/dev/null || \
    notify-send -u critical "$NOTIFY_TITLE" "$body" 2>/dev/null || true
}

is_hot() {
  local ollama_cpu=0 backend_cpu=0 combined=0
  local name cpu

  HOT_DETAIL=""

  if recent_chat_request; then
    HOT_DETAIL="Visitor sent a chat message."
    log "hot: POST /api/chat detected in backend logs"
    return 0
  fi

  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    cpu="$(cpu_for "$name")"
    combined="$(awk -v a="$combined" -v b="$cpu" 'BEGIN { print a + b }')"

    if [[ "$name" == *ollama* ]]; then
      ollama_cpu="$cpu"
    elif [[ "$name" == *backend* ]]; then
      backend_cpu="$cpu"
    fi
  done < <(container_names)

  if float_ge "$ollama_cpu" "$OLLAMA_CPU_THRESHOLD"; then
    HOT_DETAIL="Ollama CPU at ${ollama_cpu}%."
    log "hot: ollama CPU ${ollama_cpu}% (threshold ${OLLAMA_CPU_THRESHOLD}%)"
    return 0
  fi

  if float_ge "$backend_cpu" "$BACKEND_CPU_THRESHOLD"; then
    HOT_DETAIL="Backend CPU at ${backend_cpu}%."
    log "hot: backend CPU ${backend_cpu}% (threshold ${BACKEND_CPU_THRESHOLD}%)"
    return 0
  fi

  if float_ge "$combined" "$COMBINED_CPU_THRESHOLD"; then
    HOT_DETAIL="Combined chatbot CPU at ${combined}%."
    log "hot: combined CPU ${combined}% (threshold ${COMBINED_CPU_THRESHOLD}%)"
    return 0
  fi

  return 1
}

trigger_alert() {
  local detail="$1"
  log "ALERT: $detail"
  send_desktop_alert "${NOTIFY_BODY}
${detail}"
  play_alert_sound &
}

test_alert() {
  log "TEST: firing desktop notification and alert sound"
  trigger_alert "This is a test alert — chatbot monitor is working."
  log "TEST: done"
}

main() {
  if [[ "${1:-}" == "--test" ]]; then
    test_alert
    return
  fi

  local hot_streak=0 last_alert=0 now last_heartbeat=0

  log "monitor started (runs forever at interval=${INTERVAL_SEC}s, cooldown=${COOLDOWN_SEC}s)"
  log "triggers: POST /api/chat logs + ollama>=${OLLAMA_CPU_THRESHOLD}% backend>=${BACKEND_CPU_THRESHOLD}% combined>=${COMBINED_CPU_THRESHOLD}%"

  while true; do
    now=$(date +%s)

    if is_hot; then
      hot_streak=$((hot_streak + 1))

      if (( hot_streak >= SUSTAINED_ROUNDS )); then
        if (( now - last_alert >= COOLDOWN_SEC )); then
          trigger_alert "${HOT_DETAIL:-Chatbot activity detected.}"
          last_alert=$now
        else
          log "hot (alert suppressed — cooldown ${COOLDOWN_SEC}s)"
        fi
        hot_streak=0
      fi
    else
      hot_streak=0
    fi

    if (( now - last_heartbeat >= HEARTBEAT_SEC )); then
      log "listening… (next alert allowed in $(( COOLDOWN_SEC - (now - last_alert) ))s)"
      last_heartbeat=$now
    fi

    sleep "$INTERVAL_SEC" || sleep 5
  done
}

main "$@"
