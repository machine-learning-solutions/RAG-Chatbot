#!/usr/bin/env bash
# Wait until nginx gateway accepts connections (post-boot / post-resume).
for _ in $(seq 1 90); do
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:9080/ 2>/dev/null; then
    exit 0
  fi
  if ss -tln 2>/dev/null | grep -q ':9080 '; then
    exit 0
  fi
  sleep 2
done
logger -t chatbot-ngrok "Gateway :9080 not ready after wait"
exit 1
