#!/usr/bin/env bash
set -e

# Optional Cloudflare Tunnel for remote inspection/debugging from laptop
PORT=${1:-11434}
LOG_FILE="/tmp/cloudflared.log"

if ! command -v cloudflared &> /dev/null; then
  echo "Installing cloudflared binary..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
  chmod +x /tmp/cloudflared
  sudo mv /tmp/cloudflared /usr/local/bin/cloudflared
fi

echo "=== Starting Cloudflare Quick Tunnel on Port $PORT ==="
pkill -f "cloudflared" || true

if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
  echo "Starting named tunnel with provided token..."
  nohup cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN" > "$LOG_FILE" 2>&1 &
else
  echo "Starting quick tunnel without token..."
  nohup cloudflared tunnel --url "http://localhost:$PORT" > "$LOG_FILE" 2>&1 &
fi

echo "Waiting for tunnel URL in $LOG_FILE..."
sleep 3
grep -o "https://[a-zA-Z0-9.-]*\.trycloudflare\.com" "$LOG_FILE" | head -n 1 || echo "Tunnel started. Check $LOG_FILE for public URL."
