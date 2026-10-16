#!/usr/bin/env bash
set -e

echo "=== Starting Ollama Service in Background ==="
pkill -f "ollama serve" || true
nohup ollama serve > /tmp/ollama.log 2>&1 &
PID=$!
echo "Ollama process launched with PID: $PID. Logs at /tmp/ollama.log"

echo "Waiting for Ollama to become ready at http://localhost:11434..."
for i in {1..30}; do
  if curl -s http://localhost:11434/ > /dev/null 2>&1; then
    echo "Ollama service is up and running!"
    exit 0
  fi
  sleep 1
done

echo "Error: Ollama did not start within 30 seconds."
cat /tmp/ollama.log
exit 1
