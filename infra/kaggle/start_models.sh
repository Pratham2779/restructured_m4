#!/usr/bin/env bash
set -e

echo "=== Starting Ollama on Kaggle Environment ==="
export OLLAMA_MODELS=${OLLAMA_MODELS:-"/tmp/ollama_models"}
mkdir -p "$OLLAMA_MODELS"

pkill -f "ollama serve" || true
nohup ollama serve > /tmp/ollama.log 2>&1 &
echo "Ollama running in background. Logs at /tmp/ollama.log"

for i in {1..30}; do
  if curl -s http://localhost:11434/ > /dev/null 2>&1; then
    echo "Ollama ready!"
    exit 0
  fi
  sleep 1
done

echo "Ollama startup timed out."
cat /tmp/ollama.log
exit 1
