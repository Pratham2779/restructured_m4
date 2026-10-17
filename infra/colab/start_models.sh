#!/usr/bin/env bash
set -e

echo "=== Starting Ollama Service with Max GPU Parallelism ==="
pkill -f "ollama serve" || true

# Maximize Ollama GPU resource usage
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:/opt/bin:${PATH:-}"
export CUDA_VISIBLE_DEVICES="0,1"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-11}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_KEEP_ALIVE="24h"

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
