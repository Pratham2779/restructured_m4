#!/usr/bin/env bash
set -e

echo "=== Bootstrapping M4 Environment on Kaggle ==="

# Check disk and GPU
df -h /
nvidia-smi || echo "Running on CPU"

# Install Ollama
if ! command -v ollama &> /dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama
bash "$(dirname "$0")/start_models.sh"

# Pull model
MODEL=${M4_MODEL:-"qwen2.5:7b-instruct"}
echo "Pulling $MODEL into /tmp..."
export OLLAMA_MODELS="/tmp/ollama_models"
ollama pull "$MODEL"
ollama pull bge-small-en-v1.5 || ollama pull all-minilm || true

# Install Python packages
pip install --upgrade pip
pip install -e packages/m4 -e packages/m4bench

# Run smoke test
m4bench run locomo --stage smoke
