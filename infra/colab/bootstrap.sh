#!/usr/bin/env bash
set -e

echo "============================================="
echo "  M4 Benchmark Environment Setup (Google Colab)"
echo "============================================="

# 1. Install system utilities
echo "--> Installing system packages..."
sudo apt-get update -qq && sudo apt-get install -y -qq curl pciutils procps

# 2. Check GPU presence
echo "--> Checking GPU availability:"
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "Warning: No GPU detected. Execution will proceed in CPU mode."
fi

# 3. Install Ollama
if ! command -v ollama &> /dev/null; then
  echo "--> Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# 4. Start Ollama daemon
echo "--> Starting Ollama server..."
bash "$(dirname "$0")/start_models.sh"

# 5. Pull default open-source models
MODEL_NAME=${M4_MODEL:-"qwen2.5:7b-instruct"}
EMBEDDER_NAME=${M4_EMBEDDER:-"all-minilm"}

echo "--> Pulling chat model: $MODEL_NAME"
ollama pull "$MODEL_NAME"

echo "--> Pulling embedding model: $EMBEDDER_NAME"
ollama pull "$EMBEDDER_NAME" || ollama pull nomic-embed-text || echo "Embedding model ready."

# 6. Install Python packages
echo "--> Installing m4 and m4bench packages..."
pip install --upgrade pip
pip install -e packages/m4 -e packages/m4bench

# 7. Hermetic smoke test
echo "--> Running verification smoke test..."
m4bench run locomo --stage smoke --embedder "$EMBEDDER_NAME"

echo "============================================="
echo "  Bootstrap Complete! Ready for evaluation.  "
echo "============================================="
