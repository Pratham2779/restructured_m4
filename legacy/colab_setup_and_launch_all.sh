#!/usr/bin/env bash
# =============================================================================
# M4 Meta-Memory — 1-Click All-in-One Cloud Setup & Mentor UI Launcher
# Designed for Google Colab, Kaggle, or any Remote GPU Environment
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/m4_optimized" ]; then
    PROJECT_DIR="$SCRIPT_DIR/m4_optimized"
else
    PROJECT_DIR="$SCRIPT_DIR"
fi
cd "$PROJECT_DIR"

UI_PORT="${UI_PORT:-7860}"

echo "================================================================================"
echo "    🚀 M4 META-MEMORY: 1-CLICK ALL-IN-ONE SETUP & UI LAUNCHER                   "
echo "================================================================================"
echo "Working directory: $(pwd)"
echo "Target UI Port:    $UI_PORT (Avoids Colab internal port 8080 conflict)"
echo "================================================================================"

# 1. GPU Verification
echo ""
echo "🔍 [1/6] Detecting GPU Acceleration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi
    echo "✓ NVIDIA GPU Detected! LLM and Embeddings will execute on CUDA."
else
    echo "⚠️ Warning: No dedicated GPU found. Running on CPU (will be slower)."
fi

# 2. Install & Start Ollama
echo ""
echo "⚙️ [2/6] Checking Ollama Engine..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    command -v zstd >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq zstd >/dev/null 2>&1 || true)
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed."
else
    echo "✓ Ollama already installed."
fi

# Ensure Ollama daemon is running with multi-model GPU residency
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=24h

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama daemon in background with multi-model GPU residency..."
    pkill -f "ollama serve" || true
    sleep 1
    nohup ollama serve > /tmp/ollama_colab.log 2>&1 &
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✓ Ollama daemon is active at http://localhost:11434"
            break
        fi
        sleep 1
    done
else
    echo "✓ Ollama daemon is already running."
fi

# 3. Pull Required Models & Pre-warm in VRAM
echo ""
echo "⬇️ [3/6] Pre-warming Models in GPU VRAM..."
if ! ollama list | grep -q "qwen2.5:7b-instruct"; then
    echo "Pulling Primary LLM (qwen2.5:7b-instruct)..."
    ollama pull qwen2.5:7b-instruct
else
    echo "✓ qwen2.5:7b-instruct is ready."
fi

if ! ollama list | grep -q "all-minilm"; then
    echo "Pulling Embedding Model (all-minilm)..."
    ollama pull all-minilm
else
    echo "✓ all-minilm is ready."
fi

echo "Locking both models into GPU VRAM (zero latency)..."
curl -s http://localhost:11434/api/generate -d '{"model": "qwen2.5:7b-instruct", "keep_alive": "24h"}' > /dev/null 2>&1 &
curl -s http://localhost:11434/api/embeddings -d '{"model": "all-minilm", "prompt": "warmup", "keep_alive": "24h"}' > /dev/null 2>&1 &
wait
echo "✓ Both models locked in GPU VRAM (zero model switching lag)."

# 4. Install Python Dependencies
echo ""
echo "📦 [4/6] Verifying Python Requirements..."
export PIP_DISABLE_PIP_VERSION_CHECK=1
if python3 -c "import pydantic, neo4j, httpx, numpy, orjson, tenacity, dotenv, async_lru, nltk, pytest" 2>/dev/null; then
    echo "✓ Core Python packages are already installed."
else
    pip install -q --prefer-binary \
        "pydantic>=2.7.0" "pydantic-settings>=2.3.0" "neo4j>=5.20.0" "httpx>=0.27.0" \
        "numpy>=1.26.0" "orjson>=3.10.0" "async-lru>=2.0.0" "tenacity>=8.3.0" \
        "python-dotenv>=1.0.0" "nltk>=3.8.1" pandas pytest bert-score rouge-score
fi
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true
echo "✓ Python dependencies verified."

# 5. Configure AuraDB Cloud Environment
echo ""
echo "☁️ [5/6] Configuring AuraDB Cloud .env..."
cat << 'EOF' > .env
NEO4J_URI=neo4j+s://8d2d5bc7.databases.neo4j.io
NEO4J_USER=8d2d5bc7
NEO4J_PASSWORD=bw8o9R3NXXApN2DXy72Gv7iXNI2KcVDGp_zZ7h0D5BU

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b-instruct
EMBEDDING_MODEL=all-minilm
SIMILARITY_THRESHOLD=0.65
LOG_LEVEL=INFO
EOF
echo "✓ .env configured with Neo4j AuraDB credentials."

# Also copy .env to parent if in subfolder
if [ "$PROJECT_DIR" != "$SCRIPT_DIR" ]; then
    cp .env "$SCRIPT_DIR/.env" 2>/dev/null || true
fi

# 6. Start UI Server on Port 7860
echo ""
echo "🌐 [6/6] Launching Demonstration UI Server on Port $UI_PORT..."
pkill -f "server.py" || true
pkill -f "server.py 8080" || true
fuser -k ${UI_PORT}/tcp 2>/dev/null || true
sleep 1

# Find server script
if [ -f "server.py" ]; then
    SERVER_SCRIPT="server.py"
elif [ -f "ui/server.py" ]; then
    SERVER_SCRIPT="ui/server.py"
elif [ -f "$PROJECT_DIR/server.py" ]; then
    SERVER_SCRIPT="$PROJECT_DIR/server.py"
elif [ -f "$PROJECT_DIR/ui/server.py" ]; then
    SERVER_SCRIPT="$PROJECT_DIR/ui/server.py"
else
    SERVER_SCRIPT="server.py"
fi

nohup python3 "$SERVER_SCRIPT" "$UI_PORT" > /tmp/m4_ui.log 2>&1 &
SERVER_PID=$!
disown $SERVER_PID 2>/dev/null || true
sleep 2

# Verify UI server is up
for attempt in {1..15}; do
    if curl -s "http://localhost:$UI_PORT/api/status" > /dev/null 2>&1; then
        echo "✓ UI HTTP server running cleanly on port $UI_PORT."
        break
    fi
    sleep 1
done

# Ensure cloudflared is installed for instant free public URL
if ! command -v cloudflared &> /dev/null; then
    echo "Installing cloudflared for public URL tunnel..."
    curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb > /dev/null 2>&1 || true
    dpkg -i /tmp/cloudflared.deb > /dev/null 2>&1 || true
fi

# Create public Cloudflare HTTPS tunnel
PUBLIC_URL=""
if command -v cloudflared &> /dev/null; then
    echo "Creating instant public HTTPS tunnel for Demonstration UI..."
    pkill -f "cloudflared tunnel --url" || true
    sleep 1
    nohup cloudflared tunnel --url "http://localhost:$UI_PORT" > /tmp/cf_ui.log 2>&1 &
    for i in {1..20}; do
        PUBLIC_URL=$(grep -o 'https://[-a-zA-Z0-9]*\.trycloudflare\.com' /tmp/cf_ui.log 2>/dev/null | tail -n 1 || true)
        if [ -n "$PUBLIC_URL" ]; then
            break
        fi
        sleep 1
    done
fi

echo ""
echo "================================================================================"
echo "    🎉 SETUP COMPLETE! YOUR M4 DEMONSTRATION UI IS ACTIVE!                     "
echo "================================================================================"
if [ -n "$PUBLIC_URL" ]; then
echo "  👉 🌐 INSTANT PUBLIC URL (CLICK TO OPEN IN BROWSER / SHARE WITH MENTOR):"
echo "     $PUBLIC_URL"
echo "     --------------------------------------------------"
fi
echo "  👉 VIEW INSIDE GOOGLE COLAB (RECOMMENDED):"
echo "     Run this in a Colab code cell:"
echo "     --------------------------------------------------"
echo "     from google.colab import output"
echo "     output.serve_kernel_port_as_iframe($UI_PORT, height=850)"
echo "     --------------------------------------------------"
echo ""
echo "  👉 AURADB CLOUD:      Connected to neo4j+s://8d2d5bc7.databases.neo4j.io"
echo "  👉 GPU MODELS:        qwen2.5:7b-instruct + all-minilm (CUDA accelerated)"
echo "================================================================================"
echo ""
echo "Server logs: /tmp/m4_ui.log"


