# M4: Meta-Memory for Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation and productized benchmark harness for **M4 (Meta-Memory for Memory Management)** from the IEEE TASLP 2026 paper:
> **"Meta-Memory for Large Language Models"**  
> *Sheng Liang, et al., IEEE Transactions on Audio, Speech, and Language Processing (TASLP), vol. 34, 2026.*

---

## Architecture Overview

```
restructured_m4/
├─ AGENTS.md                  # Strict layering rules and testing contracts
├─ pyproject.toml             # Unified workspace specification
├─ docs/                      # Complete documentation & audit
│  ├─ AUDIT.md                # Phase 0 legacy audit & paper conformance checklist
│  ├─ ARCHITECTURE.md         # M4 pipeline design and port/adapter contracts
│  ├─ PROTOCOLS.md            # Benchmark protocols, splits, and paper alignments
│  └─ COLAB.md                # Headless Colab execution & session resumption
├─ packages/
│  ├─ m4/                     # THE ENGINE: Standalone long-term memory library
│  ├─ m4bench/                # HARNESS: Resumable execution, metrics, observers, CLI
│  └─ m4dash/                 # DASHBOARD: Read-only Streamlit results visualizer
├─ benchmarks/
│  ├─ locomo/                 # LoCoMo benchmark protocol (1,540 evaluated QA pairs)
│  ├─ msc/                    # MSC benchmark & THEANINE component ablations
│  ├─ cc/                     # Conventional Chat protocol (stub)
│  └─ longmemeval/            # LongMemEval protocol (stub)
└─ infra/
   ├─ colab/                  # Colab notebooks, bootstrap, and service scripts
   └─ kaggle/                 # Kaggle headless scripts
```

---

## 1. Quick Start

### Installation
```bash
# Clone the repository
git checkout refactor/m4-v1

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages in editable mode
pip install -e packages/m4 -e packages/m4bench -e packages/m4dash
```

### Running Hermetic Tests
Tests run deterministically in-memory with zero network and zero GPU dependencies:
```bash
.venv/bin/pytest -v
```

---

## 2. Using M4 in Any Application

M4 exposes a single, unified facade `M4Memory`:

```python
from m4.api import M4Memory
from m4.config import M4Config
from m4.types import Session, Turn
from m4.adapters.store_memory import InMemoryStore
from m4.adapters.llm_openai_compat import OpenAICompatLLM
from m4.adapters.embed_ollama import OllamaEmbedder

# 1. Initialize M4
config = M4Config(gamma=0.5, temperature=0.8)
mem = M4Memory(
    config=config,
    llm=OpenAICompatLLM(base_url="http://localhost:11434/v1", model="qwen2.5:7b-instruct"),
    embedder=OllamaEmbedder(base_url="http://localhost:11434/api/embeddings", model="bge-small-en-v1.5"),
    store=InMemoryStore(),
)

# 2. Ingest past conversation sessions
session = Session(
    id="sess_001",
    turns=[
        Turn(speaker="Alice", text="I'm training for a marathon in November."),
        Turn(speaker="Bob", text="That's awesome! How many miles are you running each week?"),
    ]
)
mem.ingest_session(namespace="user_123", session=session)

# 3. Respond with self-reflective retrieval
resp = mem.respond(namespace="user_123", query="What event is Alice preparing for?")
print(resp.text)
```

---

## 3. Running Benchmarks (`m4bench`)

### LoCoMo (Long-Context Memory Benchmark)
Evaluates across Single-Hop, Temporal, Multi-Hop, and Open-Domain questions (1,540 QA pairs total, excluding adversarial probes per paper protocol):

```bash
# Smoke test (2 items - instant verification)
.venv/bin/m4bench run locomo --stage smoke --mock

# Pilot run (20 QA pairs)
.venv/bin/m4bench run locomo --stage pilot

# Full evaluation (all 1,540 QA pairs)
.venv/bin/m4bench run locomo --stage full
```

### MSC & Component Ablations
Evaluates multi-session chat under the THEANINE protocol on the frozen 50-episode sample:

```bash
# Full 50-episode baseline
.venv/bin/m4bench run msc --stage full

# Ablation: w/o Calibrate (Table 4)
.venv/bin/m4bench run msc --stage full --ablate calibrate

# Ablation: w/o Self-reflection (Table 4)
.venv/bin/m4bench run msc --stage full --ablate self_reflection
```

### Inspecting Reports Against Paper Numbers
```bash
.venv/bin/m4bench report results/locomo/<run_id>
```

### Figure 9 Parameter Sweep (Sensitivity over $\gamma$)
```bash
.venv/bin/m4bench sweep-gamma --dataset locomo --stage pilot
```

---

## 4. Google Colab Headless Execution

Because long benchmark runs can exceed standard Colab timeouts, `m4bench` writes every evaluated item atomically to disk.

1. Open [`infra/colab/run_locomo.ipynb`](infra/colab/run_locomo.ipynb) or [`infra/colab/run_msc.ipynb`](infra/colab/run_msc.ipynb).
2. Mount Google Drive to persist outputs at `/content/drive/MyDrive/m4_results`.
3. If a Colab session disconnects, simply reconnect and re-run the cell. `m4bench` detects completed `item_id`s in `items.jsonl` and resumes instantly without redoing work!
4. See [`docs/COLAB.md`](docs/COLAB.md) for full instructions and secret management.

---

## 5. Results Dashboard (`m4dash`)

Launch the read-only Streamlit dashboard to explore runs, inspect memory retrieval traces, and compare results against paper baselines:

```bash
streamlit run packages/m4dash/src/m4dash/app.py
```

Features:
- Run selector across all benchmark executions.
- Red/green delta pills comparing obtained metrics against `expected.yaml`.
- Per-item drilldown: inspect prompt, prediction, gold answer, and latencies.
- Retrieval trace visualizer: step-by-step bidirectional chain traversal and self-reflection decisions.
- One-click ZIP run export.