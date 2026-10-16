# Running M4 Benchmarks on Google Colab (Free Tier Guide)

This guide explains how to run long-running M4 evaluations on Google Colab's free T4 GPU tier hermetically, reliably, and without losing progress across session disconnections.

---

## 1. Prerequisites & Constraints

- **Hardware:** Colab Free Tier (1x NVIDIA T4 GPU, ~12 GB VRAM, ~12.7 GB System RAM).
- **Execution Limit:** Google Colab terminates sessions after idle periods or a hard ~12-hour limit.
- **Persistence Strategy:**
  - Every benchmark run must mount Google Drive.
  - Results are written directly to Google Drive (`/content/drive/MyDrive/m4_results`).
  - Checkpointing is performed atomically per evaluated QA pair/episode (`items.jsonl`).

---

## 2. Managing Secrets (Zero Hardcoded Keys)

Never commit or hardcode credentials into notebook cells.
Use Google Colab's built-in **Secrets** manager (the key icon in the left sidebar):

1. **Accessing Secrets in Python:**
   ```python
   from google.colab import userdata
   import os

   # Optional Neo4j cloud store credentials
   os.environ["NEO4J_URI"] = userdata.get("NEO4J_URI") or ""
   os.environ["NEO4J_USER"] = userdata.get("NEO4J_USER") or "neo4j"
   os.environ["NEO4J_PASSWORD"] = userdata.get("NEO4J_PASSWORD") or ""

   # Optional Cloudflare tunnel token
   os.environ["CLOUDFLARE_TUNNEL_TOKEN"] = userdata.get("CLOUDFLARE_TUNNEL_TOKEN") or ""
   ```

---

## 3. Step-by-Step Execution Workflow

### Step 1: Open the Notebook
Open either [`infra/colab/run_locomo.ipynb`](file:///home/prathamesh/Desktop/restructured_m4/infra/colab/run_locomo.ipynb) or [`infra/colab/run_msc.ipynb`](file:///home/prathamesh/Desktop/restructured_m4/infra/colab/run_msc.ipynb) in Colab. Ensure Runtime Type is set to **T4 GPU**.

### Step 2: Mount Google Drive
Mounting Drive ensures results survive session disconnects:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Run Bootstrap
Run the headless setup script to install Ollama, download models, and install `m4` and `m4bench`:
```bash
!bash infra/colab/bootstrap.sh
```

### Step 4: Launch Benchmark
Run evaluation with the results directory pointing to Drive:
```bash
!m4bench run locomo --stage pilot --results-dir /content/drive/MyDrive/m4_results
```
Or for full runs:
```bash
!m4bench run locomo --stage full --results-dir /content/drive/MyDrive/m4_results
```

---

## 4. Handling Session Disconnects & Resumption

When Colab disconnects or reaches its 12-hour timeout:
1. Re-open the notebook and re-connect to a new T4 instance.
2. Run Step 2 (mount Drive) and Step 3 (bootstrap Ollama).
3. Re-run the exact same `m4bench run ...` command with the same `--results-dir`.
4. **Instant Resumption:** `m4bench` will inspect `items.jsonl`, detect all completed item IDs, skip them, and resume directly from the first incomplete item!

---

## 5. Inspecting Run Reports & Metrics

Run `m4bench report` on any completed or in-progress run directory:
```bash
!m4bench report /content/drive/MyDrive/m4_results/locomo/<run_id>
```
The report prints a rich terminal table with achieved metrics, comparisons against `expected.yaml` baselines from the paper, token counts, and average latencies.
