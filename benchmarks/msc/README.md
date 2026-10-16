# MSC Benchmark & Ablation Suite (`benchmarks/msc`)

This directory implements the evaluation protocol for the **Multi-Session Chat (MSC)** benchmark and the component ablation studies from *"Meta-Memory for Large Language Models"* (Liang et al., IEEE TASLP, vol. 34, 2026).

## 1. Overview & Dataset Sampling

MSC (Xu et al., 2022) is a large-scale conversational dataset containing multi-session dialogues where speakers converse over multiple distinct encounters.

Because full MSC contains thousands of dialogues with up to 14 sessions each, the paper evaluates on a fixed subset. For detailed component ablation experiments (Table 4), Liang et al. report numbers on a representative **50-episode sample**.
- This repo freezes a deterministic 50-episode sample with random seed 42 in `benchmarks/msc/sample_ids.json`.

---

## 2. THEANINE Protocol Formulation

Following the **THEANINE protocol** cited by Liang et al.:
1. **Past Sessions Ingestion (Sessions $1 \dots k-1$):**
   - Ingested chronologically into `M4Memory` via `mem.ingest_session(...)`.
   - Each session is structured into category chains with consistency calibration.
2. **Current Session Prompt (Session $k$, Turn 0):**
   - The user's opening remark in the final session is provided as the query prompt to `mem.respond(...)`.
3. **Response Evaluation (Session $k$, Turn 1):**
   - The model's response is compared directly to the target ground-truth assistant reply.
   - Evaluated metrics: token F1, BLEU-1, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, Accuracy.

---

## 3. Ablation Studies (Table 4 Reproduction)

You can ablate individual M4 components via the `--ablate` CLI flag:
- `--ablate chain`: **w/o Chain** (Disables chronological category chain ordering).
- `--ablate calibrate`: **w/o Calibrate** (Disables consistency conflict detection and updating).
- `--ablate compress`: **w/o Compress** (Disables dormant node LLMLingua compression).
- `--ablate self_reflection`: **w/o Self-reflection** (Disables iterative LLM node-level quality checks during retrieval).
- `--ablate demonstration`: **w/o Demonstration** (Removes prompt in-context formatting demonstrations).

---

## 4. How to Run

### Smoke Run (2 episodes)
```bash
.venv/bin/m4bench run msc --stage smoke
```

### Pilot Run (5 episodes)
```bash
.venv/bin/m4bench run msc --stage pilot
```

### Full Benchmark (50-episode sample)
```bash
.venv/bin/m4bench run msc --stage full
```

### Run Ablation (e.g. w/o Calibrate)
```bash
.venv/bin/m4bench run msc --stage full --ablate calibrate
```

### View Results Report
```bash
.venv/bin/m4bench report results/msc/<run_id>
```
