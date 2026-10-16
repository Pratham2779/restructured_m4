# LoCoMo Benchmark Protocol (`benchmarks/locomo`)

This directory implements the evaluation protocol for the **LoCoMo** (Long-Context Memory) benchmark from *"Meta-Memory for Large Language Models"* (Liang et al., IEEE TASLP, vol. 34, 2026).

## 1. Dataset & Task Overview

LoCoMo evaluates an LLM long-term memory system across realistic multi-session conversations spanning several simulated weeks and months.
Each conversation consists of:
- Multiple chronological sessions with explicit session timestamps (e.g. `1:56 pm on 8 May, 2023`).
- Natural dialogue exchanges between two speakers.
- Question-Answer pairs testing episodic long-term recall and reasoning.

### Question Categories
1. **Single-Hop (282 questions):** Fact retrieval directly from a single previous session turn.
2. **Temporal (321 questions):** Questions requiring understanding time progression, dates, intervals, and sequence of events.
3. **Multi-Hop (96 questions):** Complex associative reasoning linking clues across different sessions.
4. **Open-Domain (841 questions):** General dialogue recall and background inquiries.
5. **Adversarial (446 questions):** Distractor and contradiction probes.

> **Paper Protocol Alignment:** Following Section IV of Liang et al. (2026), evaluation focuses on categories 1–4, yielding exactly **1,540 evaluated QA pairs**. Category 5 is excluded from primary scoring.

---

## 2. Evaluation Protocol

1. **Chronological Ingestion:**
   - For each conversation (10 total in the benchmark), sessions are ingested sequentially into `M4Memory` via `mem.ingest_session(namespace, session)`.
   - Each session undergoes category learning, chain construction, and consistency calibration.
2. **Self-Reflection Retrieval:**
   - For each QA pair, `mem.respond(namespace, question)` checks LLM confidence.
   - If confidence is low, bidirectional retrieval is activated along relevant category chains.
   - LLM self-evaluates candidate memory nodes before deciding whether to adopt or skip.
3. **Scoring:**
   - Responses are evaluated against reference answers using token F1, BLEU (1 and 4), ROUGE (1, 2, L), and exact/substring accuracy.
   - Metrics are computed overall and broken down per category.
4. **Resumption & Checkpointing:**
   - Every evaluated QA pair is atomically appended to `results/locomo/<run_id>/items.jsonl`.
   - Interrupted runs resume without re-evaluating completed questions.

---

## 3. How to Run

### Smoke Test (fast verification: 1 conversation, 2 questions)
```bash
.venv/bin/m4bench run locomo --stage smoke
```

### Pilot Evaluation (1 conversation, 20 questions)
```bash
.venv/bin/m4bench run locomo --stage pilot
```

### Full Benchmark (all 10 conversations, 1,540 QA pairs)
```bash
.venv/bin/m4bench run locomo --stage full
```

### View Results Report
```bash
.venv/bin/m4bench report results/locomo/<run_id>
```

### Gamma Parameter Sweep (Figure 9 reproduction)
```bash
.venv/bin/m4bench sweep-gamma --dataset locomo --stage pilot
```
