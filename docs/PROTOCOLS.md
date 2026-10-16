# M4 Evaluation Protocols & Dataset Specifications

This document outlines the exact experimental evaluation protocols implemented in this repository, referencing *"Meta-Memory for Large Language Models"* (Liang et al., IEEE TASLP, vol. 34, 2026).

---

## 1. LoCoMo (Long-Context Memory Benchmark)

- **Location:** `benchmarks/locomo/`
- **Citation:** Liang et al., 2026 (Table 2)
- **Dataset Size:** 10 full conversations, comprising multiple sessions and 1,986 raw QA pairs.

### Protocol Formulation
1. **Chronological Ingestion:**
   Sessions $1 \dots N$ for each conversation are ingested into `M4Memory` sequentially using the injected session timestamps (e.g. `1:56 pm on 8 May, 2023`).
2. **Category-based Memory Pipeline:**
   - LLM extracts candidate categories and summaries.
   - Cosine similarity threshold $\gamma = 0.5$ groups categories.
   - Nodes are chained chronologically and calibrated for consistency against earlier nodes.
3. **Response Generation with Algorithm 1 Retrieval:**
   - For each evaluated question, the LLM determines whether retrieval is needed (confidence score).
   - If confidence is low, bidirectional traversal retrieves relevant nodes with LLM self-reflection.
   - LLM generates the final answer.

### Question Categories & Paper Alignment
| Category ID | Name | QA Count | Evaluated in Paper? | Notes |
|---|---|---|---|---|
| 1 | Single-Hop | 282 | Yes | Direct recall from single session |
| 2 | Temporal | 321 | Yes | Time differences, dates, durations |
| 3 | Multi-Hop | 96 | Yes | Reasoning across multiple sessions |
| 4 | Open-Domain | 841 | Yes | General recall & conversational facts |
| 5 | Adversarial | 446 | **No (Excluded)** | Contradiction & probe distractors |
| **Total** | | **1,540** | | Exactly 1,540 QA pairs evaluated |

---

## 2. MSC (Multi-Session Chat) & Ablations

- **Location:** `benchmarks/msc/`
- **Citation:** Xu et al. (2022); Liang et al. (2026, Table 4)
- **Protocol Formulation:** **THEANINE Protocol**
  - Given an episode of $k$ sessions ($k \ge 2$):
  - Sessions $1 \dots k-1$: Memory ingestion.
  - Session $k$, Turn 0: User prompt.
  - Session $k$, Turn 1: Target gold response for evaluation.
- **Frozen Sample:** Fixed 50-episode sample generated with random seed 42, saved in `benchmarks/msc/sample_ids.json`.

### Ablation Studies (Table 4 Reproduction)
The harness exposes `--ablate` flags corresponding to Table 4 in the paper:
1. `w/o Chain` (`--ablate chain`): Disables ordered chain structure.
2. `w/o Calibrate` (`--ablate calibrate`): Disables consistency update and conflict merging.
3. `w/o Compress` (`--ablate compress`): Disables LLMLingua dormant node compression.
4. `w/o Self-reflection` (`--ablate self_reflection`): Disables node-level reflection during retrieval.
5. `w/o Demonstration` (`--ablate demonstration`): Removes formatting demonstration examples from LLM prompts.

---

## 3. Evaluation Metrics

All evaluations compute standard NLP and QA metrics:
- **Token-level F1:** SQuAD-style harmonic mean of precision and recall after normalization (lowercase, punctuation, articles).
- **BLEU:** BLEU-1 and BLEU-4 exact n-gram matching.
- **ROUGE:** ROUGE-1, ROUGE-2, and ROUGE-L (longest common subsequence).
- **Accuracy:** Exact normalized string match and substring containment.

---

## 4. Checkpointing & Resumption

All benchmark runs use atomic per-item persistence:
- Results: `results/<dataset>/<run_id>/items.jsonl` (flushed with `os.fsync`).
- Trace: `results/<dataset>/<run_id>/traces/<item_id>.json`.
- Aggregates: `results/<dataset>/<run_id>/aggregate.json`.
- Manifest: `results/<dataset>/<run_id>/manifest.json`.

If a Google Colab session dies or a run is interrupted, re-running the same command resumes immediately by skipping already completed `item_id`s.
