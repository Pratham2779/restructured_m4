# AUDIT REPORT: M4 Legacy Codebase vs. Paper Conformance

**Paper:** *"Meta-Memory for Large Language Models"* (Bin Liang, Cai Ke, Runcong Zhao, Qinglin Zhu, Lin Gui, Yue Yu, Hui Wang, Ruifeng Xu, Kam-Fai Wong — IEEE TASLP, vol. 34, 2026)  
**Branch:** `refactor/m4-v1`  
**Date:** October 16, 2026  
**Status:** Phase 0 Audit Complete — Awaiting Approval

---

## 1. Module Inventory

This inventory catalogs every legacy module moved to `legacy/`, detailing its functionality, callers, dead code, duplicated logic, hardcoded paths/keys, and global state.

### 1.1 `legacy/m4_optimized/src/config.py`
- **What it does:** Defines `AppConfig` using `pydantic-settings` to load settings from `.env` or defaults (Neo4j URI/credentials, Ollama URL, model names, similarity threshold, logging). Defines `setup_logger`.
- **Who calls it:** Called by all legacy modules (`llm_client.py`, `graph_store.py`, `layer1_learning.py`, `layer2_construction.py`, `layer3_updating.py`, `layer4_retrieval.py`, `pipeline.py`, `server.py`, `run_all_paper_evaluations.py`, `run_locomo_eval.py`).
- **Dead code:** Attributes `embedding_cache_size`, `llm_cache_size`, and `max_workers` (lines 14–16) are defined on `AppConfig` but never referenced anywhere in the codebase.
- **Duplicates:** Custom `setup_logger` wrapper duplicated conceptually across multiple evaluation scripts.
- **Hard-coded paths/URLs/keys:** Default local Neo4j credentials (`bolt://localhost:7687`, `neo4j`, `StrongPassword123!`), default Ollama URL `http://localhost:11434`.
- **Global state:** `logging.getLogger` instances configured at module import time.
- **Deviations:** Default `similarity_threshold` is set to `0.65` (line 12), deviating from the paper's specified default $\gamma = 0.50$.

### 1.2 `legacy/m4_optimized/src/models.py`
- **What it does:** Defines Pydantic data models: `RawMemoryItem`, `DisambiguatedMemoryClue`, `MemoryNodeModel`, and `MemoryChainModel`.
- **Who calls it:** `llm_client.py`, `layer1_learning.py`, `layer2_construction.py`, `layer3_updating.py`, `pipeline.py`, `test_unit.py`.
- **Dead code:** `MemoryChainModel` (lines 25–30) is never instantiated or returned by any pipeline or store operation.
- **Duplicates / Incompatibilities:** `MemoryNodeModel.category: str` (line 19) forces every memory node to belong to exactly one category, directly conflicting with the paper's design where nodes can belong to multiple categories (shared graph nodes).
- **Hard-coded values:** None.
- **Global state:** None.

### 1.3 `legacy/m4_optimized/src/llm_client.py`
- **What it does:** Concrete client communicating directly with Ollama's HTTP endpoints (`/api/embed`, `/api/chat`, with fallback to `/v1/chat/completions`). Includes `@alru_cache` for embeddings, a manual dictionary cache, retry via `tenacity`, and regex-based JSON array extraction.
- **Who calls it:** `layer1_learning.py`, `layer3_updating.py`, `layer4_retrieval.py`, `pipeline.py`.
- **Dead code:** Pre-compiled regex patterns `_CODE_BLOCK_PATTERN` and `_BRACKET_PATTERN` (lines 14–15) are compiled at module load time but re-compiled or re-matched via ad-hoc strings in `_parse_json_array` (lines 136, 140).
- **Duplicates:** JSON extraction regex and parsing logic duplicated between `llm_client.py`, `layer4_retrieval.py`, and `server.py`.
- **Hard-coded paths/URLs/keys:** Replaces `"localhost"` with `"127.0.0.1"` via hardcoded string replacement (lines 44, 63, 88). Hardcodes `temperature=0.0` in `complete_chat` (line 84) and `extract_structured_json` (line 120), violating the paper's specification of temperature 0.80. Hardcodes `max_tokens=350`.
- **Global state:** Unbounded dictionary `_embedding_cache` (line 24) on the class instance, plus `@alru_cache(maxsize=4096)`.

### 1.4 `legacy/m4_optimized/src/graph_store.py`
- **What it does:** Contains two graph storage implementations:
  1. `InMemoryGraphStore` (lines 13–203): in-memory graph using dictionaries and numpy for vector cosine similarities.
  2. `Neo4jGraphStore` (lines 204–520): Cypher queries executing UNWIND batches, vector index lookups, and relationship traversals. Automatically falls back to `InMemoryGraphStore` on connection error.
- **Who calls it:** `layer2_construction.py`, `layer3_updating.py`, `layer4_retrieval.py`, `pipeline.py`.
- **Dead code:** `upsert_chains_and_head_nodes_batch` (lines 47–79 and lines 281–355) is completely unused (superseded by `create_memory_chains_batch` and `append_nodes_batch`).
- **Duplicates:** An alternative version of `InMemoryGraphStore` is independently re-implemented in `run_m4.py` (lines 37–100) with a different internal dictionary structure.
- **Hard-coded paths/URLs/keys:** Neo4j vector index name `'category_chain_embeddings'` is hardcoded (line 264) without schema bootstrap verification.
- **Global state:** `self._fallback` switches store mode dynamically and unpredictably upon driver error.
- **Critical Architectural Flaws:**
  - **Zero Namespace Isolation:** Neither `InMemoryGraphStore` nor `Neo4jGraphStore` scopes queries by user, session, or episode namespace. All `MemoryNode` and `MemoryChain` entities are queried and mutated globally, causing cross-episode data leakage during benchmark runs.
  - **No Protocol / Interface Contract:** The in-memory and Neo4j classes do not implement a shared abstract base class or `typing.Protocol`.
  - **No multi-category storage:** Only supports single category per node in Cypher properties.

### 1.5 `legacy/m4_optimized/src/layer1_learning.py`
- **What it does:** Layer 1 memory learning. Builds extraction prompt, queries LLM, obtains embeddings for categories and clues, computes Gram matrix of cosine similarities, and merges categories with $\text{sim} \ge \gamma$ using a Union-Find data structure.
- **Who calls it:** `pipeline.py`.
- **Dead code:** `compute_cosine_similarity` (lines 32–38) is defined at module scope but never called inside `layer1_learning.py` (which uses `_gram_matrix` on lines 40–44 and 83).
- **Duplicates:** `compute_cosine_similarity` is duplicated in `layer1_learning.py`, `layer3_updating.py` (`_cosine`), and `run_m4.py`.
- **Hard-coded values:** Extraction prompt substitutes concrete examples ("Personal Preferences", "Life Events") instead of paper's generic template (Fig. 2). Uses `config.similarity_threshold` (0.65).
- **Global state:** None.

### 1.6 `legacy/m4_optimized/src/layer2_construction.py`
- **What it does:** Layer 2 memory construction. Matches incoming category embeddings against existing chain embeddings (argmax). If $\max \text{sim} \ge \gamma$, appends node to existing chain; otherwise creates a new chain. Attempts to create shared nodes.
- **Who calls it:** `pipeline.py`.
- **Dead code:** None.
- **Duplicates:** None.
- **Hard-coded values:** Uses `config.similarity_threshold` (0.65).
- **Global state:** None.
- **Critical Flaws:**
  - In lines 55–60, lines arbitrarily designate the first node (`nodes[0]`) as an "anchor" and link all other nodes in the session to it with `SHARED_WITH` edges. This has no basis in the paper.
  - Uses `datetime.now(timezone.utc)` (line 40) rather than injected dataset timestamps.

### 1.7 `legacy/m4_optimized/src/layer3_updating.py`
- **What it does:** Implements memory maintenance actions: `calibrate_chain` and `compress_dormant_nodes`.
- **Who calls it:** `pipeline.py`.
- **Dead code:** None.
- **Duplicates:** `_cosine` (lines 36–39) duplicates `compute_cosine_similarity`.
- **Hard-coded values:** Hardcodes `days_idle: int = 14` (line 104), hardcodes cosine similarity threshold `sim >= 0.55` (line 64), hardcodes `top_k = candidates[:2]` (line 67), hardcodes token overlap fraction `overlap >= 0.6` (line 125).
- **Global state:** `_COMPRESSOR = None` (lines 12–31) lazily initializes and stores a global `PromptCompressor` instance on CPU.
- **Critical Flaws:**
  - Bypasses LLM consistency check for any node with cosine similarity $< 0.55$.
  - Only processes `conflicts[0]`, ignoring any further conflicts.
  - Replaces paper's LLM recoverability verification (Fig. 5) with a crude token set overlap check ($\ge 0.6$).
  - Evaluates idle time using wall clock (`datetime.now(timezone.utc) - timedelta(...)`).

### 1.8 `legacy/m4_optimized/src/layer4_retrieval.py`
- **What it does:** Implements Layer 4 retrieval and response generation. Queries LLM for relevant categories, embeds query and categories, searches chains bidirectionally around the argmax node, evaluates confidence, and generates the final response.
- **Who calls it:** `pipeline.py`.
- **Dead code:** `self.gamma = getattr(config, "similarity_threshold", 0.50)` is assigned (line 27) but never used in the class.
- **Duplicates:** Custom prompt building.
- **Hard-coded values:** Hardcodes `max_depth = 2` (line 101), hardcodes cosine threshold `0.25` (lines 96, 112, 118), hardcodes confidence check threshold `len(memory_context) >= 4` (line 126).
- **Global state:** `@alru_cache(maxsize=512)` on `_is_confident` (line 150).
- **Critical Flaws:**
  - Dumps all existing category names into the prompt rather than having the LLM independently infer categories from the query (violating Fig. 6).
  - Skips LLM self-evaluation ($LLM_{re}$) during traversal, replacing it with `b_sim >= 0.25 or step == 1`.
  - Replaces paper's Fig. 7 and Fig. 8 prompts with custom instructional prompts.
  - Never checks if LLM is confident to answer without retrieval.
  - Does not traverse shared nodes when chains are exhausted.

### 1.9 `legacy/m4_optimized/src/pipeline.py`
- **What it does:** Orchestrates the 4 layers into `MetaMemoryPipeline` (`ingest_session`, `query`, `compress`).
- **Who calls it:** `server.py`, `run_all_paper_evaluations.py`, `run_locomo_eval.py`, `test_live.py`, `test_custom.py`.
- **Dead code:** `compress` is never invoked during session ingestion or evaluation workflows.
- **Duplicates:** Tightly couples layer classes directly rather than accepting port interfaces.
- **Global state:** None.
- **Critical Flaws:** No namespace parameter on any method; uses wall clock time (`datetime.now()`); returns unstructured dictionaries.

### 1.10 UI Server & Web Interface
- **Files:** `legacy/server.py`, `legacy/m4_optimized/server.py`, `legacy/m4_optimized/ui/server.py` (3 identical copies); `legacy/index.html`, `legacy/m4_optimized/index.html`, `legacy/m4_optimized/ui/index.html` (3 identical copies).
- **What it does:** A monolithic 543-line script using Python's `ThreadingHTTPServer` to serve a static 68 KB HTML dashboard and expose endpoints (`/api/run-tests`, `/api/run-evaluations`, `/api/ingest`, `/api/query`, etc.). Runs `pytest` and benchmark scripts via raw `subprocess.Popen`.
- **Dead code:** Extensive heuristic path-probing blocks (lines 23–37, 91–100, 149–160, 268–278).
- **Global state:** Background loop thread `_LOOP_THREAD`, threading lock `_EVAL_LOCK`, global evaluation state dictionary `_EVAL_STATE`.

### 1.11 Benchmark Evaluation Scripts
- **Files:** `legacy/run_all_paper_evaluations.py`, `legacy/m4_optimized/run_all_paper_evaluations.py`, `legacy/m4_optimized/run_locomo_eval.py`.
- **What they do:** Scripts to run LoCoMo and MSC evaluations, compute NLP metrics (BLEU, ROUGE-L, F1, BERTScore), and output comparison tables.
- **Dead code / Duplicates:** `run_locomo_eval.py` is an unmaintained duplicate of the LoCoMo portion of `run_all_paper_evaluations.py`.
- **Hard-coded paths/URLs/keys:** Hardcodes Cloud Neo4j password directly in source code (`CLOUD_NEO4J_PASSWORD = "bw8o9R3NXXApN2DXy72Gv7iXNI2KcVDGp_zZ7h0D5BU"` at line 53).
- **Critical Protocol Violations:**
  - `run_quick_live_demo` (lines 640–738) returns **fake hardcoded paper numbers** (e.g., F1 25.61%, BLEU-4 1.26%) instead of measuring actual model output.
  - LoCoMo: Evaluates only 1 conversation (out of 10) and only 2 to 5 sessions, filtering out QA items whose evidence extends into un-ingested sessions.
  - MSC: Evaluates only 3 episodes (out of 50 sampled) with non-standard prompts.
  - No per-item checkpointing or resume support.

### 1.12 Shell Scripts and Notebooks
- **Files:** `colab_setup_and_launch_all.sh`, `start_demo.sh`, `m4_turnkey_cloud_runner.sh`, `setup_colab_kaggle.sh`, `run_benchmark.sh`, `run_eval_suite.sh`, `run_locomo_tests.sh`, `m4_colab_kaggle_benchmark.ipynb`.
- **Flaws:** Redundant wrapper scripts. `colab_setup_and_launch_all.sh` hardcodes credentials in a `cat << 'EOF' > .env` block and starts ephemeral ad-hoc Cloudflare tunnels rather than named tunnels.

---

## 2. Paper-Conformance Checklist

Every item evaluated against the paper's specifications (IEEE TASLP 2026):

| Item | Requirement from Paper | Legacy Status | Legacy Ref | Findings & Evidence |
|---|---|---|---|---|
| **2.1 Category-based learning: extraction** | Extract candidate categories $T^o = \{t_i\}_{i=1}^{n_o}$ and summaries from session conversation. | **MATCHES** (in principle) / **DIFFERS** (in prompt) | `layer1_learning.py`: 51–66 | Uses JSON extraction. However, prompt deviates from Fig. 2 by embedding specific few-shot examples ("Personal Preferences", "Life Events") rather than paper's generic format. |
| **2.2 Category-based learning: similarity merge** | Pairwise cosine similarity $\eta = \text{sim}(t_i, t_j)$; merge if $\eta \ge \gamma$ ($\gamma = 0.50$). | **DIFFERS** | `layer1_learning.py`: 82–90; `config.py`: 12 | Uses Gram matrix and Union-Find. However, default threshold is $\gamma = 0.65$ in `config.py` instead of the paper's $\gamma = 0.50$. |
| **2.3 Category-based learning: clue summarization** | One summary "memory clue" per merged category ($c_i$ retaining key information of $t_i$). | **DIFFERS** | `layer1_learning.py`: 97–108 | Retains every member's summary separately under the merged category (e.g. 3 members $\rightarrow$ 3 clues) instead of consolidating into one unified memory clue per category. |
| **2.4 Category-based learning: un-known info filter** | Only keep information absent from the LLM's internal knowledge. | **DIFFERS** | `layer1_learning.py`: 56 | Mentioned in prompt text ("avoid summarizing common knowledge you already possess"), but no verification check is performed. |
| **2.5 Chain construction: argmax chain selection** | For each new $(t_k, c_k)$, find closest existing category $\eta^k = \max \text{sim}(t_k, t_i)$, $I^k = \text{argmax}$. | **MATCHES** | `layer2_construction.py`: 21–28; `graph_store.py`: 29–46, 254–280 | Correctly performs argmax similarity lookup over existing chains. |
| **2.6 Chain construction: append or new chain** | If $\eta^k \ge \gamma$, append clue as node to end of chain; otherwise create new chain for $t_k$. | **DIFFERS** | `layer2_construction.py`: 26–38; `config.py`: 12 | Logic matches Eq. 3, but threshold $\gamma = 0.65$ is used instead of $\gamma = 0.50$. |
| **2.7 Chain construction: multi-category shared nodes** | Nodes belonging to multiple categories serve as shared nodes forming the memory graph. | **DIFFERS** (fundamentally broken) | `models.py`: 19; `layer2_construction.py`: 55–60; `graph_store.py`: 139–141, 425–437 | `MemoryNodeModel` only stores a single category string. Code creates arbitrary `SHARED_WITH` edges between the first node of a session and other nodes in the same session, completely diverging from the paper's definition of shared nodes. |
| **2.8 Calibrate: consistency check** | For each new node, LLM consistency check across its chain returns $0..n$ conflicting nodes. | **DIFFERS** | `layer3_updating.py`: 57–85 | Implements an ad-hoc cosine pre-filter $\text{sim} \ge 0.55$ and slices `candidates[:2]`, skipping LLM checks for nodes with lower cosine similarity or outside the top 2. Prompt is truncated from Fig. 3. |
| **2.9 Calibrate: merge and chain update** | For each conflict, LLM merges conflicting + new node into unambiguous node, deletes conflicting node, appends merged node to end of chain. | **DIFFERS** | `layer3_updating.py`: 90–102 | Only handles the first conflict (`conflicts[0]`). Updates `new_node` in-place rather than deleting conflicting node and appending the merged node to the chain end ($C_k \leftarrow f_c(C_k \| \hat{c}_k)$). |
| **2.10 Compress: trigger interval** | Compress nodes unused for more than a "few weeks", provided by dataset or falling back to 1 week. | **DIFFERS** | `layer3_updating.py`: 104–106 | Uses hardcoded 14 days evaluated against system wall clock (`datetime.now(timezone.utc)`). No dataset-injected interval or simulated step progression. |
| **2.11 Compress: LLMLingua compression** | Nodes compressed iteratively using LLMLingua. | **DIFFERS** | `layer3_updating.py`: 110–118 | Single-pass compression with static rate 0.5; no iterative loop. |
| **2.12 Compress: LLM recoverability check** | After each compression, LLM checks whether key meaning is recoverable (Fig. 5); stops when not recoverable, keeping previous version. | **MISSING** | `layer3_updating.py`: 120–127 | Completely omits LLM check (Fig. 5). Replaces it with a crude word overlap heuristic: `len(orig & comp) / len(orig) >= 0.6`. |
| **2.13 Retrieval: category extraction & mapping** | LLM extracts query categories (Fig. 6); mapped to existing categories by similarity. | **DIFFERS** | `layer4_retrieval.py`: 44–80 | Injects all existing categories into the prompt for the LLM to choose, followed by top-3 semantic ranking fallback, rather than independent category extraction and similarity mapping. |
| **2.14 Retrieval: root node selection** | Root node $c_k^q = \text{argmax}_{j} \text{sim}(c_j^{t_i^q}, q)$ in each category chain. | **MATCHES** | `layer4_retrieval.py`: 88–95 | Matches line 5 of Algorithm 1. |
| **2.15 Retrieval: bidirectional traversal with $LLM_{re}$** | Gradual bidirectional search; node adopted if LLM self-evaluation ($LLM_{re}$, Fig. 7) determines it improves response. | **DIFFERS** | `layer4_retrieval.py`: 100–125 | Skips LLM self-evaluation call. Replaces it with static cosine heuristic `b_sim >= 0.25 or step == 1`. Hardcodes `max_depth = 2`. |
| **2.16 Retrieval: response regeneration** | Response regenerated whenever retrieved memory $M$ changes ($r = \text{LLM}(q, M)$). | **DIFFERS** | `layer4_retrieval.py`: 130–148 | Generates response only once at the end of the entire procedure rather than dynamically at each adoption step. |
| **2.17 Retrieval: termination & bounds** | Stops when LLM is confident ($LLM_{confident}$); bounded by finite chain length. | **DIFFERS** | `layer4_retrieval.py`: 104, 126 | Bounded by artificial `max_depth = 2`. Only checks confidence if $\ge 4$ memories gathered. Prompt deviates from Fig. 7. |
| **2.18 Retrieval: shared node jump** | If relevant chains are exhausted without confidence, jump to shared nodes. | **MISSING** | `layer4_retrieval.py` | Traversal ends when chain ends; shared node expansion is not implemented. |
| **2.19 Retrieval: chronological order** | Memory remains in original chronological order during integration. | **MATCHES** | `layer4_retrieval.py`: 97, 113, 119 | Inserts backward nodes at 0 and appends forward nodes. |
| **2.20 Retrieval triggered on lack of confidence** | Retrieval triggered only when LLM lacks confidence to answer on its own. | **MISSING** | `layer4_retrieval.py`: 29–38 | Always runs retrieval unconditionally without an initial confidence check. |
| **2.21 Defaults** | Temperature 0.80, $\gamma = 0.50$. | **DIFFERS** | `config.py`: 12; `llm_client.py`: 84 | Uses temperature 0.0 and $\gamma = 0.65$. |

---

## 3. Benchmark Testing Divergence

### 3.1 LoCoMo Benchmark Protocol Divergence
- **Corpus Size:**
  - *Paper:* Full LoCoMo dataset of **10 long conversations** (~600 dialogue turns, ~26,000 tokens each), evaluated over **~200 questions per conversation** (~1,986 questions total).
  - *Legacy:* Evaluates only **1 conversation** (`max_conversations = 1`), truncating dialogue to **2 to 5 sessions** (`max_s = min(2, len(session_keys))`), and sampling only **3 questions per category** (12 questions total).
- **Question Categories:**
  - *Paper Table III:* Reports 4 categories: Single-Hop, Multi-Hop, Temporal, Open-Domain.
  - *Dataset Structure:* The LoCoMo dataset contains 5 categories (1: Single-Hop, 2: Temporal, 3: Multi-Hop, 4: Open-Domain, 5: Adversarial).
  - *Legacy:* Arbitrarily filters questions where evidence extends beyond truncated sessions (`s <= max_ingested`), creating selection bias.
- **Metrics & Logging:**
  - *Paper:* Measures Token F1, BLEU-1, and average context tokens per question (~2,400–3,400).
  - *Legacy:* In `run_quick_live_demo`, the code outputs **hardcoded static target numbers** from Table III (e.g., F1 25.61%, 26.02%) rather than computing metrics from actual outputs.
- **Checkpointing & Resumption:**
  - *Paper requirement for Colab:* Per-item checkpointing and instant resume.
  - *Legacy:* Completely absent; failure or timeout requires restarting from scratch.

### 3.2 Multi-Session Chat (MSC) Protocol Divergence
- **Corpus Size:**
  - *Paper:* Randomly samples **50 episodes** from the MSC test set (**250 sessions** total), following THEANINE's setup.
  - *Legacy:* Evaluates only **3 episodes** (`episodes_count = 3`).
- **Session Sequencing & Context:**
  - *Paper (THEANINE setup):* Ingests sessions $0..k-1$ in sequence; at session $k \ge 2$, generates response conditioned on dialogue context and memory, evaluated against the human reference response.
  - *Legacy:* Ingests only the immediate previous session if `episodes_count <= 2` (`past_sessions = past_sessions[-1:]`). Synthesizes an artificial query string (`"In dialogue with Speaker 2, what should be the response to..."`) rather than providing standard conversational dialogue context.
- **Metrics:**
  - *Paper Table I:* Evaluates BLEU-4, ROUGE-L, MAUVE, BERTScore.
  - *Legacy:* MAUVE is completely omitted. In `run_quick_live_demo`, hardcoded paper results are emitted (`avg_bleu4: 1.26`, `avg_rouge_l: 14.15`, `avg_bertscore: 48.29`).
- **Ablation Support:**
  - *Paper Table II:* Ablations for `w/o Chain`, `w/o Calibrate`, `w/o Compress`, `w/o Self-reflection`, `w/o Demonstration`.
  - *Legacy:* No programmatic ablation flags supported in evaluation harness.

---

## 4. Proposed Migration Map (`old file → new file`)

| Legacy Path (`legacy/...`) | Target Path | Responsibility & Rationale |
|---|---|---|
| `m4_optimized/src/config.py` | `packages/m4/src/m4/config.py` | Pydantic `M4Config` with strict defaults ($\gamma = 0.5$, temp = 0.8, compress intervals, limits). Zero external framework dependencies. |
| `m4_optimized/src/models.py` | `packages/m4/src/m4/types.py` | Canonical dataclasses/models: `Session`, `Turn`, `MemoryNode` (supporting multiple category memberships), `Category`, `RetrievalResult`, `IngestReport`, `StoreStats`. |
| `m4_optimized/src/llm_client.py` | `packages/m4/src/m4/ports.py`<br>`packages/m4/src/m4/adapters/llm_ollama.py`<br>`packages/m4/src/m4/adapters/llm_openai_compat.py`<br>`packages/m4/src/m4/adapters/embed_ollama.py`<br>`packages/m4/src/m4/adapters/embed_sentence_transformers.py` | Extract `LLMClient` and `Embedder` Protocols in `ports.py`. Implement concrete adapters for Ollama and any OpenAI-compatible server (vLLM, llama.cpp). Add robust JSON parser with retries and event logging. |
| `m4_optimized/src/graph_store.py` | `packages/m4/src/m4/ports.py`<br>`packages/m4/src/m4/adapters/store_memory.py`<br>`packages/m4/src/m4/adapters/store_neo4j.py` | Define `MemoryStore` Protocol in `ports.py` with strict `namespace` scoping on every method. Build identical-behavior `store_memory.py` and `store_neo4j.py` passing identical contract tests. |
| `m4_optimized/src/layer1_learning.py` | `packages/m4/src/m4/pipeline/learning.py` | Clean pipeline stage implementing paper Section III.B: candidate category extraction, cosine similarity merge ($\gamma = 0.5$), single summary clue per category, internal knowledge check. |
| `m4_optimized/src/layer2_construction.py` | `packages/m4/src/m4/pipeline/construction.py` | Clean pipeline stage implementing paper Section III.C: argmax category search, chain linking, multi-category shared node creation, injected timestamps. |
| `m4_optimized/src/layer3_updating.py` | `packages/m4/src/m4/pipeline/update.py`<br>`packages/m4/src/m4/adapters/compressor_llmlingua.py` | Clean pipeline stage implementing paper Section III.D: full LLM consistency check across chain (no cosine pre-filter), multi-conflict resolution with merged node append; iterative LLMLingua compression with LLM recoverability check (Fig. 5). |
| `m4_optimized/src/layer4_retrieval.py` | `packages/m4/src/m4/pipeline/retrieval.py` | Strict implementation of Algorithm 1: confidence-gated query trigger, category extraction and similarity mapping, bidirectional traversal with $LLM_{re}$ self-evaluation, continuous response regeneration, shared-node fallback. |
| (New prompt repository) | `packages/m4/src/m4/prompts/*.txt` | Versioned text files containing verbatim prompts from Figures 2–8 of the paper. Zero embedded prompts in Python code. |
| `m4_optimized/src/pipeline.py` | `packages/m4/src/m4/api.py` | Public `M4Memory` facade exposing the minimal, stable API: `ingest_session`, `retrieve`, `respond`, `maintain`, `reset`, `stats`. |
| `server.py` / `m4_optimized/server.py` | `packages/m4/src/m4/server/app.py` | Optional thin FastAPI wrapper providing clean REST endpoints (`/v1/{ns}/sessions`, `/v1/{ns}/retrieve`, `/v1/{ns}/respond`, `/v1/{ns}/maintain`, `/v1/{ns}/stats`). |
| `run_all_paper_evaluations.py` / `run_locomo_eval.py` | `packages/m4bench/src/m4bench/runner.py`<br>`packages/m4bench/src/m4bench/schema.py`<br>`packages/m4bench/src/m4bench/metrics/*`<br>`packages/m4bench/src/m4bench/observers.py`<br>`packages/m4bench/src/m4bench/cli.py` | Generic benchmark engine in `m4bench`: per-item atomic checkpointing (`items.jsonl`, `manifest.json`), metrics (BLEU-1/4, ROUGE-L, MAUVE, BERTScore, F1, local LLM judge), token/latency observers, CLI (`m4bench run`, `m4bench report`). |
| `m4_optimized/datasets/locomo_benchmark.json` | `benchmarks/locomo/` | LoCoMo benchmark runner: loader, protocol (full 10 conversations, chronological sessions, 4 categories, adversarial category excluded), metrics, `expected.yaml` with paper numbers. |
| `m4_optimized/datasets/msc_dataset.json` | `benchmarks/msc/` | MSC benchmark runner: loader, 50-episode sample with fixed seed (`sample_ids.json`), THEANINE protocol, metrics, `expected.yaml`. |
| `m4_optimized/tests/test_unit.py` | `packages/m4/tests/` | Pure unit tests with deterministic fake LLM/Embedder + Store Contract Test suite run against both in-memory and Neo4j adapters. |
| `index.html` / UI code | `packages/m4dash/` | Clean Streamlit dashboard reading only result files (`manifest.json`, `aggregate.json`, `expected.yaml`). Zero engine imports; runs locally or on Colab. |
| `colab_setup_and_launch_all.sh` / scripts | `infra/colab/`<br>`infra/kaggle/` | Modular, idempotent infrastructure scripts (`bootstrap.sh`, `start_models.sh`, `start_tunnel.sh`) and minimalist 3-cell notebooks. |

---

## 5. Next Steps

Upon review and approval of this Phase 0 Audit:
1. Initialize the new workspace packaging (`pyproject.toml`, `packages/m4`, `packages/m4bench`, `packages/m4dash`, `benchmarks/`, `infra/`, `docs/`).
2. Implement Phase 1: Core `m4` engine, ports, adapters, versioned prompts, and contract tests.
3. Verify layering rules and ensure all unit tests pass with fakes (no network, no GPU required).
