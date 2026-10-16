# AGENTS.md — Architecture Rules & Guidelines for Agents

## 1. Architectural Layering Rules (Strictly Enforced)

1. **`packages/m4` (Core Library):**
   - The engine must remain a completely standalone, pip-installable library.
   - **`m4` MUST NEVER import** anything from `m4bench`, `benchmarks`, `m4dash`, or any notebook/script.
   - Zero hard-coded paths, URLs, keys, or dataset-specific logic (no mention of LoCoMo, MSC, Streamlit, Cloudflare, Colab, etc.).

2. **`packages/m4bench` (Benchmark Harness):**
   - Depends only on `m4`.
   - Never imported by `m4`.
   - Houses reusable benchmark execution logic, atomic checkpointing/resumption, NLP metrics, and observers.

3. **`benchmarks/*` (Dataset Protocols):**
   - Imports `m4` and `m4bench` only.
   - Datasets must never import each other (e.g. `benchmarks/locomo` never imports `benchmarks/msc`).

4. **`packages/m4dash` (Dashboard):**
   - Reads files only (`results/**/manifest.json`, `items.jsonl`, `aggregate.json`, `expected.yaml`).
   - Must never import `m4`, `m4bench`, or any benchmark runner.

## 2. Public API Stability

All interactions with M4 must proceed strictly through the unified facade `M4Memory` defined in `m4.api`:
- `mem = M4Memory(config=..., llm=..., embedder=..., store=..., compressor=..., observer=...)`
- `mem.ingest_session(namespace, session)`
- `mem.retrieve(namespace, query, dialogue_context=...)`
- `mem.respond(namespace, query, dialogue_context=...)`
- `mem.maintain(namespace, now)`
- `mem.reset(namespace)`
- `mem.stats(namespace)`

## 3. Time Injection Rule

Time must always be injected (via `Session.timestamp` or `maintain(namespace, now)`).
**Never read from the system wall clock (`datetime.now()`)** inside the core pipeline or store algorithms.

## 4. How to Run Tests

Unit tests and contract tests run completely hermetically with deterministic in-memory fakes (zero network, zero GPU):

```bash
.venv/bin/pytest packages/m4/tests -v
```

To run Neo4j integration contract tests (optional, only when active instance is provided):
```bash
NEO4J_URI=bolt://... NEO4J_USER=... NEO4J_PASSWORD=... .venv/bin/pytest packages/m4/tests/test_ports_contract.py -v
```
