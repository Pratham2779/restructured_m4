# M4 — Meta-Memory for Large Language Models

M4 is a standalone, pip-installable long-term memory engine for Large Language Models based on the paper *"Meta-Memory for Large Language Models"* (Liang et al., IEEE TASLP, vol. 34, 2026).

## Key Features
- **Zero Framework Lock-in:** Standalone library with pluggable protocol interfaces (`LLMClient`, `Embedder`, `MemoryStore`, `Compressor`, `Observer`).
- **Category-Based Learning:** Disambiguates and consolidates information by topic categories.
- **Chain-Based Memory Construction:** Maintains chronological evolutionary chains and multi-category shared memory graphs.
- **Self-Monitoring Updating:** Continuous consistency calibration and iterative token compression.
- **Self-Reflection Retrieval:** Dynamic bidirectional traversal bounded by LLM confidence.
- **Strict Namespace Isolation:** Every operation is scoped by namespace (user / conversation episode).
- **Dual Store Engine:** High-speed in-memory store adapter for local testing/benchmarks and production Neo4j store.
