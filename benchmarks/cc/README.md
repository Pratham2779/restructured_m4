# Conventional Chat (CC) Benchmark Protocol (Stub)

## Status: Planned / Upcoming

This directory will house the protocol implementation for the **Conventional Chat (CC)** benchmark evaluated in *"Meta-Memory for Large Language Models"* (Liang et al., IEEE TASLP, vol. 34, 2026).

## Dataset Expectation
- **Input Data Format:** Standard multi-turn open-domain chat dialogue JSON.
- **Expected Location:** `benchmarks/cc/data/cc_dataset.json` (gitignored).
- **Task:** Evaluating conversational coherence and factual consistency over conventional multi-turn dialogues without artificial temporal gaps.

## Why Not Yet Implemented
Priority was given to implementing and verifying the core episodic memory benchmarks:
1. **LoCoMo**: Multi-session long-context reasoning across 1,540 QA pairs.
2. **MSC**: Multi-session chat under THEANINE protocol and component ablations.

Once CC dataset files are acquired, this directory will follow the exact same interface:
- `config.yaml`
- `loader.py`
- `protocol.py` (exposing `run_benchmark(...)`)
- `expected.yaml`
