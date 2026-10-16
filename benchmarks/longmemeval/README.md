# LongMemEval Benchmark Protocol (Stub)

## Status: Planned / Upcoming

This directory will house the protocol implementation for the **LongMemEval** benchmark evaluated in *"Meta-Memory for Large Language Models"* (Liang et al., IEEE TASLP, vol. 34, 2026).

## Dataset Expectation
- **Input Data Format:** Synthetic and naturalistic long-horizon conversational sessions designed specifically to stress-test ultra-long context and multi-session memory architectures.
- **Expected Location:** `benchmarks/longmemeval/data/longmemeval.json` (gitignored).
- **Task:** Evaluating episodic recall and multi-hop fact synthesis across very long sequences of conversational turns.

## Why Not Yet Implemented
Priority in this phase was completing the primary LoCoMo and MSC benchmarks. When the LongMemEval dataset is placed in `benchmarks/longmemeval/data/`, it will integrate with `m4bench` with the same standardized components:
- `config.yaml`
- `loader.py`
- `protocol.py` (exposing `run_benchmark(...)`)
- `expected.yaml`
