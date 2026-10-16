# m4bench

Reusable benchmark execution harness for Meta-Memory (M4).

## Features
- Per-item atomic checkpointing (`items.jsonl`) with instant resumption.
- Automated `manifest.json` generation and post-run `aggregate.json` reporting.
- Standard long-term memory NLP metrics: token F1, BLEU-1/2/3/4, ROUGE-1/2/L, Accuracy, MAUVE, BERTScore, LLM-as-a-Judge.
- CLI for evaluation runs, reports against expected paper baselines, and parameter sweeps.
