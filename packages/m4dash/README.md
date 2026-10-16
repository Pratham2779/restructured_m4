# m4dash

Read-only Streamlit results dashboard for Meta-Memory (M4).

## Constraints
- **Zero engine imports:** `m4dash` strictly never imports `m4`, `m4bench`, or any benchmark runner.
- Pure consumer of result directories (`manifest.json`, `items.jsonl`, `aggregate.json`, `expected.yaml`).

## Features
- Run discovery across `results/**/manifest.json`.
- Metadata inspection (git commit, hardware, models, gamma).
- Metrics comparison against paper `expected.yaml` with color-coded delta indicators.
- Per-item drill-down and trace visualizer.
- Token, latency, and resource counters.
- One-click ZIP run exporter.

## Usage
```bash
streamlit run packages/m4dash/src/m4dash/app.py
```
