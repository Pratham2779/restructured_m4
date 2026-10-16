# M4 Benchmark Execution on Kaggle

This directory contains scripts tailored for running M4 evaluations on Kaggle Notebooks (Dual T4 GPU or P100 environment).

## Kaggle Environment Specifics
- **Disk Limit:** 20 GB root disk. Model storage is placed under `/tmp/ollama_models` to prevent filling the root filesystem.
- **Background Processes:** Ollama is run as a background service via `start_models.sh`.
- **Output Artifacts:** Checkpointed results should be directed to `/kaggle/working/results` or an attached Kaggle Dataset for permanence.

## Usage
1. Open a new Kaggle Notebook with GPU enabled (T4 x2 or P100).
2. Run in a code cell:
   ```bash
   !bash infra/kaggle/bootstrap.sh
   !m4bench run locomo --stage pilot --results-dir /kaggle/working/results
   ```
