"""NLP and benchmark metrics."""
from m4bench.metrics.f1 import compute_f1, normalize_answer
from m4bench.metrics.bleu import compute_bleu
from m4bench.metrics.rouge import compute_rouge_l
from m4bench.metrics.bertscore import compute_bertscore_batch
from m4bench.metrics.mauve import compute_mauve
from m4bench.metrics.accuracy import compute_accuracy
from m4bench.metrics.llm_judge import evaluate_with_local_judge, LOCAL_JUDGE_NOTICE

__all__ = [
    "compute_f1",
    "normalize_answer",
    "compute_bleu",
    "compute_rouge_l",
    "compute_bertscore_batch",
    "compute_mauve",
    "compute_accuracy",
    "evaluate_with_local_judge",
    "LOCAL_JUDGE_NOTICE",
]
