"""Exact match and substring accuracy metrics."""
from m4bench.metrics.f1 import normalize_answer


def compute_accuracy(pred: str, gold: str) -> float:
    """Exact normalized match accuracy (0.0 or 100.0)."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return 0.0
    return 100.0 if p == g or g in p else 0.0
