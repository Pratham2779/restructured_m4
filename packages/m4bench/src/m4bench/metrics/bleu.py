"""BLEU-1 and BLEU-4 metrics."""
from typing import Tuple
from m4bench.metrics.f1 import normalize_answer

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _SMOOTH = SmoothingFunction().method1
except ImportError:
    _SMOOTH = None


def compute_bleu(pred: str, gold: str) -> Tuple[float, float]:
    """
    Compute BLEU-1 and BLEU-4 exact n-gram matching scores as percentages (0-100).
    Returns:
        (bleu1, bleu4)
    """
    p_tokens = normalize_answer(pred).split()
    g_tokens = [normalize_answer(gold).split()]

    if not p_tokens or not g_tokens[0]:
        return 0.0, 0.0

    if _SMOOTH is not None:
        b1 = sentence_bleu(g_tokens, p_tokens, weights=(1.0, 0, 0, 0), smoothing_function=_SMOOTH) * 100.0
        b4 = sentence_bleu(g_tokens, p_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH) * 100.0
        return round(b1, 2), round(b4, 2)

    # Standard n-gram precision calculation when nltk is absent
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    # 1-gram
    p1 = get_ngrams(p_tokens, 1)
    g1 = set(get_ngrams(g_tokens[0], 1))
    p1_match = sum(1 for ng in p1 if ng in g1)
    b1 = (p1_match / len(p1) * 100.0) if p1 else 0.0

    # 4-gram
    p4 = get_ngrams(p_tokens, 4)
    g4 = set(get_ngrams(g_tokens[0], 4))
    p4_match = sum(1 for ng in p4 if ng in g4)
    b4 = (p4_match / len(p4) * 100.0) if p4 else 0.0

    return round(b1, 2), round(b4, 2)
