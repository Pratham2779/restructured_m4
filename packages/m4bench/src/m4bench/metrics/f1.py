"""Token-level F1 score calculation."""
import string
import re
from typing import Any


def normalize_answer(s: Any) -> str:
    """Normalize text: lowercase, remove punctuation, remove articles, trim whitespace."""
    if s is None:
        return ""
    if isinstance(s, (list, tuple)):
        s = " ".join(str(x) for x in s)
    text = str(s).lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Normalize whitespace
    return " ".join(text.split())


def compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score (harmonic mean of precision and recall) as percentage (0-100)."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()

    if not p_tokens or not g_tokens:
        return 100.0 if p_tokens == g_tokens else 0.0

    common = set(p_tokens) & set(g_tokens)
    num_same = sum(min(p_tokens.count(w), g_tokens.count(w)) for w in common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return round((2.0 * precision * recall / (precision + recall)) * 100.0, 2)
