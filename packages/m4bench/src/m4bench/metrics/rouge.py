"""ROUGE-L (Longest Common Subsequence) metric."""
from m4bench.metrics.f1 import normalize_answer

try:
    from rouge_score import rouge_scorer
    _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except ImportError:
    _ROUGE_SCORER = None


def compute_rouge_l(pred: str, gold: str) -> float:
    """Compute ROUGE-L F-measure as percentage (0-100)."""
    if not pred or not gold:
        return 0.0

    if _ROUGE_SCORER is not None:
        try:
            score = _ROUGE_SCORER.score(gold, pred)
            return round(score["rougeL"].fmeasure * 100.0, 2)
        except Exception:
            pass

    # Exact DP LCS implementation
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return 0.0

    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if p[i] == g[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0

    precision = lcs / m
    recall = lcs / n
    f_measure = (2.0 * precision * recall) / (precision + recall)
    return round(f_measure * 100.0, 2)
