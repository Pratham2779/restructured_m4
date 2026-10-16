"""Unit tests for m4bench NLP and QA evaluation metrics."""
import pytest
from m4bench.metrics.f1 import compute_f1
from m4bench.metrics.accuracy import compute_accuracy
from m4bench.metrics.bleu import compute_bleu
from m4bench.metrics.rouge import compute_rouge_l


def test_token_f1():
    # Exact match
    assert compute_f1("The quick brown fox", "the quick brown fox") == 100.0
    # Complete mismatch
    assert compute_f1("apple orange", "car truck") == 0.0
    # Partial overlap
    # Pred: "quick fox" (2 words) -> precision 2/2 = 1.0, Gold: "quick brown fox" (3 words) -> recall 2/3 = 0.666...
    # F1 = 2 * (1.0 * 2/3) / (1.0 + 2/3) = (4/3) / (5/3) = 4/5 = 80.0
    f1 = compute_f1("quick fox", "quick brown fox")
    assert round(f1, 1) == 80.0


def test_accuracy():
    assert compute_accuracy("Yes", "yes") == 100.0
    assert compute_accuracy("Option A", "option a.") == 100.0
    assert compute_accuracy("Paris", "London") == 0.0


def test_bleu():
    b1, b4 = compute_bleu("The cat sat on the mat", "The cat sat on the mat")
    assert b1 == 100.0
    assert b4 == 100.0


def test_rouge():
    score = compute_rouge_l("The cat sat on the mat", "The cat sat on the mat")
    assert score == 100.0
