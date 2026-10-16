"""Utility functions for M4: prompt loading, tolerant JSON parsing, vector math."""
import os
import re
import json
import math
from typing import Any, List, Optional, Sequence


def get_prompt_path(filename: str) -> str:
    """Return the absolute path to a prompt file in the prompts directory."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "prompts", filename)


def load_prompt(filename: str) -> str:
    """Load text content from a versioned prompt file."""
    path = get_prompt_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_json_tolerant(raw_text: str) -> Any:
    """
    Tolerantly extract and parse JSON structures from LLM response text.
    Handles Markdown code fences, preamble/postamble conversational text,
    single quotes, and basic formatting glitches typical of 7B parameter models.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response text cannot be parsed as JSON")

    text = raw_text.strip()

    # 1. Look for ```json ... ``` or ``` ... ``` code blocks
    fence_match = re.search(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # 2. Extract outermost [ ... ] or { ... }
        bracket_match = re.search(r"([\[\{].*[\]\}])", text, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(1).strip()

    # 3. Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Clean common JSON syntax errors from 7B models
    cleaned = text
    # Replace single quotes with double quotes around keys and strings
    cleaned = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse tolerant JSON from text: {raw_text[:200]}...") from err


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return float(max(-1.0, min(1.0, sim)))
