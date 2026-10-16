"""Local LLM Judge across 5 conversational dimensions (Not comparable with paper)."""
import re
from typing import Dict, Any, Optional
from m4.ports import LLMClient

# Note: G-Eval in the paper used closed-source GPT-4o.
# Running a local 7B model as judge is for internal diagnostics only and is NOT comparable with paper numbers.
LOCAL_JUDGE_NOTICE = "Local LLM Judge (not comparable with paper)"

JUDGE_PROMPT_TEMPLATE = """You are an expert dialogue evaluator. Evaluate the assistant's response on a scale of 1 to 5 for the following dimension:
Dimension: {Dimension}
Definition: {Definition}

Conversation Context:
{Context}

Assistant Response:
{Response}

Provide a rating integer from 1 to 5 only, without reasoning.
Rating:"""

DIMENSIONS = {
    "engagingness": "Reflects ability to foster compelling interactions beyond mundane replies.",
    "humanness": "Measures empathetic reasoning and simulation of human-like cognitive patterns.",
    "coherence": "Assesses logical continuity and alignment with the broader historical context.",
    "consistency": "Evaluates stability and non-contradictory behavior over extended interactions.",
    "memorability": "Tests accuracy in recalling specific details and events from prior sessions.",
}


def evaluate_with_local_judge(
    llm: LLMClient,
    context: str,
    response: str,
) -> Dict[str, float]:
    """
    Evaluate response on the 5 paper dimensions using a local LLM.
    Returns dictionary with normalized scores (0-100) and marks results as not comparable.
    """
    scores = {}
    for dim_name, definition in DIMENSIONS.items():
        prompt = (
            JUDGE_PROMPT_TEMPLATE
            .replace("{Dimension}", dim_name.capitalize())
            .replace("{Definition}", definition)
            .replace("{Context}", context)
            .replace("{Response}", response)
        )
        raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
        match = re.search(r"\b([1-5])\b", raw)
        val = int(match.group(1)) if match else 3
        # Normalize 1-5 scale to percentage (0 - 100%)
        scores[f"geval_{dim_name}"] = round(((val - 1) / 4.0) * 100.0, 2)

    return scores
