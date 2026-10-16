"""Data loader and schema definitions for the LoCoMo benchmark dataset."""
import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dateutil import parser as date_parser

from m4.types import Session, Turn


CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}


class LoCoMoQA(BaseModel):
    """A single QA pair from LoCoMo."""
    qa_id: str
    question: str
    answer: str
    category: int
    category_name: str
    evidence: List[str] = Field(default_factory=list)


class LoCoMoConversation(BaseModel):
    """A full LoCoMo conversation containing multiple chronological sessions and QA pairs."""
    sample_id: str
    speaker_a: str = "User"
    speaker_b: str = "Assistant"
    sessions: List[Session]
    qa_items: List[LoCoMoQA]


def parse_locomo_session_time(time_str: Optional[str], fallback_date: datetime) -> datetime:
    """Parse session timestamp string (e.g. '1:56 pm on 8 May, 2023') to UTC datetime."""
    if not time_str:
        return fallback_date
    try:
        dt = date_parser.parse(time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return fallback_date


def load_locomo_dataset(
    file_path: Optional[str] = None,
    exclude_adversarial: bool = True,
) -> List[LoCoMoConversation]:
    """
    Load and parse LoCoMo benchmark conversations.
    
    Args:
        file_path: Path to locomo_benchmark.json. If None, checks default locations.
        exclude_adversarial: If True, filters out Category 5 (Adversarial) QA pairs as done in the paper.
        
    Returns:
        List of parsed LoCoMoConversation objects.
    """
    if file_path is None:
        candidates = [
            "benchmarks/locomo/data/locomo_benchmark.json",
            "legacy/m4_optimized/datasets/locomo_benchmark.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(
            f"LoCoMo benchmark dataset not found. Please place locomo_benchmark.json in 'benchmarks/locomo/data/'."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    conversations: List[LoCoMoConversation] = []

    for item in raw_data:
        sample_id = item.get("sample_id", "unknown")
        raw_conv = item.get("conversation", {})
        speaker_a = raw_conv.get("speaker_a", "Speaker A")
        speaker_b = raw_conv.get("speaker_b", "Speaker B")

        # 1. Parse chronological sessions
        sessions: List[Session] = []
        base_fallback = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Sessions are indexed session_1, session_2, ...
        s_idx = 1
        while True:
            sess_key = f"session_{s_idx}"
            if sess_key not in raw_conv:
                break

            time_key = f"{sess_key}_date_time"
            raw_time = raw_conv.get(time_key)
            fallback_dt = base_fallback + timedelta(days=s_idx * 7)
            session_dt = parse_locomo_session_time(raw_time, fallback_dt)

            raw_turns = raw_conv.get(sess_key, [])
            turns: List[Turn] = []
            for t in raw_turns:
                spk = t.get("speaker", "User")
                txt = t.get("text", "")
                turns.append(Turn(speaker=spk, text=txt))

            sessions.append(
                Session(
                    id=f"{sample_id}_s{s_idx}",
                    timestamp=session_dt,
                    turns=turns,
                )
            )
            s_idx += 1

        # 2. Parse QA items
        raw_qa = item.get("qa", [])
        qa_items: List[LoCoMoQA] = []
        for q_idx, q in enumerate(raw_qa):
            cat = q.get("category", 1)
            if exclude_adversarial and cat == 5:
                # Exclude adversarial category per paper protocol
                continue

            qa_items.append(
                LoCoMoQA(
                    qa_id=f"{sample_id}_qa_{q_idx}",
                    question=str(q.get("question", "")),
                    answer=str(q.get("answer", "")),
                    category=cat,
                    category_name=CATEGORY_NAMES.get(cat, f"category_{cat}"),
                    evidence=q.get("evidence", []),
                )
            )

        conversations.append(
            LoCoMoConversation(
                sample_id=sample_id,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                sessions=sessions,
                qa_items=qa_items,
            )
        )

    return conversations
