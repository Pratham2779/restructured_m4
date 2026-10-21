"""Data loader for LongMemEval benchmark."""
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from m4.types import Session, Turn


def parse_lme_date(d_str: str) -> datetime:
    """Parse LongMemEval date string format, e.g. '2023/04/10 (Mon) 17:50'."""
    try:
        if "(" in d_str and ")" in d_str:
            clean = d_str.split("(")[0].strip() + " " + d_str.split(")")[1].strip()
            return datetime.strptime(clean, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
        return datetime.strptime(d_str.strip(), "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
    except Exception:
        return datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class LongMemEvalItem(BaseModel):
    """An individual LongMemEval question item."""
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: datetime
    sessions: List[Session] = Field(default_factory=list)


def load_longmemeval_dataset(file_path: Optional[str] = None) -> List[LongMemEvalItem]:
    """Load LongMemEval questions and convert haystack sessions into M4 Sessions."""
    if file_path is None:
        candidates = [
            "benchmarks/longmemeval/data/longmemeval_oracle.json",
            os.path.join(os.path.dirname(__file__), "data", "longmemeval_oracle.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(
            f"LongMemEval dataset not found. Please place 'longmemeval_oracle.json' in 'benchmarks/longmemeval/data/'."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items: List[LongMemEvalItem] = []
    for raw in raw_items:
        qid = raw.get("question_id", "")
        qtype = raw.get("question_type", "unknown")
        question = raw.get("question", "")
        answer = str(raw.get("answer", ""))
        q_date = parse_lme_date(raw.get("question_date", ""))

        haystack_sessions = raw.get("haystack_sessions", [])
        haystack_dates = raw.get("haystack_dates", [])
        haystack_ids = raw.get("haystack_session_ids", [])

        sessions: List[Session] = []
        for s_idx, session_turns in enumerate(haystack_sessions):
            s_date_str = haystack_dates[s_idx] if s_idx < len(haystack_dates) else ""
            s_dt = parse_lme_date(s_date_str) if s_date_str else q_date
            s_id = haystack_ids[s_idx] if s_idx < len(haystack_ids) else f"{qid}_s{s_idx}"

            turns: List[Turn] = []
            for t_idx, turn_dict in enumerate(session_turns):
                role = turn_dict.get("role", "user")
                speaker = "User" if role.lower() == "user" else "Assistant"
                text = str(turn_dict.get("content", ""))
                turns.append(Turn(speaker=speaker, text=text))

            sessions.append(
                Session(
                    id=f"{qid}_{s_id}",
                    timestamp=s_dt,
                    turns=turns,
                )
            )

        items.append(
            LongMemEvalItem(
                question_id=qid,
                question_type=qtype,
                question=question,
                answer=answer,
                question_date=q_date,
                sessions=sessions,
            )
        )

    return items
