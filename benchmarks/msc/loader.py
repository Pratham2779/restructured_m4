"""Data loader for Multi-Session Chat (MSC) following the THEANINE protocol."""
import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from m4.types import Session, Turn


class MSCEpisode(BaseModel):
    """
    An MSC evaluation episode formulated under THEANINE protocol:
    - Sessions 1..k-1: Past conversation sessions for long-term memory ingestion.
    - Session k, Turn 0: User input prompt.
    - Session k, Turn 1: Target ground-truth response.
    """
    episode_id: int
    persona1: List[str] = Field(default_factory=list)
    persona2: List[str] = Field(default_factory=list)
    ingestion_sessions: List[Session]
    test_prompt: str
    test_gold: str
    total_sessions: int


def load_msc_sample_ids(sample_ids_path: Optional[str] = None) -> List[int]:
    """Load the frozen 50-episode sample IDs for MSC benchmark and ablation evaluations."""
    if sample_ids_path is None:
        candidates = [
            "benchmarks/msc/sample_ids.json",
            os.path.join(os.path.dirname(__file__), "sample_ids.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                sample_ids_path = c
                break

    if not sample_ids_path or not os.path.exists(sample_ids_path):
        raise FileNotFoundError(
            f"MSC sample_ids.json not found. Please ensure 'benchmarks/msc/sample_ids.json' exists."
        )

    with open(sample_ids_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_msc_dataset(
    file_path: Optional[str] = None,
    sample_ids: Optional[List[int]] = None,
    min_sessions: int = 2,
) -> List[MSCEpisode]:
    """
    Load MSC episodes under the THEANINE protocol.
    
    Args:
        file_path: Path to msc_dataset.json.
        sample_ids: Optional list of episode (dialogue) IDs to filter to. If None, uses frozen 50-episode sample.
        min_sessions: Minimum sessions required per episode (default 2).
        
    Returns:
        List of prepared MSCEpisode evaluation objects.
    """
    if file_path is None:
        candidates = [
            "benchmarks/msc/data/msc_dataset.json",
            "legacy/m4_optimized/datasets/msc_dataset.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(
            f"MSC dataset not found. Please place msc_dataset.json in 'benchmarks/msc/data/'."
        )

    if sample_ids is None:
        sample_ids = load_msc_sample_ids()

    sample_id_set = set(sample_ids) if sample_ids is not None else None

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Group raw session records by dialoug_id
    episodes_raw: Dict[int, List[Dict[str, Any]]] = {}
    for item in raw_data:
        did = item.get("dialoug_id")
        if sample_id_set is not None and did not in sample_id_set:
            continue
        episodes_raw.setdefault(did, []).append(item)

    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    episodes: List[MSCEpisode] = []

    for did in (sample_ids if sample_ids else sorted(episodes_raw.keys())):
        sess_list = episodes_raw.get(did, [])
        if len(sess_list) < min_sessions:
            continue

        # Ingestion sessions: 0 to k-2
        ingestion_sessions: List[Session] = []
        for s_idx, s_data in enumerate(sess_list[:-1]):
            dialogue_turns = s_data.get("dialogue", [])
            speakers = s_data.get("speaker", [])
            turns: List[Turn] = []
            for t_idx, txt in enumerate(dialogue_turns):
                spk = speakers[t_idx] if t_idx < len(speakers) else ("User" if t_idx % 2 == 0 else "Assistant")
                turns.append(Turn(speaker=spk, text=str(txt)))

            injected_dt = base_time + timedelta(days=s_idx * 7)
            ingestion_sessions.append(
                Session(
                    id=f"msc_{did}_s{s_idx}",
                    timestamp=injected_dt,
                    turns=turns,
                )
            )

        # Final test session: k-1
        last_sess = sess_list[-1]
        last_turns = last_sess.get("dialogue", [])
        if len(last_turns) < 2:
            # Need at least user prompt and gold response
            continue

        test_prompt = str(last_turns[0])
        test_gold = str(last_turns[1])

        p1 = last_sess.get("persona1", [])
        p2 = last_sess.get("persona2", [])

        episodes.append(
            MSCEpisode(
                episode_id=did,
                persona1=p1,
                persona2=p2,
                ingestion_sessions=ingestion_sessions,
                test_prompt=test_prompt,
                test_gold=test_gold,
                total_sessions=len(sess_list),
            )
        )

    return episodes
