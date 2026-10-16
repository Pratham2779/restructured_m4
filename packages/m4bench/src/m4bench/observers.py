"""Benchmark observers recording tokens, latency, memory size, and events."""
import time
from typing import Dict, Any, List
from m4.ports import Observer


class BenchmarkObserver(Observer):
    """
    Comprehensive observer for benchmark runs.
    Tracks:
    - LLM calls count
    - Prompt and completion tokens per purpose
    - Latency per query
    - Memory size in tokens
    - JSON parse errors and events
    """

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.token_usage_by_purpose: Dict[str, Dict[str, int]] = {}
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.json_parse_failures: int = 0
        self.memory_sizes_per_conversation: Dict[str, int] = {}

    def on_llm_call(self, purpose: str, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        self.calls.append({
            "purpose": purpose,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seconds": seconds,
            "timestamp": time.time(),
        })
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        if purpose not in self.token_usage_by_purpose:
            self.token_usage_by_purpose[purpose] = {"calls": 0, "prompt": 0, "completion": 0}

        self.token_usage_by_purpose[purpose]["calls"] += 1
        self.token_usage_by_purpose[purpose]["prompt"] += prompt_tokens
        self.token_usage_by_purpose[purpose]["completion"] += completion_tokens

    def on_event(self, name: str, payload: Dict[str, Any]) -> None:
        self.events.append({
            "name": name,
            "payload": payload,
            "timestamp": time.time(),
        })
        if name == "json_parse_error":
            self.json_parse_failures += 1

    def record_memory_size(self, conversation_id: str, token_count: int) -> None:
        """Record total memory size in tokens for a conversation (for Fig. 12/13 analysis)."""
        self.memory_sizes_per_conversation[conversation_id] = token_count

    def get_summary(self) -> Dict[str, Any]:
        """Produce an aggregated dictionary of telemetry data."""
        avg_mem_size = (
            sum(self.memory_sizes_per_conversation.values()) / max(1, len(self.memory_sizes_per_conversation))
            if self.memory_sizes_per_conversation else 0.0
        )
        return {
            "total_llm_calls": len(self.calls),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "json_parse_failures": self.json_parse_failures,
            "avg_memory_size_tokens": round(avg_mem_size, 2),
            "token_usage_by_purpose": self.token_usage_by_purpose,
        }
