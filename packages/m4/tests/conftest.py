"""Hermetic test fixtures and deterministic fakes for M4 test suite."""
import json
import pytest
from typing import List, Dict, Optional, Any
from m4.ports import LLMClient, Embedder, Compressor, Observer


class FakeLLMClient(LLMClient):
    """Deterministic fake LLM client returning scripted responses based on keywords."""

    def __init__(self, scripted_responses: Optional[Dict[str, str]] = None):
        self.scripted_responses = scripted_responses or {}
        self.history: List[List[Dict[str, str]]] = []

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None
    ) -> str:
        self.history.append(messages)
        content = messages[-1]["content"]

        # Check explicit scripted keys
        for key, val in self.scripted_responses.items():
            if key in content:
                return val

        # Default scripted responses matching prompt scenarios
        if "Please analyze the conversation and identify distinct pieces of information to memorize" in content:
            return json.dumps([
                {"Category": "Personal Preferences", "Summary": "User loves green tea in the morning."},
                {"Category": "Pets", "Summary": "User adopted a rescue golden retriever named Max."},
            ])
        elif "judge whether there is a conflict or update between them" in content:
            lower = content.lower()
            if "new_summary:" in lower:
                new_sum_text = lower.split("new_summary:")[1]
                if "instead of" in new_sum_text or "switched to" in new_sum_text or "now strictly" in new_sum_text:
                    return "1"
            return "0"
        elif "merge them into a new summary without any reasoning" in content:
            return "User previously drank coffee, but now strictly drinks tea in the morning."
        elif "make an honest judgment as to whether most of the key information in the original text can be restored" in content:
            return "1"
        elif "determine which categories it is relevant to" in content:
            return json.dumps([{"Category": "Personal Preferences"}, {"Category": "Pets"}])
        elif "whether you can generate the best response based on the known memories" in content:
            c_low = content.lower()
            if "memories: none" in c_low or "memories:\nnone" in c_low:
                return "0"
            if "pets" in c_low or "dog" in c_low:
                return "1" if ("max" in c_low or "dog" in c_low or "golden retriever" in c_low) else "0"
            if "tea" in c_low or "drink" in c_low or "coffee" in c_low:
                return "1" if ("tea" in c_low or "coffee" in c_low) else "0"
            return "1"
        elif "Generate the most plausible answer based memories without any reasoning" in content:
            return "Based on your timeline, you drink green tea in the morning and have a dog named Max."

        return "Fake default assistant response."


class FakeEmbedder(Embedder):
    """Deterministic fake dense embedder returning consistent normalized 4D float vectors."""
    dim: int = 4

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Deterministic hash to 4D float
            val = sum(ord(c) for c in text) % 1000 / 1000.0
            t_lower = text.lower()
            if "tea" in t_lower or "preference" in t_lower or "drink" in t_lower:
                vec = [0.9, 0.1, val * 0.1, 0.0]
            elif "pet" in t_lower or "dog" in t_lower or "max" in t_lower:
                vec = [0.1, 0.9, 0.0, val * 0.1]
            elif "work" in t_lower or "job" in t_lower:
                vec = [0.0, 0.1, 0.9, 0.1]
            else:
                vec = [val, val * 0.5, 0.2, 0.1]
            # Normalize
            norm = (sum(x * x for x in vec) ** 0.5) or 1.0
            results.append([x / norm for x in vec])
        return results


class FakeCompressor(Compressor):
    """Deterministic fake compressor cutting words by rate."""

    def compress(self, text: str, rate: float = 0.5) -> str:
        words = text.split()
        if len(words) <= 2:
            return text
        k = max(2, int(len(words) * rate))
        return " ".join(words[:k])


class RecordingObserver(Observer):
    """Telemetry observer recording calls and events for testing assertions."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    def on_llm_call(self, purpose: str, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        self.calls.append({
            "purpose": purpose,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seconds": seconds,
        })

    def on_event(self, name: str, payload: Dict[str, Any]) -> None:
        self.events.append({"name": name, "payload": payload})


@pytest.fixture
def fake_llm():
    return FakeLLMClient()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_compressor():
    return FakeCompressor()


@pytest.fixture
def recording_observer():
    return RecordingObserver()
