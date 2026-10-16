"""Hermetic tests for LoCoMo benchmark runner and protocol."""
import os
import json
import tempfile
import pytest

from m4.ports import LLMClient, Embedder
from m4.adapters.store_memory import InMemoryStore
from benchmarks.locomo.protocol import run_benchmark


class FakeBenchmarkLLM(LLMClient):
    def __init__(self):
        self.call_count = 0

    def chat(self, messages, *, temperature=None, json_mode=False, max_tokens=None) -> str:
        self.call_count += 1
        content = messages[-1]["content"]

        # Category learning JSON response
        if "Please analyze the conversation and identify distinct pieces of information to memorize" in content:
            return json.dumps([
                {"Category": "LGBTQ Support Group", "Summary": "Caroline attended an LGBTQ support group meeting on 7 May 2023."}
            ])
        # Conflict calibration
        if "judge whether there is a conflict or update between them" in content:
            return "0"
        # Query category relevance
        if "determine which categories it is relevant to" in content:
            return json.dumps([{"Category": "LGBTQ Support Group"}])
        # Confidence check
        if "whether you can generate the best response based on the known memories" in content:
            c_low = content.lower()
            if "memories: none" in c_low or "memories:\nnone" in c_low:
                return "0"
            return "1"
        # Final answer generation
        if "Generate the most plausible answer based memories without any reasoning" in content:
            return "7 May 2023"

        return "7 May 2023"


class FakeBenchmarkEmbedder(Embedder):
    dim: int = 4

    def embed(self, texts):
        res = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            res.append([0.8, 0.2, val * 0.1, 0.0])
        return res


def test_locomo_smoke_run_hermetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_llm = FakeBenchmarkLLM()
        fake_emb = FakeBenchmarkEmbedder()
        fake_store = InMemoryStore()

        agg = run_benchmark(
            stage="smoke",
            results_dir=tmpdir,
            llm_client=fake_llm,
            embedder_client=fake_emb,
            store=fake_store,
        )

        assert agg.total_items == 2
        assert "f1" in agg.overall_metrics
        assert agg.dataset == "locomo"
        assert agg.stage == "smoke"
        assert os.path.exists(os.path.join(tmpdir, "locomo", agg.run_id, "manifest.json"))
        assert os.path.exists(os.path.join(tmpdir, "locomo", agg.run_id, "items.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "locomo", agg.run_id, "aggregate.json"))
