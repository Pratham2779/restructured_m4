"""Hermetic tests for LongMemEval benchmark runner and protocol."""
import os
import json
import tempfile
import pytest

from m4.ports import LLMClient, Embedder
from m4.adapters.store_memory import InMemoryStore
from benchmarks.longmemeval.protocol import run_benchmark, parse_ablations


class FakeLMELLM(LLMClient):
    def chat(self, messages, *, temperature=None, json_mode=False, max_tokens=None) -> str:
        content = messages[-1]["content"]

        if "Please analyze the conversation and identify distinct pieces of information to memorize" in content:
            return json.dumps([
                {"Category": "Automotive", "Summary": "User had GPS issues with their new car."}
            ])
        if "judge whether there is a conflict or update between them" in content:
            return "0"
        if "determine which categories it is relevant to" in content:
            return json.dumps([{"Category": "Automotive"}])
        if "whether you can generate the best response based on the known memories" in content:
            return "1"
        if "Generate the most plausible answer based memories without any reasoning" in content:
            return "GPS system not functioning correctly"

        return "GPS system not functioning correctly"


class FakeLMEEmbedder(Embedder):
    dim: int = 4

    def embed(self, texts):
        res = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            res.append([0.8, 0.2, val * 0.1, 0.0])
        return res


def test_longmemeval_smoke_run_hermetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_llm = FakeLMELLM()
        fake_emb = FakeLMEEmbedder()
        fake_store = InMemoryStore()

        agg = run_benchmark(
            stage="smoke",
            results_dir=tmpdir,
            llm_client=fake_llm,
            embedder_client=fake_emb,
            store=fake_store,
            mock=True,
        )

        assert agg.total_items == 2
        assert agg.dataset == "longmemeval"
        assert agg.stage == "smoke"
        assert "accuracy" in agg.overall_metrics
        assert "f1" in agg.overall_metrics
        assert os.path.exists(os.path.join(tmpdir, "longmemeval", agg.run_id, "manifest.json"))
        assert os.path.exists(os.path.join(tmpdir, "longmemeval", agg.run_id, "items.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "longmemeval", agg.run_id, "aggregate.json"))


def test_longmemeval_ablation_flags():
    flags = parse_ablations(["chain", "calibrate"])
    assert flags["enable_chain"] is False
    assert flags["enable_calibrate"] is False
    assert flags["enable_compress"] is True
    assert flags["enable_self_reflection"] is True
