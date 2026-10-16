"""Hermetic tests for MSC benchmark runner and ablation flags."""
import os
import json
import tempfile
import pytest

from m4.ports import LLMClient, Embedder
from m4.adapters.store_memory import InMemoryStore
from benchmarks.msc.protocol import run_benchmark, parse_ablations


class FakeMSCLLM(LLMClient):
    def chat(self, messages, *, temperature=None, json_mode=False, max_tokens=None) -> str:
        content = messages[-1]["content"]

        if "Please analyze the conversation and identify distinct pieces of information to memorize" in content:
            return json.dumps([
                {"Category": "Activities", "Summary": "User went running and walks a deaf dog named Sparky."}
            ])
        if "judge whether there is a conflict or update between them" in content:
            return "0"
        if "determine which categories it is relevant to" in content:
            return json.dumps([{"Category": "Activities"}])
        if "whether you can generate the best response based on the known memories" in content:
            c_low = content.lower()
            if "memories: none" in c_low or "memories:\nnone" in c_low:
                return "0"
            return "1"
        if "Generate the most plausible answer based memories without any reasoning" in content:
            return "Great thanks. What are you doing?"

        return "Great thanks. What are you doing?"


class FakeMSCEmbedder(Embedder):
    dim: int = 4

    def embed(self, texts):
        res = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            res.append([0.7, 0.3, val * 0.1, 0.0])
        return res


def test_msc_smoke_run_hermetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_llm = FakeMSCLLM()
        fake_emb = FakeMSCEmbedder()
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
        assert agg.dataset == "msc"
        assert agg.stage == "smoke"
        assert os.path.exists(os.path.join(tmpdir, "msc", agg.run_id, "manifest.json"))
        assert os.path.exists(os.path.join(tmpdir, "msc", agg.run_id, "items.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "msc", agg.run_id, "aggregate.json"))


def test_ablation_flags_mapping():
    flags = parse_ablations(["chain", "calibrate"])
    assert flags["enable_chain"] is False
    assert flags["enable_calibrate"] is False
    assert flags["enable_compress"] is True
    assert flags["enable_self_reflection"] is True

    flags2 = parse_ablations(["wo_self_reflection", "demonstration"])
    assert flags2["enable_self_reflection"] is False
    assert flags2["enable_demonstration"] is False
    assert flags2["enable_chain"] is True
