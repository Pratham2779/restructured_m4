"""Tests for atomic checkpointing, resumption, and manifest management."""
import os
import json
import tempfile
import pytest
from m4bench.runner import BenchmarkRunner
from m4bench.schema import ItemResult


def test_atomic_checkpointing_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id = "test_run_001"
        # 1. Initialize runner and record 2 items
        runner1 = BenchmarkRunner(
            dataset_name="test_dataset",
            stage="smoke",
            run_id=run_id,
            results_base_dir=tmpdir,
        )
        manifest = runner1.init_manifest(["item_1", "item_2", "item_3"])
        assert os.path.exists(runner1.manifest_file)

        item1 = ItemResult(
            item_id="item_1",
            conversation_id="c1",
            prediction="answer 1",
            gold="answer 1",
            metrics={"f1": 100.0},
            tokens_used=50,
            latency_seconds=0.5,
        )
        runner1.record_item(item1)

        item2 = ItemResult(
            item_id="item_2",
            conversation_id="c2",
            prediction="wrong answer",
            gold="correct answer",
            metrics={"f1": 0.0},
            tokens_used=40,
            latency_seconds=0.4,
        )
        runner1.record_item(item2)

        assert runner1.is_item_completed("item_1")
        assert runner1.is_item_completed("item_2")
        assert not runner1.is_item_completed("item_3")

        # 2. Simulate interruption and resumption by initializing a new runner on same dir
        runner2 = BenchmarkRunner(
            dataset_name="test_dataset",
            stage="smoke",
            run_id=run_id,
            results_base_dir=tmpdir,
        )
        # Resumption: runner2 should recognize item_1 and item_2 as completed
        assert runner2.is_item_completed("item_1")
        assert runner2.is_item_completed("item_2")
        assert not runner2.is_item_completed("item_3")

        # Complete item 3
        item3 = ItemResult(
            item_id="item_3",
            conversation_id="c3",
            prediction="answer 3",
            gold="answer 3",
            metrics={"f1": 100.0},
            tokens_used=60,
            latency_seconds=0.6,
        )
        runner2.record_item(item3)

        # Finalize
        agg = runner2.finalize()
        assert agg.total_items == 3
        # Average F1 should be (100 + 0 + 100) / 3 = 66.67
        assert agg.overall_metrics["f1"] == 66.67
        assert agg.avg_tokens_per_item == 50.0

        # Check aggregate.json exists
        assert os.path.exists(runner2.aggregate_file)
        with open(runner2.manifest_file, "r") as mf:
            m_data = json.load(mf)
            assert m_data.get("end_time") is not None
