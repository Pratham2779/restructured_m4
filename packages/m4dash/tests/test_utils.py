"""Unit tests for m4dash utilities."""
import os
import json
import tempfile
import zipfile
import pytest

from m4dash.utils import discover_runs, load_run_data, load_item_trace, create_run_zip


def test_discover_and_load_run_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "locomo", "run_20261016_120000")
        traces_dir = os.path.join(run_dir, "traces")
        os.makedirs(traces_dir)

        # 1. Manifest
        manifest_data = {
            "run_id": "run_20261016_120000",
            "dataset": "locomo",
            "stage": "pilot",
            "model_name": "qwen2.5:7b-instruct",
            "gamma": 0.5,
        }
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest_data, f)

        # 2. Aggregate
        agg_data = {
            "run_id": "run_20261016_120000",
            "dataset": "locomo",
            "total_items": 1,
            "overall_metrics": {"f1": 85.0},
        }
        with open(os.path.join(run_dir, "aggregate.json"), "w") as f:
            json.dump(agg_data, f)

        # 3. Items
        item_data = {
            "item_id": "item_1",
            "category": "single_hop",
            "prediction": "Paris",
            "gold": "Paris",
            "metrics": {"f1": 100.0},
        }
        with open(os.path.join(run_dir, "items.jsonl"), "w") as f:
            f.write(json.dumps(item_data) + "\n")

        # 4. Trace
        trace_data = {"retrieval_triggered": True, "query_categories": ["Geography"]}
        with open(os.path.join(traces_dir, "item_1.json"), "w") as f:
            json.dump(trace_data, f)

        # Test discover_runs
        discovered = discover_runs(tmpdir)
        assert len(discovered) == 1
        assert discovered[0]["run_id"] == "run_20261016_120000"
        assert discovered[0]["has_aggregate"] is True

        # Test load_run_data
        data = load_run_data(run_dir)
        assert data["manifest"]["model_name"] == "qwen2.5:7b-instruct"
        assert data["aggregate"]["overall_metrics"]["f1"] == 85.0
        assert len(data["items"]) == 1

        # Test load_item_trace
        tr = load_item_trace(run_dir, "item_1")
        assert tr is not None
        assert tr["retrieval_triggered"] is True

        # Test create_run_zip
        zip_bytes = create_run_zip(run_dir)
        assert len(zip_bytes) > 0
        with tempfile.NamedTemporaryFile(suffix=".zip") as tf:
            tf.write(zip_bytes)
            tf.flush()
            with zipfile.ZipFile(tf.name) as zf:
                namelist = zf.namelist()
                assert "manifest.json" in namelist
                assert "aggregate.json" in namelist
                assert "items.jsonl" in namelist
                assert "traces/item_1.json" in namelist
