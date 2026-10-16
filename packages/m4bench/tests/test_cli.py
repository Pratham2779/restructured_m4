"""Tests for m4bench CLI commands."""
import os
import json
import tempfile
from click.testing import CliRunner
from m4bench.cli import main
import yaml


def test_cli_report():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "test_run")
        os.makedirs(run_dir)

        manifest_data = {
            "run_id": "test_run",
            "dataset": "locomo",
            "stage": "smoke",
            "git_sha": "abc1234",
            "model_name": "qwen2.5:7b-instruct",
            "embedder_name": "bge-small-en-v1.5",
            "gamma": 0.5,
            "temperature": 0.8,
            "seed": 42,
            "sampled_item_ids": ["1"],
            "start_time": "2026-10-16T12:00:00Z",
            "end_time": "2026-10-16T12:05:00Z",
        }
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest_data, f)

        agg_data = {
            "run_id": "test_run",
            "dataset": "locomo",
            "stage": "smoke",
            "total_items": 1,
            "overall_metrics": {"f1": 75.0, "rouge_1": 60.0},
            "per_category_metrics": {"single_hop": {"count": 1, "f1": 75.0}},
            "avg_tokens_per_item": 120.0,
            "avg_latency_seconds": 1.5,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 20,
            "memory_size_tokens_avg": 50.0,
            "known_deviations": ["Test deviation"],
        }
        with open(os.path.join(run_dir, "aggregate.json"), "w") as f:
            json.dump(agg_data, f)

        expected_data = {
            "dataset": "locomo",
            "metrics": {
                "f1": 70.0,
                "rouge_1": 65.0,
            }
        }
        expected_path = os.path.join(run_dir, "expected.yaml")
        with open(expected_path, "w") as f:
            yaml.dump(expected_data, f)

        result = runner.invoke(main, ["report", run_dir, "--expected-file", expected_path])
        assert result.exit_code == 0
        assert "M4 Benchmark Run Report" in result.output
        assert "PASS" in result.output
        assert "DIFF" in result.output
