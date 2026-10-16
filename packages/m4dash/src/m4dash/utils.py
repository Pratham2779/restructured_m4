"""File utilities for m4dash dashboard (zero dependencies on m4/m4bench)."""
import os
import json
import io
import zipfile
from typing import Dict, Any, List, Optional
import yaml


def discover_runs(base_dir: str = "results") -> List[Dict[str, Any]]:
    """Scan base_dir for benchmark run directories containing manifest.json."""
    discovered = []
    if not os.path.exists(base_dir):
        return discovered

    for root, dirs, files in os.walk(base_dir):
        if "manifest.json" in files:
            manifest_path = os.path.join(root, "manifest.json")
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                run_id = manifest.get("run_id", os.path.basename(root))
                dataset = manifest.get("dataset", "unknown")
                has_agg = os.path.exists(os.path.join(root, "aggregate.json"))
                has_items = os.path.exists(os.path.join(root, "items.jsonl"))

                discovered.append({
                    "run_id": run_id,
                    "dataset": dataset,
                    "stage": manifest.get("stage", "unknown"),
                    "run_dir": root,
                    "manifest": manifest,
                    "has_aggregate": has_agg,
                    "has_items": has_items,
                })
            except Exception:
                continue

    # Sort descending by run_id / timestamp
    discovered.sort(key=lambda x: x["run_id"], reverse=True)
    return discovered


def load_run_data(run_dir: str) -> Dict[str, Any]:
    """Load all relevant artifacts from a run directory."""
    manifest_path = os.path.join(run_dir, "manifest.json")
    aggregate_path = os.path.join(run_dir, "aggregate.json")
    items_path = os.path.join(run_dir, "items.jsonl")

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    aggregate = {}
    if os.path.exists(aggregate_path):
        with open(aggregate_path, "r", encoding="utf-8") as f:
            aggregate = json.load(f)

    items = []
    if os.path.exists(items_path):
        with open(items_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue

    # Locate expected.yaml
    dataset = manifest.get("dataset") or aggregate.get("dataset", "")
    expected = {}
    candidates = [
        os.path.join(run_dir, "expected.yaml"),
        os.path.join("benchmarks", dataset, "expected.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    expected = yaml.safe_load(f) or {}
                break
            except Exception:
                continue

    return {
        "manifest": manifest,
        "aggregate": aggregate,
        "items": items,
        "expected": expected,
        "run_dir": run_dir,
    }


def load_item_trace(run_dir: str, item_id: str) -> Optional[Dict[str, Any]]:
    """Load trace JSON for an individual evaluated item."""
    trace_path = os.path.join(run_dir, "traces", f"{item_id}.json")
    if os.path.exists(trace_path):
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def create_run_zip(run_dir: str) -> bytes:
    """Pack run directory files into an in-memory zip archive for download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, run_dir)
                zf.write(abs_path, arcname=rel_path)
    return buf.getvalue()
