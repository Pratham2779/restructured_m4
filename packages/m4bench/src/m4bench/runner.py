"""Generic benchmark runner with atomic checkpointing, resume, and execution management."""
import os
import json
import time
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Optional, Callable
from m4bench.schema import RunManifest, ItemResult, AggregateResult
from m4bench.observers import BenchmarkObserver


KNOWN_DEVIATIONS = [
    "Embedder: bge-small-en-v1.5 (384-d) used instead of text-embedding-3-large (3072-d). Cosine similarity distributions differ.",
    "Backbone: Open-source models (Qwen2.5-7B default) run via Ollama/vLLM; compare against paper's Qwen2.5-7B rows.",
    "G-Eval: Local model judge used instead of closed-source GPT-4o (not comparable with paper).",
    "Temperature: 0.80 introduces stochastic variance across runs.",
]


def get_git_sha() -> str:
    """Retrieve current Git commit SHA."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


class BenchmarkRunner:
    """
    Executes benchmark evaluations with per-item atomic checkpointing and resumption.
    Saves outputs to results/<dataset>/<run_id>/ directory.
    """

    def __init__(
        self,
        dataset_name: str,
        stage: str,
        run_id: Optional[str] = None,
        results_base_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: str = "qwen2.5:7b-instruct",
        embedder_name: str = "bge-small-en-v1.5",
        gamma: float = 0.5,
        temperature: float = 0.8,
        seed: int = 42,
        workers: int = 1,
    ):
        self.dataset_name = dataset_name
        self.stage = stage
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = results_base_dir or os.environ.get("RESULTS_DIR", "results")
        self.run_dir = os.path.join(self.base_dir, self.dataset_name, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "traces"), exist_ok=True)

        self.config = config or {}
        self.model_name = model_name
        self.embedder_name = embedder_name
        self.gamma = gamma
        self.temperature = temperature
        self.seed = seed
        self.workers = workers

        self.manifest_file = os.path.join(self.run_dir, "manifest.json")
        self.items_file = os.path.join(self.run_dir, "items.jsonl")
        self.aggregate_file = os.path.join(self.run_dir, "aggregate.json")

        self.observer = BenchmarkObserver()
        self.completed_item_ids: Set[str] = self._load_existing_completed_ids()

    def _load_existing_completed_ids(self) -> Set[str]:
        """Read items.jsonl if restarting an interrupted run and collect completed IDs."""
        ids = set()
        if os.path.exists(self.items_file):
            with open(self.items_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "item_id" in data:
                                ids.add(data["item_id"])
                        except Exception:
                            continue
        return ids

    def init_manifest(self, sampled_ids: List[str]) -> RunManifest:
        """Write or update manifest.json at run initialization."""
        manifest = RunManifest(
            run_id=self.run_id,
            dataset=self.dataset_name,
            stage=self.stage,
            git_sha=get_git_sha(),
            config=self.config,
            model_name=self.model_name,
            embedder_name=self.embedder_name,
            gamma=self.gamma,
            temperature=self.temperature,
            seed=self.seed,
            sampled_item_ids=sampled_ids,
            start_time=datetime.now(timezone.utc).isoformat(),
            known_deviations=KNOWN_DEVIATIONS,
        )
        self.total_target_items = len(sampled_ids)
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        return manifest

    def record_item(self, item: ItemResult, trace_detail: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a single completed item:
        - Append to items.jsonl and flush immediately (atomic checkpointing).
        - Save full trace to traces/{item_id}.json.
        """
        with open(self.items_file, "a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

        self.completed_item_ids.add(item.item_id)
        total_str = f"/{self.total_target_items}" if hasattr(self, "total_target_items") else ""
        print(f"[{len(self.completed_item_ids)}{total_str}] Completed item {item.item_id} (latency: {item.latency_seconds:.2f}s, F1: {item.metrics.get('f1', 0):.1f}%)", flush=True)

        if trace_detail:
            trace_path = os.path.join(self.run_dir, "traces", f"{item.item_id}.json")
            with open(trace_path, "w", encoding="utf-8") as tf:
                json.dump(trace_detail, tf, indent=2)

    def is_item_completed(self, item_id: str) -> bool:
        """Check if item was already processed in a previous attempt (resumption)."""
        return item_id in self.completed_item_ids

    def finalize(self) -> AggregateResult:
        """
        Compute aggregate statistics over all lines in items.jsonl,
        write aggregate.json, and stamp end_time on manifest.json.
        """
        items: List[ItemResult] = []
        if os.path.exists(self.items_file):
            with open(self.items_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(ItemResult.model_validate_json(line))

        total_items = len(items)
        overall_metrics: Dict[str, float] = {}
        per_cat_items: Dict[str, List[ItemResult]] = {}

        total_tokens = sum(it.tokens_used for it in items)
        total_latency = sum(it.latency_seconds for it in items)

        metric_keys = set()
        for it in items:
            metric_keys.update(it.metrics.keys())
            if it.category:
                per_cat_items.setdefault(it.category, []).append(it)

        for k in metric_keys:
            vals = [it.metrics[k] for it in items if k in it.metrics]
            if vals:
                overall_metrics[k] = round(sum(vals) / len(vals), 2)

        per_category_metrics: Dict[str, Dict[str, float]] = {}
        for cat, cat_list in per_cat_items.items():
            per_category_metrics[cat] = {}
            for k in metric_keys:
                c_vals = [it.metrics[k] for it in cat_list if k in it.metrics]
                if c_vals:
                    per_category_metrics[cat][k] = round(sum(c_vals) / len(c_vals), 2)
            per_category_metrics[cat]["count"] = len(cat_list)

        telemetry = self.observer.get_summary()

        agg = AggregateResult(
            run_id=self.run_id,
            dataset=self.dataset_name,
            stage=self.stage,
            total_items=total_items,
            overall_metrics=overall_metrics,
            per_category_metrics=per_category_metrics,
            avg_tokens_per_item=round(total_tokens / max(1, total_items), 2),
            avg_latency_seconds=round(total_latency / max(1, total_items), 2),
            total_llm_calls=telemetry["total_llm_calls"],
            total_prompt_tokens=telemetry["total_prompt_tokens"],
            total_completion_tokens=telemetry["total_completion_tokens"],
            memory_size_tokens_avg=telemetry["avg_memory_size_tokens"],
            known_deviations=KNOWN_DEVIATIONS,
        )

        with open(self.aggregate_file, "w", encoding="utf-8") as f:
            f.write(agg.model_dump_json(indent=2))

        # Update manifest with end time
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
            m_data["end_time"] = datetime.now(timezone.utc).isoformat()
            with open(self.manifest_file, "w", encoding="utf-8") as mf:
                json.dump(m_data, mf, indent=2)

        return agg
