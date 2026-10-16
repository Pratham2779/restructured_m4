"""Data schemas for benchmark manifests, items, and aggregation."""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    """Manifest describing complete execution metadata and environment configuration."""
    run_id: str
    dataset: str
    stage: str = Field(description="'smoke', 'pilot', or 'full'")
    git_sha: str
    config: Dict[str, Any] = Field(default_factory=dict)
    model_name: str
    embedder_name: str
    gamma: float
    temperature: float
    seed: int
    sampled_item_ids: List[str] = Field(default_factory=list)
    start_time: str
    end_time: Optional[str] = None
    known_deviations: List[str] = Field(default_factory=list)


class ItemResult(BaseModel):
    """Result of an individual evaluated question or session response."""
    item_id: str
    conversation_id: str
    category: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    prediction: str
    gold: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    trace_summary: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    latency_seconds: float = 0.0


class AggregateResult(BaseModel):
    """Final aggregated metrics across an entire benchmark run."""
    run_id: str
    dataset: str
    stage: str
    total_items: int
    overall_metrics: Dict[str, float] = Field(default_factory=dict)
    per_category_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    avg_tokens_per_item: float = 0.0
    avg_latency_seconds: float = 0.0
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    memory_size_tokens_avg: float = 0.0
    known_deviations: List[str] = Field(default_factory=list)
