"""Configuration for M4 (Meta-Memory for Memory Management)."""
from typing import Optional
from pydantic import BaseModel, Field


class M4Config(BaseModel):
    """Configuration parameters for M4 memory framework."""
    # Core mathematical and algorithmic parameters from paper
    gamma: float = Field(default=0.5, description="Category similarity threshold (gamma in paper)")
    temperature: float = Field(default=0.8, description="Sampling temperature for LLM generation")
    
    # Compression intervals
    compress_interval_days: int = Field(
        default=7,
        description="Threshold of inactivity before a node is compressed (default 1 week / 7 days)"
    )
    fallback_simulated_step_days: int = Field(
        default=7,
        description="Simulated elapsed days per session when timestamps are absent (fallback step progression)"
    )
    compression_rate: float = Field(
        default=0.5,
        description="Iterative prompt compression target rate per iteration"
    )
    max_compression_rounds: int = Field(
        default=5,
        description="Maximum iterative compression passes before halting"
    )
    
    # Traversal & safety bounds
    max_retrieval_steps: int = Field(
        default=100,
        description="Safety bound on bidirectional chain traversal steps to prevent infinite loops"
    )
    max_json_retries: int = Field(
        default=3,
        description="Number of retry attempts when parsing ill-formed JSON from 7B models"
    )
    
    # Ablation toggles (mirroring paper Table II)
    enable_chain: bool = Field(default=True, description="Enable ordered memory chains (False for w/o Chain)")
    enable_calibrate: bool = Field(default=True, description="Enable consistency calibration (False for w/o Calibrate)")
    enable_compress: bool = Field(default=True, description="Enable memory compression (False for w/o Compress)")
    enable_self_reflection: bool = Field(default=True, description="Enable retrieval self-reflection (False for w/o Self-reflection)")
    enable_demonstration: bool = Field(default=True, description="Include format demonstrations in prompts (False for w/o Demonstration)")

    model_config = {"extra": "ignore"}
