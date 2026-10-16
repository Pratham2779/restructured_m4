"""Domain data structures and types for M4 memory framework."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Turn(BaseModel):
    """A single turn in a conversation session."""
    speaker: str
    text: str


class Session(BaseModel):
    """A multi-turn conversation session."""
    id: str
    turns: List[Turn]
    timestamp: Optional[datetime] = None


class Category(BaseModel):
    """Memory category entity."""
    id: str
    name: str
    embedding: List[float] = Field(default_factory=list)


class MemoryNode(BaseModel):
    """
    A memory node in the M4 graph.
    Crucially supports belonging to multiple categories (forming the shared nodes graph).
    """
    id: str
    text: str
    category_ids: List[str] = Field(default_factory=list)
    session_id: str
    timestamp: datetime
    last_used_at: datetime
    embedding: List[float] = Field(default_factory=list)
    compressed_level: int = 0
    calibrated: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result of memory retrieval containing ordered nodes and audit trace."""
    nodes: List[MemoryNode] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)


class Response(BaseModel):
    """Generated response from M4 memory layer."""
    text: str
    retrieval: RetrievalResult


class IngestReport(BaseModel):
    """Summary report produced after ingesting a session."""
    session_id: str
    categories_found: List[str] = Field(default_factory=list)
    nodes_added: List[str] = Field(default_factory=list)
    calibrated_count: int = 0
    compressed_count: int = 0
    total_nodes: int = 0


class StoreStats(BaseModel):
    """Statistics for a namespace memory store."""
    node_count: int = 0
    category_count: int = 0
    chain_count: int = 0
    shared_node_count: int = 0
