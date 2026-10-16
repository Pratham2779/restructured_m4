"""M4: Meta-Memory for Large Language Models."""
from m4.api import M4Memory
from m4.config import M4Config
from m4.types import (
    Session,
    Turn,
    MemoryNode,
    Category,
    RetrievalResult,
    Response,
    IngestReport,
    StoreStats,
)
from m4.ports import (
    LLMClient,
    Embedder,
    MemoryStore,
    Compressor,
    Observer,
)

__version__ = "1.0.0"

__all__ = [
    "M4Memory",
    "M4Config",
    "Session",
    "Turn",
    "MemoryNode",
    "Category",
    "RetrievalResult",
    "Response",
    "IngestReport",
    "StoreStats",
    "LLMClient",
    "Embedder",
    "MemoryStore",
    "Compressor",
    "Observer",
]
