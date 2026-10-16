"""Protocol specifications (ports) for M4 architecture."""
from typing import Protocol, List, Dict, Any, Optional, Tuple, runtime_checkable
from m4.types import MemoryNode, Category, StoreStats


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM generation backends."""
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send chat messages and return assistant response text."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for dense text embedding backends."""
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of input texts."""
        ...


@runtime_checkable
class Compressor(Protocol):
    """Protocol for text compressor backends (e.g. LLMLingua)."""
    def compress(self, text: str, rate: float = 0.5) -> str:
        """Compress input text towards the target compression rate."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """
    Protocol for graph memory stores.
    Every single call is explicitly scoped by a namespace (ns).
    """
    def upsert_category(self, ns: str, category: Category) -> None:
        """Upsert a category node within the given namespace."""
        ...

    def add_node(self, ns: str, node: MemoryNode, category_ids: List[str]) -> None:
        """Add a memory node linked to one or more categories in the namespace."""
        ...

    def link_next(self, ns: str, prev_id: str, next_id: str) -> None:
        """Create a directed NEXT_IN_CHAIN edge between two nodes in the namespace."""
        ...

    def replace_nodes(self, ns: str, old_ids: List[str], new_node: MemoryNode) -> None:
        """
        Replace one or more conflicting nodes with a calibrated node.
        Used by the Calibrate action to maintain chain consistency.
        """
        ...

    def update_node_text(self, ns: str, node_id: str, text: str, *, compressed_level: int) -> None:
        """Update a node's text and compression level in the namespace."""
        ...

    def get_chain(self, ns: str, category_id: str) -> List[MemoryNode]:
        """Retrieve all nodes in a category's chain in chronological order."""
        ...

    def neighbors(self, ns: str, node_id: str) -> Tuple[Optional[MemoryNode], Optional[MemoryNode]]:
        """Return (previous_node, next_node) along the chain for a given node."""
        ...

    def search_categories(self, ns: str, vector: List[float], top_k: int) -> List[Tuple[Category, float]]:
        """Find the most similar categories to a query vector in the namespace."""
        ...

    def search_nodes(self, ns: str, category_id: str, vector: List[float], top_k: int) -> List[Tuple[MemoryNode, float]]:
        """Find the most similar nodes within a category chain to a query vector."""
        ...

    def shared_nodes(self, ns: str, node_id: str) -> List[MemoryNode]:
        """Return nodes that share multiple category memberships with the given node."""
        ...

    def stats(self, ns: str) -> StoreStats:
        """Return graph statistics for the specified namespace."""
        ...

    def clear_namespace(self, ns: str) -> None:
        """Completely purge all nodes, chains, and relationships for the namespace."""
        ...


@runtime_checkable
class Observer(Protocol):
    """Optional telemetry hook used by benchmarks to record calls, tokens, latency, and events."""
    def on_llm_call(self, purpose: str, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        """Recorded after every LLM chat completion."""
        ...

    def on_event(self, name: str, payload: Dict[str, Any]) -> None:
        """Recorded on significant pipeline events (e.g. json_parse_error, calibrate, compress)."""
        ...
