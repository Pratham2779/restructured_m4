"""M4 Public Interface Facade (The Only Public Entry Point for M4)."""
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
from m4.config import M4Config
from m4.types import (
    Session,
    Response,
    RetrievalResult,
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
from m4.pipeline.learning import CategoryLearningPipeline
from m4.pipeline.construction import ChainConstructionPipeline
from m4.pipeline.update import MemoryUpdatingPipeline
from m4.pipeline.retrieval import SelfReflectionRetrievalPipeline


class M4Memory:
    """
    Public facade for Meta-Memory for Memory Management (M4).
    Enables any LLM application, chatbot, or benchmark harness to plug M4 in as a long-term memory layer.
    """

    def __init__(
        self,
        config: M4Config,
        llm: LLMClient,
        embedder: Embedder,
        store: MemoryStore,
        compressor: Optional[Compressor] = None,
        observer: Optional[Observer] = None,
    ):
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.compressor = compressor
        self.observer = observer

        # Initialize pipeline stages
        self._learning = CategoryLearningPipeline(config, llm, embedder, observer)
        self._construction = ChainConstructionPipeline(config, store, observer)
        self._updating = MemoryUpdatingPipeline(config, llm, store, embedder, compressor, observer)
        self._retrieval = SelfReflectionRetrievalPipeline(config, llm, store, embedder, observer)

        # Namespace -> simulated session counter (for sessions lacking timestamps)
        self._simulated_steps: Dict[str, int] = {}
        # Base anchor date for simulation
        self._sim_base_date = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def _resolve_session_time(self, ns: str, session: Session) -> datetime:
        """Resolve injected time. Uses real session timestamp if provided, else simulated week step."""
        if session.timestamp is not None:
            return session.timestamp

        # Fallback simulated timeline progression
        current_step = self._simulated_steps.get(ns, 0)
        simulated_time = self._sim_base_date + timedelta(days=current_step * self.config.fallback_simulated_step_days)
        self._simulated_steps[ns] = current_step + 1
        return simulated_time

    def ingest_session(self, namespace: str, session: Session) -> IngestReport:
        """
        Ingest a conversation session:
        1. Extract and consolidate category-based memory clues.
        2. Construct or append to category chains (and link shared graph nodes).
        3. Calibrate consistency against previous nodes in the chains.
        
        Args:
            namespace: Isolation scope (e.g. user_id, episode_id)
            session: Session object containing dialogue turns and optional timestamp
            
        Returns:
            IngestReport summarizing categories, added nodes, and calibration counts.
        """
        injected_time = self._resolve_session_time(namespace, session)

        # 1. Category-based learning
        clues = self._learning.process_session(session.id, session.turns)
        if not clues:
            stats = self.store.stats(namespace)
            return IngestReport(
                session_id=session.id,
                categories_found=[],
                nodes_added=[],
                calibrated_count=0,
                compressed_count=0,
                total_nodes=stats.node_count,
            )

        categories_found = list({c[0] for c in clues})

        # 2. Chain-based construction
        created_nodes = self._construction.construct_chains(
            ns=namespace,
            clues=clues,
            session_id=session.id,
            injected_time=injected_time,
        )

        # 3. Consistency calibration (Self-monitoring)
        calibrated_count = 0
        for node in created_nodes:
            calibrated_count += self._updating.calibrate_new_node(namespace, node)

        stats = self.store.stats(namespace)
        return IngestReport(
            session_id=session.id,
            categories_found=categories_found,
            nodes_added=[n.id for n in created_nodes],
            calibrated_count=calibrated_count,
            compressed_count=0,
            total_nodes=stats.node_count,
        )

    def retrieve(
        self,
        namespace: str,
        query: str,
        dialogue_context: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve contributory memories for query using Algorithm 1.
        
        Args:
            namespace: Isolation scope
            query: User query or utterance
            dialogue_context: Optional current conversational context
            
        Returns:
            RetrievalResult containing chronologically ordered MemoryNodes and search trace.
        """
        return self._retrieval.retrieve(namespace, query, dialogue_context=dialogue_context)

    def respond(
        self,
        namespace: str,
        query: str,
        dialogue_context: Optional[str] = None,
    ) -> Response:
        """
        Generate answer: runs retrieval with self-reflection and synthesizes response.
        
        Args:
            namespace: Isolation scope
            query: User query or utterance
            dialogue_context: Optional current conversational context
            
        Returns:
            Response containing generated text and full retrieval result trace.
        """
        retrieval = self.retrieve(namespace, query, dialogue_context=dialogue_context)
        resp_text = self._retrieval.generate_response(query, retrieval, dialogue_context=dialogue_context)
        return Response(text=resp_text, retrieval=retrieval)

    def maintain(self, namespace: str, now: datetime) -> int:
        """
        Execute memory maintenance (iterative compression on dormant nodes).
        
        Args:
            namespace: Isolation scope
            now: Injected current timestamp
            
        Returns:
            Number of compressed nodes.
        """
        return self._updating.compress_dormant_nodes(namespace, now)

    def reset(self, namespace: str) -> None:
        """Purge all memory data for the given namespace."""
        self.store.clear_namespace(namespace)
        self._simulated_steps.pop(namespace, None)

    def stats(self, namespace: str) -> StoreStats:
        """Get memory graph statistics for the namespace."""
        return self.store.stats(namespace)
