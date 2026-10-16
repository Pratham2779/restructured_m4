"""Chain-Based Memory Construction (Paper Section III.C)."""
import uuid
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from m4.config import M4Config
from m4.types import MemoryNode, Category
from m4.ports import MemoryStore, Observer


class ChainConstructionPipeline:
    """Organizes memory clues into ordered category chains and constructs shared graphs."""

    def __init__(
        self,
        config: M4Config,
        store: MemoryStore,
        observer: Optional[Observer] = None,
    ):
        self.config = config
        self.store = store
        self.observer = observer

    def construct_chains(
        self,
        ns: str,
        clues: List[Tuple[str, str, List[float], List[float]]],
        session_id: str,
        injected_time: datetime,
    ) -> List[MemoryNode]:
        """
        Link each memory clue into its corresponding category chain.
        
        Args:
            ns: Namespace identifier
            clues: List of (category_name, clue_summary, category_embedding, clue_embedding)
            session_id: Session identifier
            injected_time: Injected timestamp driving the memory timeline
            
        Returns:
            List of created MemoryNode objects.
        """
        created_nodes: List[MemoryNode] = []
        gamma = self.config.gamma

        for cat_name, clue_text, cat_emb, clue_emb in clues:
            # 1. Argmax lookup against existing categories in this namespace
            search_matches = self.store.search_categories(ns, cat_emb, top_k=1)
            target_category_id: str

            if search_matches and search_matches[0][1] >= gamma:
                matched_cat, score = search_matches[0]
                target_category_id = matched_cat.id
            else:
                # Create a new category
                new_cat_id = f"cat_{uuid.uuid4().hex[:8]}"
                new_cat = Category(id=new_cat_id, name=cat_name, embedding=cat_emb)
                self.store.upsert_category(ns, new_cat)
                target_category_id = new_cat_id

            # 2. Create the new MemoryNode with injected time
            node_id = f"mem_{uuid.uuid4().hex[:12]}"
            node = MemoryNode(
                id=node_id,
                text=clue_text,
                category_ids=[target_category_id],
                session_id=session_id,
                timestamp=injected_time,
                last_used_at=injected_time,
                embedding=clue_emb,
                compressed_level=0,
                calibrated=False,
            )

            # 3. Add to store
            self.store.add_node(ns, node, [target_category_id])

            # 4. Link to chain end if chains are enabled (w/o Chain ablation check)
            if self.config.enable_chain:
                existing_chain = self.store.get_chain(ns, target_category_id)
                # If there are prior nodes in this chain, link the preceding one to this new node
                if len(existing_chain) > 1:
                    prev_node = existing_chain[-2]
                    self.store.link_next(ns, prev_node.id, node.id)

            created_nodes.append(node)

        return created_nodes
