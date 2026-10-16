"""In-memory Graph Memory Store adapter implementing MemoryStore protocol."""
from typing import Dict, List, Optional, Tuple, Set
from m4.types import MemoryNode, Category, StoreStats
from m4.utils import cosine_similarity


class InMemoryMemoryStore:
    """
    High-performance in-memory graph store.
    Every operation is strictly isolated by namespace (ns).
    Matches the MemoryStore protocol and passes the store contract test suite.
    """

    def __init__(self):
        # ns -> {cat_id -> Category}
        self._categories: Dict[str, Dict[str, Category]] = {}
        # ns -> {node_id -> MemoryNode}
        self._nodes: Dict[str, Dict[str, MemoryNode]] = {}
        # ns -> {cat_id -> [node_id, ...]} (chronological chain order)
        self._chains: Dict[str, Dict[str, List[str]]] = {}
        # ns -> {prev_node_id -> next_node_id}
        self._next_links: Dict[str, Dict[str, str]] = {}
        # ns -> {next_node_id -> prev_node_id}
        self._prev_links: Dict[str, Dict[str, str]] = {}
        # ns -> {node_id -> set(cat_ids)}
        self._node_categories: Dict[str, Dict[str, Set[str]]] = {}

    def _ensure_ns(self, ns: str):
        if ns not in self._categories:
            self._categories[ns] = {}
            self._nodes[ns] = {}
            self._chains[ns] = {}
            self._next_links[ns] = {}
            self._prev_links[ns] = {}
            self._node_categories[ns] = {}

    def upsert_category(self, ns: str, category: Category) -> None:
        self._ensure_ns(ns)
        self._categories[ns][category.id] = category
        if category.id not in self._chains[ns]:
            self._chains[ns][category.id] = []

    def add_node(self, ns: str, node: MemoryNode, category_ids: List[str]) -> None:
        self._ensure_ns(ns)
        node.category_ids = list(set(node.category_ids + category_ids))
        self._nodes[ns][node.id] = node
        self._node_categories[ns].setdefault(node.id, set()).update(category_ids)

        for cid in category_ids:
            if cid not in self._chains[ns]:
                self._chains[ns][cid] = []
            chain = self._chains[ns][cid]
            if node.id not in chain:
                chain.append(node.id)

    def link_next(self, ns: str, prev_id: str, next_id: str) -> None:
        self._ensure_ns(ns)
        self._next_links[ns][prev_id] = next_id
        self._prev_links[ns][next_id] = prev_id

    def replace_nodes(self, ns: str, old_ids: List[str], new_node: MemoryNode) -> None:
        """
        Replace one or more conflicting nodes with a calibrated node.
        Maintains chain connectivity and appends calibrated node to chain end.
        """
        self._ensure_ns(ns)
        cats_to_update = set(new_node.category_ids)

        # 1. Remove old nodes from chain links
        for oid in old_ids:
            if oid in self._nodes[ns]:
                old_node = self._nodes[ns][oid]
                cats_to_update.update(old_node.category_ids)

            p = self._prev_links[ns].pop(oid, None)
            n = self._next_links[ns].pop(oid, None)

            if p and n:
                self._next_links[ns][p] = n
                self._prev_links[ns][n] = p
            elif p:
                self._next_links[ns].pop(p, None)
            elif n:
                self._prev_links[ns].pop(n, None)

            for cid in list(self._chains[ns].keys()):
                if oid in self._chains[ns][cid]:
                    self._chains[ns][cid].remove(oid)

            self._nodes[ns].pop(oid, None)
            self._node_categories[ns].pop(oid, None)

        # 2. Add and link new calibrated node to end of chains
        self.add_node(ns, new_node, list(cats_to_update))
        for cid in cats_to_update:
            chain = self._chains[ns].get(cid, [])
            if len(chain) > 1:
                prev_id = chain[-2]
                self.link_next(ns, prev_id, new_node.id)

    def update_node_text(self, ns: str, node_id: str, text: str, *, compressed_level: int) -> None:
        self._ensure_ns(ns)
        node = self._nodes[ns].get(node_id)
        if node:
            node.text = text
            node.compressed_level = compressed_level

    def get_chain(self, ns: str, category_id: str) -> List[MemoryNode]:
        self._ensure_ns(ns)
        node_ids = self._chains[ns].get(category_id, [])
        return [self._nodes[ns][nid] for nid in node_ids if nid in self._nodes[ns]]

    def neighbors(self, ns: str, node_id: str) -> Tuple[Optional[MemoryNode], Optional[MemoryNode]]:
        self._ensure_ns(ns)
        prev_id = self._prev_links[ns].get(node_id)
        next_id = self._next_links[ns].get(node_id)
        prev_node = self._nodes[ns].get(prev_id) if prev_id else None
        next_node = self._nodes[ns].get(next_id) if next_id else None
        return prev_node, next_node

    def search_categories(self, ns: str, vector: List[float], top_k: int) -> List[Tuple[Category, float]]:
        self._ensure_ns(ns)
        scored: List[Tuple[Category, float]] = []
        for cat in self._categories[ns].values():
            sim = cosine_similarity(cat.embedding, vector) if cat.embedding else 0.0
            scored.append((cat, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_nodes(self, ns: str, category_id: str, vector: List[float], top_k: int) -> List[Tuple[MemoryNode, float]]:
        self._ensure_ns(ns)
        chain_nodes = self.get_chain(ns, category_id)
        scored: List[Tuple[MemoryNode, float]] = []
        for node in chain_nodes:
            sim = cosine_similarity(node.embedding, vector) if node.embedding else 0.0
            scored.append((node, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def shared_nodes(self, ns: str, node_id: str) -> List[MemoryNode]:
        """Find memory nodes that share categories with the given node (multi-category graph)."""
        self._ensure_ns(ns)
        target_cats = self._node_categories[ns].get(node_id, set())
        if not target_cats:
            return []

        shared: List[MemoryNode] = []
        for nid, cats in self._node_categories[ns].items():
            if nid != node_id and (cats & target_cats):
                shared.append(self._nodes[ns][nid])
        return shared

    def stats(self, ns: str) -> StoreStats:
        self._ensure_ns(ns)
        nodes = self._nodes[ns]
        shared_count = sum(1 for cats in self._node_categories[ns].values() if len(cats) > 1)
        return StoreStats(
            node_count=len(nodes),
            category_count=len(self._categories[ns]),
            chain_count=len([c for c in self._chains[ns].values() if len(c) > 0]),
            shared_node_count=shared_count,
        )

    def clear_namespace(self, ns: str) -> None:
        if ns in self._categories:
            del self._categories[ns]
            del self._nodes[ns]
            del self._chains[ns]
            del self._next_links[ns]
            del self._prev_links[ns]
            del self._node_categories[ns]


InMemoryStore = InMemoryMemoryStore

