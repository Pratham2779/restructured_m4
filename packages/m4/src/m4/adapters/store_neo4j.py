"""Neo4j Graph Memory Store adapter implementing MemoryStore protocol."""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from m4.types import MemoryNode, Category, StoreStats
from m4.utils import cosine_similarity


class Neo4jMemoryStore:
    """
    Neo4j implementation of the MemoryStore protocol.
    Every query is strictly scoped by namespace (ns) to provide clean multi-tenant isolation.
    """

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    def connect(self):
        """Establish connection to Neo4j database."""
        if not self._driver:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=50,
                connection_acquisition_timeout=5.0,
            )
            self._driver.verify_connectivity()

    def close(self):
        """Close driver connections."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def _get_session(self):
        if not self._driver:
            self.connect()
        return self._driver.session(database=self.database) if self.database else self._driver.session()

    def upsert_category(self, ns: str, category: Category) -> None:
        query = """
        MERGE (c:MemoryCategory {id: $id, namespace: $ns})
        SET c.name = $name,
            c.embedding = $embedding
        """
        with self._get_session() as session:
            session.run(query, ns=ns, id=category.id, name=category.name, embedding=category.embedding)

    def add_node(self, ns: str, node: MemoryNode, category_ids: List[str]) -> None:
        ts_str = node.timestamp.isoformat() if hasattr(node.timestamp, "isoformat") else str(node.timestamp)
        last_used_str = node.last_used_at.isoformat() if hasattr(node.last_used_at, "isoformat") else str(node.last_used_at)

        query = """
        MERGE (n:MemoryNode {id: $id, namespace: $ns})
        SET n.text = $text,
            n.session_id = $session_id,
            n.timestamp = $timestamp,
            n.last_used_at = $last_used_at,
            n.embedding = $embedding,
            n.compressed_level = $compressed_level,
            n.calibrated = $calibrated

        WITH n
        UNWIND $category_ids AS cid
        MATCH (c:MemoryCategory {id: cid, namespace: $ns})
        MERGE (n)-[:BELONGS_TO {namespace: $ns}]->(c)
        """
        with self._get_session() as session:
            session.run(
                query,
                ns=ns,
                id=node.id,
                text=node.text,
                session_id=node.session_id,
                timestamp=ts_str,
                last_used_at=last_used_str,
                embedding=node.embedding,
                compressed_level=node.compressed_level,
                calibrated=node.calibrated,
                category_ids=category_ids,
            )

    def link_next(self, ns: str, prev_id: str, next_id: str) -> None:
        query = """
        MATCH (p:MemoryNode {id: $prev_id, namespace: $ns})
        MATCH (nxt:MemoryNode {id: $next_id, namespace: $ns})
        MERGE (p)-[:NEXT_IN_CHAIN {namespace: $ns}]->(nxt)
        """
        with self._get_session() as session:
            session.run(query, ns=ns, prev_id=prev_id, next_id=next_id)

    def replace_nodes(self, ns: str, old_ids: List[str], new_node: MemoryNode) -> None:
        """
        Replaces conflicting nodes with a calibrated node, preserving chain links.
        """
        with self._get_session() as session:
            # 1. Fetch categories that old nodes belonged to
            res = session.run(
                """
                MATCH (n:MemoryNode {namespace: $ns})-[:BELONGS_TO]->(c:MemoryCategory)
                WHERE n.id IN $old_ids
                RETURN collect(DISTINCT c.id) AS cat_ids
                """,
                ns=ns,
                old_ids=old_ids,
            ).single()
            cat_ids = res["cat_ids"] if res else []
            combined_cat_ids = list(set(cat_ids + new_node.category_ids))

            # 2. Bridge NEXT_IN_CHAIN relationships around old nodes
            bridge_query = """
            MATCH (n:MemoryNode {namespace: $ns})
            WHERE n.id IN $old_ids
            OPTIONAL MATCH (p:MemoryNode)-[r1:NEXT_IN_CHAIN {namespace: $ns}]->(n)
            OPTIONAL MATCH (n)-[r2:NEXT_IN_CHAIN {namespace: $ns}]->(nxt:MemoryNode)
            FOREACH (_ IN CASE WHEN p IS NOT NULL AND nxt IS NOT NULL AND NOT p.id IN $old_ids AND NOT nxt.id IN $old_ids THEN [1] ELSE [] END |
                MERGE (p)-[:NEXT_IN_CHAIN {namespace: $ns}]->(nxt)
            )
            DETACH DELETE n
            """
            session.run(bridge_query, ns=ns, old_ids=old_ids)

        # 3. Add and link the new calibrated node
        self.add_node(ns, new_node, combined_cat_ids)
        for cid in combined_cat_ids:
            chain = self.get_chain(ns, cid)
            if len(chain) > 1:
                self.link_next(ns, chain[-2].id, new_node.id)

    def update_node_text(self, ns: str, node_id: str, text: str, *, compressed_level: int) -> None:
        query = """
        MATCH (n:MemoryNode {id: $id, namespace: $ns})
        SET n.text = $text,
            n.compressed_level = $compressed_level
        """
        with self._get_session() as session:
            session.run(query, ns=ns, id=node_id, text=text, compressed_level=compressed_level)

    def get_chain(self, ns: str, category_id: str) -> List[MemoryNode]:
        query = """
        MATCH (n:MemoryNode {namespace: $ns})-[:BELONGS_TO]->(c:MemoryCategory {id: $cat_id, namespace: $ns})
        RETURN n.id AS id, n.text AS text, n.session_id AS session_id,
               n.timestamp AS timestamp, n.last_used_at AS last_used_at,
               n.embedding AS embedding, n.compressed_level AS compressed_level,
               n.calibrated AS calibrated
        ORDER BY n.timestamp ASC
        """
        with self._get_session() as session:
            records = session.run(query, ns=ns, cat_id=category_id).data()
            nodes: List[MemoryNode] = []
            for r in records:
                nodes.append(self._record_to_node(r, [category_id]))
            return nodes

    def neighbors(self, ns: str, node_id: str) -> Tuple[Optional[MemoryNode], Optional[MemoryNode]]:
        query = """
        MATCH (n:MemoryNode {id: $node_id, namespace: $ns})
        OPTIONAL MATCH (p:MemoryNode)-[:NEXT_IN_CHAIN {namespace: $ns}]->(n)
        OPTIONAL MATCH (n)-[:NEXT_IN_CHAIN {namespace: $ns}]->(nxt:MemoryNode)
        RETURN p, nxt
        """
        with self._get_session() as session:
            rec = session.run(query, ns=ns, node_id=node_id).single()
            if not rec:
                return None, None
            p_node = self._neo4j_node_to_model(rec["p"]) if rec["p"] else None
            nxt_node = self._neo4j_node_to_model(rec["nxt"]) if rec["nxt"] else None
            return p_node, nxt_node

    def search_categories(self, ns: str, vector: List[float], top_k: int) -> List[Tuple[Category, float]]:
        query = """
        MATCH (c:MemoryCategory {namespace: $ns})
        RETURN c.id AS id, c.name AS name, c.embedding AS embedding
        """
        with self._get_session() as session:
            records = session.run(query, ns=ns).data()
            scored = []
            for r in records:
                cat = Category(id=r["id"], name=r["name"], embedding=r.get("embedding") or [])
                sim = cosine_similarity(cat.embedding, vector) if cat.embedding else 0.0
                scored.append((cat, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def search_nodes(self, ns: str, category_id: str, vector: List[float], top_k: int) -> List[Tuple[MemoryNode, float]]:
        chain = self.get_chain(ns, category_id)
        scored = []
        for n in chain:
            sim = cosine_similarity(n.embedding, vector) if n.embedding else 0.0
            scored.append((n, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def shared_nodes(self, ns: str, node_id: str) -> List[MemoryNode]:
        query = """
        MATCH (n:MemoryNode {id: $node_id, namespace: $ns})-[:BELONGS_TO]->(c:MemoryCategory)<-[:BELONGS_TO]-(other:MemoryNode {namespace: $ns})
        WHERE other.id <> $node_id
        RETURN DISTINCT other.id AS id, other.text AS text, other.session_id AS session_id,
                        other.timestamp AS timestamp, other.last_used_at AS last_used_at,
                        other.embedding AS embedding, other.compressed_level AS compressed_level,
                        other.calibrated AS calibrated
        """
        with self._get_session() as session:
            records = session.run(query, ns=ns, node_id=node_id).data()
            return [self._record_to_node(r, []) for r in records]

    def stats(self, ns: str) -> StoreStats:
        query = """
        MATCH (n:MemoryNode {namespace: $ns})
        WITH count(n) AS node_count
        OPTIONAL MATCH (c:MemoryCategory {namespace: $ns})
        WITH node_count, count(c) AS category_count
        RETURN node_count, category_count
        """
        with self._get_session() as session:
            res = session.run(query, ns=ns).single()
            n_count = res["node_count"] if res else 0
            c_count = res["category_count"] if res else 0
            return StoreStats(node_count=n_count, category_count=c_count, chain_count=c_count, shared_node_count=0)

    def clear_namespace(self, ns: str) -> None:
        query = "MATCH (n {namespace: $ns}) DETACH DELETE n"
        with self._get_session() as session:
            session.run(query, ns=ns)

    def _record_to_node(self, r: Dict[str, Any], cat_ids: List[str]) -> MemoryNode:
        ts = r["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif not isinstance(ts, datetime):
            ts = datetime.now(timezone.utc)

        lu = r["last_used_at"]
        if isinstance(lu, str):
            lu = datetime.fromisoformat(lu)
        elif not isinstance(lu, datetime):
            lu = ts

        return MemoryNode(
            id=r["id"],
            text=r["text"],
            category_ids=cat_ids,
            session_id=r["session_id"],
            timestamp=ts,
            last_used_at=lu,
            embedding=r.get("embedding") or [],
            compressed_level=r.get("compressed_level", 0),
            calibrated=r.get("calibrated", False),
        )

    def _neo4j_node_to_model(self, node) -> MemoryNode:
        props = dict(node)
        return self._record_to_node(props, [])
