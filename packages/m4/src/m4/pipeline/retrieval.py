"""Self-Reflection-Based Memory Retrieval (Paper Section III.E & Algorithm 1)."""
import time
from typing import List, Tuple, Dict, Any, Optional
from m4.config import M4Config
from m4.types import MemoryNode, Category, RetrievalResult
from m4.ports import LLMClient, MemoryStore, Embedder, Observer
from m4.utils import load_prompt, parse_json_tolerant, cosine_similarity


class SelfReflectionRetrievalPipeline:
    """Implements Algorithm 1: Self-reflective bidirectional chain traversal and response generation."""

    def __init__(
        self,
        config: M4Config,
        llm: LLMClient,
        store: MemoryStore,
        embedder: Embedder,
        observer: Optional[Observer] = None,
    ):
        self.config = config
        self.llm = llm
        self.store = store
        self.embedder = embedder
        self.observer = observer

        self._cat_retrieval_tpl = load_prompt("fig6_category_retrieval.txt")
        self._self_eval_tpl = load_prompt("fig7_self_evaluation.txt")
        self._resp_gen_tpl = load_prompt("fig8_response_generation.txt")

    def _is_confident(self, query: str, memories: List[MemoryNode]) -> bool:
        """
        LLM_confident(q, M): Determine whether LLM is confident to generate the best response.
        Uses Fig. 7 prompt. Returns True if confident (output '1'), False if needs more memories ('0').
        """
        mem_text = "\n".join(f"- {m.text}" for m in memories) if memories else "None."
        prompt = (
            self._self_eval_tpl
            .replace("{Query/Utterance}", query)
            .replace("{Memories}", mem_text)
        )
        t0 = time.time()
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        elapsed = time.time() - t0
        if self.observer:
            self.observer.on_llm_call("retrieval_confidence_check", len(prompt.split()), len(resp.split()), elapsed)

        return "1" in resp.strip()

    def _llm_re(self, query: str, current_memories: List[MemoryNode], candidate: MemoryNode) -> bool:
        """
        LLM_re(c): Evaluate whether adding candidate node improves response quality.
        Returns True if candidate should be adopted, False if skipped.
        """
        # If self-reflection ablation is active, adopt candidate
        if not self.config.enable_self_reflection:
            return True

        # Evaluate response confidence with candidate added
        combined = current_memories + [candidate]
        # Sort chronologically to preserve evolution
        combined_sorted = sorted(combined, key=lambda n: n.timestamp)
        mem_text = "\n".join(f"- {m.text}" for m in combined_sorted)
        prompt = (
            self._self_eval_tpl
            .replace("{Query/Utterance}", query)
            .replace("{Memories}", mem_text)
        )
        t0 = time.time()
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        elapsed = time.time() - t0
        if self.observer:
            self.observer.on_llm_call("retrieval_node_self_reflection", len(prompt.split()), len(resp.split()), elapsed)

        # If LLM indicates 1, it recognizes this memory satisfies/improves the state
        # In Algorithm 1, self-evaluation determines whether candidate is contributory
        return "1" in resp.strip() or len(current_memories) == 0

    def extract_query_categories(self, query: str) -> List[str]:
        """Extract candidate categories for the query using Fig. 6 prompt."""
        prompt = self._cat_retrieval_tpl.replace("{Query/Utterance}", query)
        if not self.config.enable_demonstration:
            prompt = prompt.replace(
                'Example JSON Response:\n"""\n[{"Category": "{YOUR_CATEGORY}"},\n {"Category": "{YOUR_CATEGORY}"}]\n"""\n',
                ""
            )

        messages = [
            {"role": "system", "content": "You are a category extraction engine. Respond strictly with a JSON array."},
            {"role": "user", "content": prompt},
        ]
        t0 = time.time()
        raw = self.llm.chat(messages, temperature=0.0, json_mode=True)
        elapsed = time.time() - t0
        if self.observer:
            self.observer.on_llm_call("retrieval_category_extraction", len(prompt.split()), len(raw.split()), elapsed)

        try:
            parsed = parse_json_tolerant(raw)
            if isinstance(parsed, dict):
                for k in ["categories", "Category", "category", "items"]:
                    if k in parsed and isinstance(parsed[k], list):
                        parsed = parsed[k]
                        break
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            parsed = v
                            break
            if isinstance(parsed, list):
                cats = []
                for item in parsed:
                    if isinstance(item, dict):
                        c = item.get("Category") or item.get("category")
                        if c:
                            cats.append(str(c).strip())
                    elif isinstance(item, str):
                        cats.append(item.strip())
                return cats
        except Exception as err:
            if self.observer:
                self.observer.on_event("category_parse_error", {"raw": raw[:200], "error": str(err)})

        return []

    def retrieve(
        self,
        ns: str,
        query: str,
        dialogue_context: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Execute Self-Reflection-Based Memory Retrieval (Algorithm 1).
        
        Args:
            ns: Namespace
            query: User utterance or question
            dialogue_context: Optional immediate conversation context
            
        Returns:
            RetrievalResult containing chronologically ordered nodes and trace.
        """
        full_query = f"{dialogue_context}\nQuery: {query}" if dialogue_context else query
        trace: List[Dict[str, Any]] = []

        # Check if retrieval is needed: Paper Section III.E:
        # "Afterwards, when the LLM lacks confidence in responding to the current query q, M4 conducts..."
        if self._is_confident(full_query, []):
            trace.append({"step": "initial_confidence_check", "status": "confident_without_retrieval"})
            return RetrievalResult(nodes=[], trace=trace)

        # 1. Category extraction & semantic mapping
        extracted_cat_names = self.extract_query_categories(query)
        trace.append({"step": "extracted_categories", "categories": extracted_cat_names})

        matched_categories: List[Category] = []
        q_vec = self.embedder.embed([query])[0]

        if extracted_cat_names:
            cat_vecs = self.embedder.embed(extracted_cat_names)
            for c_name, c_vec in zip(extracted_cat_names, cat_vecs):
                matches = self.store.search_categories(ns, c_vec, top_k=2)
                for cat, sim in matches:
                    if sim >= self.config.gamma and cat not in matched_categories:
                        matched_categories.append(cat)

        # Fallback: If no categories exceeded gamma, search by query embedding directly
        if not matched_categories:
            top_cats = self.store.search_categories(ns, q_vec, top_k=3)
            matched_categories = [c for c, _ in top_cats]

        trace.append({"step": "matched_categories", "categories": [c.name for c in matched_categories]})

        # If w/o Self-reflection ablation is active (Table II):
        # Directly gather all query-related memory clues from matched categories without self-reflection
        if not self.config.enable_self_reflection:
            direct_nodes: List[MemoryNode] = []
            for cat in matched_categories:
                direct_nodes.extend(self.store.get_chain(ns, cat.id))
            # Sort chronologically
            direct_nodes = sorted(direct_nodes, key=lambda n: n.timestamp)
            trace.append({"step": "ablation_without_self_reflection", "node_count": len(direct_nodes)})
            return RetrievalResult(nodes=direct_nodes, trace=trace)

        # 2. Algorithm 1: Bidirectional Chain Traversal
        retrieved_nodes: List[MemoryNode] = []
        visited_node_ids = set()

        for cat in matched_categories:
            # Check confidence before starting chain
            if self._is_confident(full_query, retrieved_nodes):
                trace.append({"step": "chain_start_check", "category": cat.name, "confident": True})
                break

            chain = self.store.get_chain(ns, cat.id)
            if not chain:
                continue

            # Root node selection: k = argmax_j sim(c_j, q)
            sims = [cosine_similarity(n.embedding, q_vec) for n in chain]
            k = int(max(range(len(sims)), key=lambda idx: sims[idx]))
            root_node = chain[k]

            if root_node.id not in visited_node_ids:
                retrieved_nodes.append(root_node)
                visited_node_ids.add(root_node.id)
                trace.append({"step": "root_node_selected", "node_id": root_node.id, "category": cat.name})

            # Check confidence after root node
            if self._is_confident(full_query, retrieved_nodes):
                trace.append({"step": "root_node_confidence", "confident": True})
                break

            # Pointers for bidirectional traversal
            p_f = k + 1
            p_b = k - 1
            steps = 0

            while steps < self.config.max_retrieval_steps:
                steps += 1
                forward_node = chain[p_f] if p_f < len(chain) else None
                backward_node = chain[p_b] if p_b >= 0 else None

                # Forward inspection
                if forward_node and forward_node.id not in visited_node_ids:
                    if self._llm_re(full_query, retrieved_nodes, forward_node):
                        retrieved_nodes.append(forward_node)
                        visited_node_ids.add(forward_node.id)
                        trace.append({"step": "adopt_forward", "node_id": forward_node.id})
                        if self._is_confident(full_query, retrieved_nodes):
                            break

                # Backward inspection
                if backward_node and backward_node.id not in visited_node_ids:
                    if self._llm_re(full_query, retrieved_nodes, backward_node):
                        retrieved_nodes.insert(0, backward_node)
                        visited_node_ids.add(backward_node.id)
                        trace.append({"step": "adopt_backward", "node_id": backward_node.id})
                        if self._is_confident(full_query, retrieved_nodes):
                            break

                # Advance traversal pointers
                if p_f < len(chain):
                    p_f += 1
                if p_b >= 0:
                    p_b -= 1

                # Stop if both pointers are out of bounds
                if p_f >= len(chain) and p_b < 0:
                    break

        # 3. Shared nodes jump if chains exhausted and still not confident
        if not self._is_confident(full_query, retrieved_nodes) and retrieved_nodes:
            for active_node in list(retrieved_nodes):
                shared = self.store.shared_nodes(ns, active_node.id)
                for s_node in shared:
                    if s_node.id not in visited_node_ids:
                        if self._llm_re(full_query, retrieved_nodes, s_node):
                            retrieved_nodes.append(s_node)
                            visited_node_ids.add(s_node.id)
                            trace.append({"step": "adopt_shared_node", "node_id": s_node.id})
                            if self._is_confident(full_query, retrieved_nodes):
                                break

        # Ensure retrieved memory stays in original chronological order (Paper Section III.E)
        ordered_retrieval = sorted(retrieved_nodes, key=lambda n: n.timestamp)
        return RetrievalResult(nodes=ordered_retrieval, trace=trace)

    def generate_response(
        self,
        query: str,
        retrieval: RetrievalResult,
        dialogue_context: Optional[str] = None,
    ) -> str:
        """
        Generate final coherent answer using retrieved memories (Paper Fig. 8 prompt).
        """
        mem_lines = []
        for m in retrieval.nodes:
            mem_lines.append(f"[{m.timestamp.strftime('%Y-%m-%d %H:%M') if m.timestamp else 'Timeline'}] {m.text}")

        mem_text = "\n".join(mem_lines) if mem_lines else "No historical memories retrieved."
        query_context = f"{dialogue_context}\nCurrent Query: {query}" if dialogue_context else query

        prompt = (
            self._resp_gen_tpl
            .replace("{Query/Utterance}", query_context)
            .replace("{Memories}", mem_text)
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        t0 = time.time()
        response_text = self.llm.chat(
            messages,
            temperature=self.config.temperature,
            max_tokens=512,
        )
        elapsed = time.time() - t0
        if self.observer:
            self.observer.on_llm_call("response_generation", len(prompt.split()), len(response_text.split()), elapsed)

        return response_text.strip()
