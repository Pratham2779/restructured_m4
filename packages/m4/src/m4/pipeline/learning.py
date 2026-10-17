"""Category-Based Memory Learning (Paper Section III.B)."""
import time
from typing import List, Tuple, Dict, Any, Optional
from m4.config import M4Config
from m4.types import Turn
from m4.ports import LLMClient, Embedder, Observer
from m4.utils import load_prompt, parse_json_tolerant, cosine_similarity


class CategoryLearningPipeline:
    """Extracts, disambiguates, and consolidates conversational memory clues by category."""

    def __init__(
        self,
        config: M4Config,
        llm: LLMClient,
        embedder: Embedder,
        observer: Optional[Observer] = None,
    ):
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.observer = observer
        self._prompt_template = load_prompt("fig2_memory_learning.txt")

    def format_conversation(self, turns: List[Turn]) -> str:
        """Format dialogue turns into a legible transcript for the prompt."""
        return "\n".join(f"{t.speaker}: {t.text}" for t in turns)

    def process_session(
        self,
        session_id: str,
        turns: List[Turn],
    ) -> List[Tuple[str, str, List[float], List[float]]]:
        """
        Extract memory items, merge categories with similarity >= gamma,
        and generate one consolidated memory clue per category.
        
        Returns:
            List of tuples: (category_name, clue_summary, category_embedding, clue_embedding)
        """
        if not turns:
            return []

        conv_text = self.format_conversation(turns)
        prompt = self._prompt_template.replace("{Conversation}", conv_text)

        if not self.config.enable_demonstration:
            # Table II ablation: w/o Demonstration (remove sample JSON format)
            prompt = prompt.replace(
                'Example JSON Response:\n"""\n[{"Category": "{YOUR_CATEGORY}", "Summary": "{YOUR_SUMMARY}"},\n {"Category": "{YOUR_CATEGORY}", "Summary": "{YOUR_SUMMARY}"}]\n"""\n\n',
                ""
            )

        messages = [
            {"role": "system", "content": "You are a memory extraction engine. Respond strictly with a JSON array."},
            {"role": "user", "content": prompt},
        ]

        t0 = time.time()
        raw_output = self.llm.chat(
            messages,
            temperature=self.config.temperature,
            json_mode=True,
        )
        elapsed = time.time() - t0
        if self.observer:
            self.observer.on_llm_call(
                purpose="category_learning_extraction",
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(raw_output.split()),
                seconds=elapsed,
            )

        parsed_items = None
        for attempt in range(self.config.max_json_retries):
            try:
                parsed = parse_json_tolerant(raw_output)
                if isinstance(parsed, list):
                    parsed_items = parsed
                    break
                elif isinstance(parsed, dict):
                    for k in ["memories", "memory_items", "items", "clues"]:
                        if k in parsed and isinstance(parsed[k], list):
                            parsed_items = parsed[k]
                            break
                    if parsed_items is None:
                        for v in parsed.values():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                parsed_items = v
                                break
                    if parsed_items is not None:
                        break
            except Exception as err:
                if self.observer:
                    self.observer.on_event("json_parse_error", {"attempt": attempt, "error": str(err), "raw": raw_output[:200]})
                if attempt < self.config.max_json_retries - 1:
                    retry_msg = [
                        {"role": "system", "content": "Fix this into a valid JSON array of objects with keys 'Category' and 'Summary'."},
                        {"role": "user", "content": f"Output:\n{raw_output}\nFix now:"},
                    ]
                    raw_output = self.llm.chat(retry_msg, temperature=0.0, json_mode=True)

        if not parsed_items:
            return []

        # Validate raw items
        candidates: List[Tuple[str, str]] = []
        for item in parsed_items:
            if isinstance(item, dict):
                cat = item.get("Category") or item.get("category")
                summ = item.get("Summary") or item.get("summary")
                if cat and summ:
                    candidates.append((str(cat).strip(), str(summ).strip()))

        if not candidates:
            return []

        # Embed all candidate categories
        cat_names = [c[0] for c in candidates]
        cat_embeddings = self.embedder.embed(cat_names)

        # Merge categories using pairwise cosine similarity >= gamma
        n = len(candidates)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        gamma = self.config.gamma
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(cat_embeddings[i], cat_embeddings[j])
                if sim >= gamma:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Produce one consolidated clue per category
        group_items = []
        for root_idx, member_indices in groups.items():
            canonical_cat = candidates[root_idx][0]
            cat_emb = cat_embeddings[root_idx]

            if len(member_indices) == 1:
                clue_text = candidates[root_idx][1]
            else:
                # Combine summaries of the same category into one coherent clue
                combined_points = " ".join(candidates[idx][1] for idx in member_indices)
                clue_text = combined_points
            group_items.append((canonical_cat, clue_text, cat_emb))

        clue_texts = [item[1] for item in group_items]
        clue_embeddings = self.embedder.embed(clue_texts)

        results: List[Tuple[str, str, List[float], List[float]]] = []
        for (canonical_cat, clue_text, cat_emb), clue_emb in zip(group_items, clue_embeddings):
            results.append((canonical_cat, clue_text, cat_emb, clue_emb))

        return results
