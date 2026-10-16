"""Self-Monitoring-Based Memory Updating: Calibrate and Compress (Paper Section III.D)."""
import time
import uuid
import re
from datetime import datetime
from typing import List, Optional
from m4.config import M4Config
from m4.types import MemoryNode
from m4.ports import LLMClient, MemoryStore, Compressor, Embedder, Observer
from m4.utils import load_prompt


class MemoryUpdatingPipeline:
    """Implements self-monitoring memory calibration (conflict resolution) and iterative compression."""

    def __init__(
        self,
        config: M4Config,
        llm: LLMClient,
        store: MemoryStore,
        embedder: Embedder,
        compressor: Optional[Compressor] = None,
        observer: Optional[Observer] = None,
    ):
        self.config = config
        self.llm = llm
        self.store = store
        self.embedder = embedder
        self.compressor = compressor
        self.observer = observer

        self._conflict_prompt_tpl = load_prompt("fig3_conflict_detection.txt")
        self._calibrate_prompt_tpl = load_prompt("fig4_memory_calibrating.txt")
        self._compress_prompt_tpl = load_prompt("fig5_compression_detection.txt")

    def calibrate_new_node(self, ns: str, new_node: MemoryNode) -> int:
        """
        Perform consistency check across all chains that the new node belongs to.
        Merges conflicting nodes with new node into an unambiguous node, deletes the conflicting node,
        and links the calibrated node to the end of the chain.
        
        Returns:
            Number of conflicting nodes calibrated.
        """
        if not self.config.enable_calibrate:
            return 0

        calibrated_count = 0

        for cat_id in new_node.category_ids:
            chain_nodes = self.store.get_chain(ns, cat_id)
            if len(chain_nodes) <= 1:
                continue

            # Prior nodes in the chain excluding the newly added node
            prior_nodes = [n for n in chain_nodes if n.id != new_node.id]
            conflicts: List[MemoryNode] = []

            # Step 1: Detect inconsistent nodes via LLM consistency check (Eq. 4)
            for prior in prior_nodes:
                prompt = (
                    self._conflict_prompt_tpl
                    .replace("{Previous_summary}", prior.text)
                    .replace("{New_summary}", new_node.text)
                )
                t0 = time.time()
                resp = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                elapsed = time.time() - t0
                if self.observer:
                    self.observer.on_llm_call("calibrate_conflict_check", len(prompt.split()), len(resp.split()), elapsed)

                clean_resp = resp.strip()
                if "1" in clean_resp:
                    conflicts.append(prior)

            # Step 2: Merge each conflicting node with new_node (Eq. 5-6)
            current_active_node = new_node
            for conflict_node in conflicts:
                merge_prompt = (
                    self._calibrate_prompt_tpl
                    .replace("{Previous_summary}", conflict_node.text)
                    .replace("{New_summary}", current_active_node.text)
                )
                t0 = time.time()
                merged_summary = self.llm.chat(
                    [{"role": "user", "content": merge_prompt}],
                    temperature=self.config.temperature,
                    max_tokens=256,
                )
                elapsed = time.time() - t0
                if self.observer:
                    self.observer.on_llm_call("calibrate_merge", len(merge_prompt.split()), len(merged_summary.split()), elapsed)

                clean_merged = merged_summary.strip()
                merged_emb = self.embedder.embed([clean_merged])[0]

                # Create calibrated node
                calibrated_node = MemoryNode(
                    id=f"cal_{uuid.uuid4().hex[:12]}",
                    text=clean_merged,
                    category_ids=[cat_id],
                    session_id=current_active_node.session_id,
                    timestamp=current_active_node.timestamp,
                    last_used_at=current_active_node.last_used_at,
                    embedding=merged_emb,
                    compressed_level=0,
                    calibrated=True,
                )

                # Replace old conflicting node and current uncalibrated node in chain
                self.store.replace_nodes(ns, [conflict_node.id, current_active_node.id], calibrated_node)
                current_active_node = calibrated_node
                calibrated_count += 1

                if self.observer:
                    self.observer.on_event("calibrated_node", {
                        "category_id": cat_id,
                        "conflict_id": conflict_node.id,
                        "calibrated_id": calibrated_node.id,
                    })

        return calibrated_count

    def compress_dormant_nodes(self, ns: str, now: datetime) -> int:
        """
        Compress distant memory nodes unused for more than the compress_interval.
        Runs iteratively using Compressor (e.g. LLMLingua) with LLM recoverability check (Fig. 5).
        
        Returns:
            Number of successfully compressed nodes.
        """
        if not self.config.enable_compress or not self.compressor:
            return 0

        # Gather all nodes across all chains in this namespace
        # Note: We query chain by chain to inspect node last_used_at
        stats = self.store.stats(ns)
        if stats.node_count == 0:
            return 0

        categories = self.store.search_categories(ns, [0.0] * self.embedder.dim, top_k=1000)
        seen_node_ids = set()
        dormant_nodes: List[MemoryNode] = []

        threshold_days = self.config.compress_interval_days
        for cat, _ in categories:
            chain = self.store.get_chain(ns, cat.id)
            for node in chain:
                if node.id in seen_node_ids:
                    continue
                seen_node_ids.add(node.id)
                idle_days = (now - node.last_used_at).total_seconds() / 86400.0
                if idle_days >= threshold_days:
                    dormant_nodes.append(node)

        compressed_count = 0
        for node in dormant_nodes:
            original_text = node.text
            current_text = node.text
            accepted_text = current_text
            current_level = node.compressed_level

            for round_idx in range(self.config.max_compression_rounds):
                # 1. Compress via compressor adapter
                try:
                    candidate = self.compressor.compress(current_text, rate=self.config.compression_rate)
                except Exception as err:
                    if self.observer:
                        self.observer.on_event("compression_error", {"node_id": node.id, "error": str(err)})
                    break

                if not candidate or candidate.strip() == current_text.strip():
                    break

                # 2. LLM recoverability check (Fig. 5)
                check_prompt = (
                    self._compress_prompt_tpl
                    .replace("{Original_text}", original_text)
                    .replace("{Compressed_text}", candidate)
                )
                t0 = time.time()
                resp = self.llm.chat(
                    [{"role": "user", "content": check_prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                elapsed = time.time() - t0
                if self.observer:
                    self.observer.on_llm_call("compress_check", len(check_prompt.split()), len(resp.split()), elapsed)

                if "1" in resp.strip():
                    # Key meaning is still recoverable -> adopt and continue loop
                    accepted_text = candidate
                    current_text = candidate
                    current_level += 1
                else:
                    # Key meaning lost -> stop and keep previous accepted version
                    break

            if accepted_text != original_text:
                self.store.update_node_text(ns, node.id, accepted_text, compressed_level=current_level)
                compressed_count += 1
                if self.observer:
                    self.observer.on_event("node_compressed", {
                        "node_id": node.id,
                        "rounds": current_level - node.compressed_level,
                        "original_length": len(original_text),
                        "compressed_length": len(accepted_text),
                    })

        return compressed_count
