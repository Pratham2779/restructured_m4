"""MSC evaluation protocol and ablation runner under THEANINE protocol."""
import os
import time
from datetime import timedelta
from typing import Dict, Any, List, Optional

from m4.api import M4Memory
from m4.config import M4Config
from m4.ports import LLMClient, Embedder, MemoryStore, Compressor
from m4.adapters.store_memory import InMemoryStore
from m4.adapters.llm_openai_compat import OpenAICompatLLM
from m4.adapters.embed_ollama import OllamaEmbedder

from m4bench.runner import BenchmarkRunner
from m4bench.schema import ItemResult, AggregateResult
from m4bench.metrics.f1 import compute_f1
from m4bench.metrics.accuracy import compute_accuracy
from m4bench.metrics.bleu import compute_bleu
from m4bench.metrics.rouge import compute_rouge_l
from benchmarks.msc.loader import load_msc_dataset, MSCEpisode


def parse_ablations(ablations: Optional[List[str]]) -> Dict[str, bool]:
    """Map CLI ablation strings to M4Config boolean flags."""
    flags = {
        "enable_chain": True,
        "enable_calibrate": True,
        "enable_compress": True,
        "enable_self_reflection": True,
        "enable_demonstration": True,
    }
    if not ablations:
        return flags

    for a in ablations:
        a_clean = a.lower().replace("-", "_").replace("w/o_", "").replace("wo_", "")
        if "chain" in a_clean:
            flags["enable_chain"] = False
        elif "calibrate" in a_clean:
            flags["enable_calibrate"] = False
        elif "compress" in a_clean:
            flags["enable_compress"] = False
        elif "self_reflection" in a_clean or "reflection" in a_clean:
            flags["enable_self_reflection"] = False
        elif "demonstration" in a_clean:
            flags["enable_demonstration"] = False

    return flags


def run_benchmark(
    stage: str = "smoke",
    workers: int = 1,
    gamma: float = 0.5,
    temperature: float = 0.8,
    model: str = "qwen2.5:7b-instruct",
    embedder: str = "bge-small-en-v1.5",
    seed: int = 42,
    results_dir: str = "results",
    ablations: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
    embedder_client: Optional[Embedder] = None,
    store: Optional[MemoryStore] = None,
    compressor: Optional[Compressor] = None,
    data_path: Optional[str] = None,
    mock: bool = False,
) -> AggregateResult:
    """
    Run the MSC benchmark evaluation per THEANINE protocol.
    
    Stages:
      - smoke: 2 episodes.
      - pilot: 5 episodes.
      - full: 50-episode sample from sample_ids.json.
    """
    episodes = load_msc_dataset(file_path=data_path)

    # Filter by stage
    if stage == "smoke":
        episodes = episodes[:2]
    elif stage == "pilot":
        episodes = episodes[:5]
    elif stage == "full":
        # Uses full 50-episode sample
        episodes = episodes[:50]

    sampled_ids = [f"msc_{ep.episode_id}" for ep in episodes]

    # Initialize Benchmark Runner
    runner = BenchmarkRunner(
        dataset_name="msc",
        stage=stage,
        results_base_dir=results_dir,
        model_name=model,
        embedder_name=embedder,
        gamma=gamma,
        temperature=temperature,
        seed=seed,
        workers=workers,
    )
    runner.init_manifest(sampled_ids)

    # Configure M4 Memory with ablation flags
    ablation_flags = parse_ablations(ablations)
    config = M4Config(
        gamma=gamma,
        temperature=temperature,
        default_model=model,
        default_embedder=embedder,
        **ablation_flags,
    )

    if mock:
        from benchmarks.msc.tests.test_msc_protocol import FakeMSCLLM, FakeMSCEmbedder
        llm = llm_client or FakeMSCLLM()
        emb = embedder_client or FakeMSCEmbedder()
        mem_store = store or InMemoryStore()
    else:
        llm = llm_client or OpenAICompatLLM(
            base_url=os.environ.get("M4_LLM_URL", "http://localhost:11434/v1"),
            model=model,
            temperature=temperature,
        )
        emb = embedder_client or OllamaEmbedder(
            base_url=os.environ.get("M4_EMBED_URL", "http://localhost:11434"),
            model=embedder,
        )
        mem_store = store or InMemoryStore()

    mem = M4Memory(
        config=config,
        llm=llm,
        embedder=emb,
        store=mem_store,
        compressor=compressor,
        observer=runner.observer,
    )

    # Evaluate each episode
    for ep in episodes:
        item_id = f"msc_{ep.episode_id}"
        if runner.is_item_completed(item_id):
            continue

        ns = item_id
        mem.reset(ns)

        # 1. Ingest past sessions (Sessions 1..k-1)
        for session in ep.ingestion_sessions:
            mem.ingest_session(ns, session)

        # 2. Maintain memory if compression is enabled and sessions exist
        if config.enable_compress and compressor and ep.ingestion_sessions:
            last_time = ep.ingestion_sessions[-1].timestamp
            mem.maintain(ns, now=last_time + timedelta(days=14))

        # 3. Query with Session k, Turn 0
        t0 = time.time()
        resp = mem.respond(ns, ep.test_prompt)
        latency = time.time() - t0

        # 4. Compute metrics against Session k, Turn 1 (Gold)
        f1 = compute_f1(resp.text, ep.test_gold)
        acc = compute_accuracy(resp.text, ep.test_gold)
        b1, b4 = compute_bleu(resp.text, ep.test_gold)
        r_l = compute_rouge_l(resp.text, ep.test_gold)

        metrics = {
            "f1": f1,
            "accuracy": acc,
            "bleu_1": b1,
            "bleu_4": b4,
            "rouge_l": r_l,
        }

        item_res = ItemResult(
            item_id=item_id,
            conversation_id=str(ep.episode_id),
            category="theanine_evaluation",
            inputs={"prompt": ep.test_prompt},
            prediction=resp.text,
            gold=ep.test_gold,
            metrics=metrics,
            trace_summary={"node_count": len(resp.retrieval.nodes)},
            tokens_used=runner.observer.total_prompt_tokens + runner.observer.total_completion_tokens,
            latency_seconds=round(latency, 3),
        )
        runner.record_item(item_res, trace_detail=resp.retrieval.trace)

    return runner.finalize()
