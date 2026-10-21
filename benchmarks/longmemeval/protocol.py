"""LongMemEval evaluation protocol under M4 benchmark harness."""
import os
import time
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
from benchmarks.longmemeval.loader import load_longmemeval_dataset, LongMemEvalItem


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
    embedder: str = "3072",
    seed: int = 42,
    results_dir: str = "results",
    ablations: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    embedder_client: Optional[Embedder] = None,
    store: Optional[MemoryStore] = None,
    compressor: Optional[Compressor] = None,
    data_path: Optional[str] = None,
    mock: bool = False,
) -> AggregateResult:
    """
    Run LongMemEval benchmark evaluation.
    
    Stages:
      - smoke: 2 items
      - pilot: 20 items
      - full: 500 items
    """
    items = load_longmemeval_dataset(file_path=data_path)

    if stage == "smoke":
        items = items[:2]
    elif stage == "pilot":
        items = items[:20]
    elif stage == "full":
        items = items[:500]

    sampled_ids = [f"lme_{it.question_id}" for it in items]

    # Initialize Benchmark Runner
    runner = BenchmarkRunner(
        dataset_name="longmemeval",
        stage=stage,
        run_id=run_id,
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
        from benchmarks.longmemeval.tests.test_longmemeval_protocol import FakeLMELLM, FakeLMEEmbedder
        llm = llm_client or FakeLMELLM()
        emb = embedder_client or FakeLMEEmbedder()
        mem_store = store or InMemoryStore()
    else:
        llm = llm_client or OpenAICompatLLM(
            base_url=os.environ.get("M4_LLM_URL", "http://localhost:11434/v1"),
            model=model,
            temperature=temperature,
        )
        if embedder_client:
            emb = embedder_client
        elif embedder in ("3072", "mxbai-3072", "SeanLee97/mxbai-embed-large-v1-nli-matryoshka") or "/" in embedder or os.environ.get("M4_EMBED_BACKEND") == "sentence_transformers":
            from m4.adapters.embed_sentence_transformers import SentenceTransformersEmbedder
            model_name = "SeanLee97/mxbai-embed-large-v1-nli-matryoshka" if embedder in ("3072", "mxbai-3072") else embedder
            emb = SentenceTransformersEmbedder(model_name=model_name)
            print(f"[Embedder] Using SentenceTransformers adapter: {model_name} (dim: {emb.dim}, device: {emb.device})", flush=True)
        else:
            emb = OllamaEmbedder(
                base_url=os.environ.get("M4_EMBED_URL", "http://localhost:11434"),
                model=embedder,
            )
            print(f"[Embedder] Using Ollama adapter: {embedder}", flush=True)

        if store is not None:
            mem_store = store
        elif os.environ.get("M4_STORE_BACKEND") == "neo4j" and os.environ.get("NEO4J_URI"):
            from m4.adapters.store_neo4j import Neo4jMemoryStore
            mem_store = Neo4jMemoryStore(
                uri=os.environ["NEO4J_URI"],
                user=os.environ.get("NEO4J_USER", "neo4j"),
                password=os.environ.get("NEO4J_PASSWORD", "password"),
            )
            print(f"[Store] Using Neo4j Graph DB: {os.environ['NEO4J_URI']}", flush=True)
        else:
            mem_store = InMemoryStore()
            print("[Store] Using high-performance InMemoryStore", flush=True)

    mem = M4Memory(
        config=config,
        llm=llm,
        embedder=emb,
        store=mem_store,
        compressor=compressor,
        observer=runner.observer,
    )

    pending_items = [it for it in items if not runner.is_item_completed(f"lme_{it.question_id}")]

    def process_item(item: LongMemEvalItem) -> None:
        item_id = f"lme_{item.question_id}"
        if runner.is_item_completed(item_id):
            return

        ns = item_id
        mem.reset(ns)

        # 1. Ingest haystack sessions
        for session in item.sessions:
            mem.ingest_session(ns, session)

        # 2. Query with the question
        t0 = time.time()
        resp = mem.respond(ns, item.question)
        latency = time.time() - t0

        # 3. Compute metrics against gold answer
        acc = compute_accuracy(resp.text, item.answer)
        f1 = compute_f1(resp.text, item.answer)
        b1, b4 = compute_bleu(resp.text, item.answer)
        r_l = compute_rouge_l(resp.text, item.answer)

        metrics = {
            "accuracy": acc,
            "f1": f1,
            "bleu_1": b1,
            "bleu_4": b4,
            "rouge_l": r_l,
        }

        item_res = ItemResult(
            item_id=item_id,
            conversation_id=item.question_id,
            category=item.question_type,
            inputs={"question": item.question, "date": str(item.question_date)},
            prediction=resp.text,
            gold=item.answer,
            metrics=metrics,
            trace_summary={"node_count": len(resp.retrieval.nodes)},
            tokens_used=runner.observer.total_prompt_tokens + runner.observer.total_completion_tokens,
            latency_seconds=round(latency, 3),
        )
        runner.record_item(item_res, trace_detail=resp.retrieval.trace)

    if workers > 1 and pending_items:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(process_item, pending_items))
    else:
        for it in pending_items:
            process_item(it)

    return runner.finalize()
