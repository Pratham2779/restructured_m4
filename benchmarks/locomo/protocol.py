"""LoCoMo evaluation protocol implementation."""
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
from benchmarks.locomo.loader import load_locomo_dataset, LoCoMoConversation, LoCoMoQA


def run_benchmark(
    stage: str = "smoke",
    workers: int = 6,
    gamma: float = 0.5,
    temperature: float = 0.8,
    model: str = "qwen2.5:7b-instruct",
    embedder: str = "bge-small-en-v1.5",
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
    Run the LoCoMo benchmark evaluation per paper protocol.
    
    Stages:
      - smoke: 1 conversation, first 2 QA items (fast piping verification).
      - pilot: 1 conversation, first 20 QA items.
      - full: all 10 conversations, all 1,540 QA items.
    """
    conversations = load_locomo_dataset(file_path=data_path, exclude_adversarial=True)

    # Filter based on stage
    if stage == "smoke":
        conversations = conversations[:1]
        for c in conversations:
            c.qa_items = c.qa_items[:2]
    elif stage == "pilot":
        conversations = conversations[:1]
        for c in conversations:
            c.qa_items = c.qa_items[:20]
    elif stage == "full":
        # Keep all conversations and all 1,540 QA pairs
        pass

    # Collect sampled item IDs
    all_target_qa: List[LoCoMoQA] = []
    for c in conversations:
        all_target_qa.extend(c.qa_items)
    sampled_ids = [q.qa_id for q in all_target_qa]

    # Initialize Benchmark Runner
    runner = BenchmarkRunner(
        dataset_name="locomo",
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

    # Configure M4 Memory
    config = M4Config(
        gamma=gamma,
        temperature=temperature,
        default_model=model,
        default_embedder=embedder,
    )

    if mock:
        from benchmarks.locomo.tests.test_locomo_protocol import FakeBenchmarkLLM, FakeBenchmarkEmbedder
        llm = llm_client or FakeBenchmarkLLM()
        emb = embedder_client or FakeBenchmarkEmbedder()
        mem_store = store or InMemoryStore()
    else:
        # Initialize ports / adapters if not passed (e.g. Injected fake for testing)
        llm = llm_client or OpenAICompatLLM(
            base_url=os.environ.get("M4_LLM_URL", "http://localhost:11434/v1"),
            model=model,
            temperature=temperature,
        )
        emb = embedder_client or OllamaEmbedder(
            base_url=os.environ.get("M4_EMBED_URL", "http://localhost:11434"),
            model=embedder,
        )
        if store is not None:
            mem_store = store
        elif os.environ.get("NEO4J_URI"):
            from m4.adapters.store_neo4j import Neo4jMemoryStore
            mem_store = Neo4jMemoryStore(
                uri=os.environ["NEO4J_URI"],
                user=os.environ.get("NEO4J_USER", "neo4j"),
                password=os.environ.get("NEO4J_PASSWORD", "password"),
            )
            print(f"[Store] Using Neo4j Graph DB: {os.environ['NEO4J_URI']}", flush=True)
        else:
            mem_store = InMemoryStore()

    mem = M4Memory(
        config=config,
        llm=llm,
        embedder=emb,
        store=mem_store,
        compressor=compressor,
        observer=runner.observer,
    )

    # Execute evaluation per conversation
    for conv in conversations:
        sample_id = conv.sample_id
        qa_pending = [q for q in conv.qa_items if not runner.is_item_completed(q.qa_id)]
        
        # Ingest sessions chronologically into conversation namespace if needed
        if qa_pending:
            mem.reset(sample_id)
            total_sess = len(conv.sessions)
            print(f"\n[Conversation: {sample_id}] Ingesting {total_sess} sessions...", flush=True)
            for s_idx, session in enumerate(conv.sessions, 1):
                print(f"  -> Ingesting session {s_idx}/{total_sess} (ID: {session.id})...", flush=True)
                mem.ingest_session(sample_id, session)

        # Evaluate QA items
        total_qa = len(conv.qa_items)
        pending_qa = [qa for qa in conv.qa_items if not runner.is_item_completed(qa.qa_id)]
        if pending_qa:
            print(f"\n[Conversation: {sample_id}] Evaluating {len(pending_qa)}/{total_qa} QA items (workers={workers})...", flush=True)

            def eval_single_qa(qa: LoCoMoQA) -> None:
                if runner.is_item_completed(qa.qa_id):
                    return

                t0 = time.time()
                resp = mem.respond(sample_id, qa.question)
                latency = time.time() - t0

                # Compute NLP metrics
                f1 = compute_f1(resp.text, qa.answer)
                acc = compute_accuracy(resp.text, qa.answer)
                b1, b4 = compute_bleu(resp.text, qa.answer)
                r_l = compute_rouge_l(resp.text, qa.answer)

                metrics = {
                    "f1": f1,
                    "accuracy": acc,
                    "bleu_1": b1,
                    "bleu_4": b4,
                    "rouge_l": r_l,
                }

                item_res = ItemResult(
                    item_id=qa.qa_id,
                    conversation_id=sample_id,
                    category=qa.category_name,
                    inputs={"question": qa.question, "category": qa.category},
                    prediction=resp.text,
                    gold=qa.answer,
                    metrics=metrics,
                    trace_summary={"node_count": len(resp.retrieval.nodes)},
                    tokens_used=runner.observer.total_prompt_tokens + runner.observer.total_completion_tokens,
                    latency_seconds=round(latency, 3),
                )
                runner.record_item(item_res, trace_detail=resp.retrieval.trace)

            if workers > 1:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    list(executor.map(eval_single_qa, pending_qa))
            else:
                for qa in pending_qa:
                    eval_single_qa(qa)

    return runner.finalize()
