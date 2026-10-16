"""Unit tests for M4 pipeline stages using hermetic fakes."""
from datetime import datetime, timezone, timedelta
from m4.config import M4Config
from m4.types import Turn, MemoryNode, Category
from m4.adapters.store_memory import InMemoryMemoryStore
from m4.pipeline.learning import CategoryLearningPipeline
from m4.pipeline.construction import ChainConstructionPipeline
from m4.pipeline.update import MemoryUpdatingPipeline
from m4.pipeline.retrieval import SelfReflectionRetrievalPipeline


def test_category_learning_extraction_and_merge(fake_llm, fake_embedder, recording_observer):
    config = M4Config(gamma=0.5)
    pipeline = CategoryLearningPipeline(config, fake_llm, fake_embedder, recording_observer)

    turns = [
        Turn(speaker="User", text="I always drink green tea in the morning."),
        Turn(speaker="Assistant", text="That is very healthy."),
        Turn(speaker="User", text="Also, I adopted a golden retriever named Max."),
    ]

    clues = pipeline.process_session("s1", turns)
    assert len(clues) == 2
    cat_names = [c[0] for c in clues]
    assert "Personal Preferences" in cat_names
    assert "Pets" in cat_names

    # Verify telemetry observer recorded call
    assert any(c["purpose"] == "category_learning_extraction" for c in recording_observer.calls)


def test_chain_construction_and_time_injection(fake_embedder):
    config = M4Config(gamma=0.5)
    store = InMemoryMemoryStore()
    pipeline = ChainConstructionPipeline(config, store)

    ns = "test_construction_ns"
    injected_time = datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)

    clues = [
        ("Personal Preferences", "User loves green tea", [0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]),
    ]

    nodes = pipeline.construct_chains(ns, clues, session_id="s1", injected_time=injected_time)
    assert len(nodes) == 1
    node = nodes[0]

    # Verify timestamp was INJECTED, not read from wall clock
    assert node.timestamp == injected_time
    assert node.last_used_at == injected_time

    # Verify category was created
    cats = store.search_categories(ns, [0.9, 0.1, 0.0, 0.0], top_k=1)
    assert len(cats) == 1
    assert cats[0][0].name == "Personal Preferences"


def test_calibrate_conflict_resolution(fake_llm, fake_embedder, recording_observer):
    config = M4Config(gamma=0.5)
    store = InMemoryMemoryStore()
    updating = MemoryUpdatingPipeline(
        config, fake_llm, store, fake_embedder, observer=recording_observer
    )

    ns = "test_calibrate_ns"
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)

    cat = Category(id="cat_diet", name="Diet", embedding=[1.0, 0.0, 0.0, 0.0])
    store.upsert_category(ns, cat)

    # Initial node: drinks coffee
    n1 = MemoryNode(
        id="n1", text="User drinks black coffee every morning.", category_ids=["cat_diet"],
        session_id="s1", timestamp=t1, last_used_at=t1, embedding=[0.9, 0.1, 0.0, 0.0]
    )
    store.add_node(ns, n1, ["cat_diet"])

    # New conflicting node: switched to tea instead of coffee
    n2 = MemoryNode(
        id="n2", text="User drinks warm green tea instead of coffee now.", category_ids=["cat_diet"],
        session_id="s2", timestamp=t2, last_used_at=t2, embedding=[0.9, 0.1, 0.0, 0.0]
    )
    store.add_node(ns, n2, ["cat_diet"])
    store.link_next(ns, n1.id, n2.id)

    # Run calibration
    cal_count = updating.calibrate_new_node(ns, n2)
    assert cal_count == 1

    # Verify chain now contains the calibrated merged node
    chain = store.get_chain(ns, "cat_diet")
    assert len(chain) == 1
    assert chain[0].calibrated is True
    assert "coffee, but now strictly drinks tea" in chain[0].text


def test_compress_dormant_nodes_iterative(fake_llm, fake_embedder, fake_compressor, recording_observer):
    config = M4Config(compress_interval_days=7, max_compression_rounds=3)
    store = InMemoryMemoryStore()
    updating = MemoryUpdatingPipeline(
        config, fake_llm, store, fake_embedder, compressor=fake_compressor, observer=recording_observer
    )

    ns = "test_compress_ns"
    past_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    current_time = datetime(2025, 1, 20, tzinfo=timezone.utc)  # 19 days later (> 7 days)

    cat = Category(id="c_dormant", name="Hobbies", embedding=[1.0, 0.0, 0.0, 0.0])
    store.upsert_category(ns, cat)

    node = MemoryNode(
        id="n_dormant",
        text="User participated in an extensive five kilometer charity road race last spring.",
        category_ids=["c_dormant"],
        session_id="s_past",
        timestamp=past_time,
        last_used_at=past_time,
        embedding=[0.5, 0.5, 0.0, 0.0],
    )
    store.add_node(ns, node, ["c_dormant"])

    orig_words = len(node.text.split())
    count = updating.compress_dormant_nodes(ns, now=current_time)
    assert count == 1

    chain = store.get_chain(ns, "c_dormant")
    assert len(chain) == 1
    assert chain[0].compressed_level > 0
    assert len(chain[0].text.split()) < orig_words


def test_self_reflection_retrieval_algorithm_1(fake_llm, fake_embedder, recording_observer):
    config = M4Config(gamma=0.5)
    store = InMemoryMemoryStore()
    retrieval_pipeline = SelfReflectionRetrievalPipeline(
        config, fake_llm, store, fake_embedder, observer=recording_observer
    )

    ns = "test_retrieval_ns"
    cat = Category(id="cat_pref", name="Personal Preferences", embedding=[0.9, 0.1, 0.0, 0.0])
    store.upsert_category(ns, cat)

    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)

    n1 = MemoryNode(id="m1", text="User loves green tea in the morning.", category_ids=["cat_pref"], session_id="s1", timestamp=t1, last_used_at=t1, embedding=[0.9, 0.1, 0.0, 0.0])
    n2 = MemoryNode(id="m2", text="User avoids caffeinated drinks after 2 PM.", category_ids=["cat_pref"], session_id="s2", timestamp=t2, last_used_at=t2, embedding=[0.85, 0.15, 0.0, 0.0])
    store.add_node(ns, n1, ["cat_pref"])
    store.add_node(ns, n2, ["cat_pref"])
    store.link_next(ns, n1.id, n2.id)

    # Query with need for memory
    result = retrieval_pipeline.retrieve(ns, "What does the user like to drink?")
    assert len(result.nodes) > 0
    assert any("green tea" in n.text for n in result.nodes)

    # Response generation
    response_text = retrieval_pipeline.generate_response("What does the user like to drink?", result)
    assert len(response_text) > 0
    assert "green tea" in response_text
