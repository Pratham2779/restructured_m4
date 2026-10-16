"""Unit tests for the M4Memory public interface facade."""
from datetime import datetime, timezone
from m4.api import M4Memory
from m4.config import M4Config
from m4.types import Session, Turn
from m4.adapters.store_memory import InMemoryMemoryStore


def test_m4_facade_end_to_end_flow(fake_llm, fake_embedder, fake_compressor, recording_observer):
    config = M4Config(gamma=0.5, temperature=0.8, compress_interval_days=7)
    store = InMemoryMemoryStore()
    mem = M4Memory(
        config=config,
        llm=fake_llm,
        embedder=fake_embedder,
        store=store,
        compressor=fake_compressor,
        observer=recording_observer,
    )

    ns = "user_42"

    # 1. Ingest session with real timestamp
    t1 = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    s1 = Session(
        id="s1",
        turns=[
            Turn(speaker="User", text="I always drink green tea in the morning."),
            Turn(speaker="User", text="I adopted a rescue golden retriever named Max."),
        ],
        timestamp=t1,
    )
    report = mem.ingest_session(ns, s1)
    assert report.session_id == "s1"
    assert len(report.categories_found) >= 2
    assert len(report.nodes_added) >= 2

    # Check stats
    st = mem.stats(ns)
    assert st.node_count >= 2

    # 2. Retrieve memories
    retrieval = mem.retrieve(ns, "What pets do I have?")
    assert len(retrieval.nodes) > 0
    assert any("Max" in n.text for n in retrieval.nodes)

    # 3. Respond
    resp = mem.respond(ns, "What pets do I have?")
    assert len(resp.text) > 0
    assert resp.retrieval is not None

    # 4. Maintenance / Compress after 10 days
    t2 = t1 + datetime.resolution * 86400 * 10
    t2_dt = datetime(2025, 3, 15, tzinfo=timezone.utc)
    comp_count = mem.maintain(ns, now=t2_dt)
    assert comp_count >= 0

    # 5. Reset namespace
    mem.reset(ns)
    st_after = mem.stats(ns)
    assert st_after.node_count == 0


def test_m4_simulated_step_timeline_fallback(fake_llm, fake_embedder):
    """When dataset does not provide real timestamps, fallback to simulated week progression."""
    config = M4Config(fallback_simulated_step_days=7)
    store = InMemoryMemoryStore()
    mem = M4Memory(
        config=config,
        llm=fake_llm,
        embedder=fake_embedder,
        store=store,
    )

    ns = "episode_sim_1"

    # Ingest session without timestamp
    s0 = Session(id="sess_0", turns=[Turn(speaker="Speaker 1", text="I like hiking.")], timestamp=None)
    mem.ingest_session(ns, s0)

    # Ingest second session without timestamp
    s1 = Session(id="sess_1", turns=[Turn(speaker="Speaker 1", text="I enjoy camping.")], timestamp=None)
    mem.ingest_session(ns, s1)

    # Inspect the node timestamps in store
    categories = store.search_categories(ns, [0.0] * 4, top_k=10)
    assert len(categories) > 0
    chain = store.get_chain(ns, categories[0][0].id)
    assert len(chain) >= 2
    delta_days = (chain[1].timestamp - chain[0].timestamp).days
    assert delta_days == 7
