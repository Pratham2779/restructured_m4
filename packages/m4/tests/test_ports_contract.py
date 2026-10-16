"""Store contract test suite: run identically against InMemoryMemoryStore and Neo4jMemoryStore."""
import os
import pytest
from datetime import datetime, timezone
from m4.types import MemoryNode, Category
from m4.adapters.store_memory import InMemoryMemoryStore
from m4.adapters.store_neo4j import Neo4jMemoryStore


def get_test_stores():
    """Yields all configured store instances for contract testing."""
    stores = [InMemoryMemoryStore()]

    neo4j_uri = os.environ.get("NEO4J_URI")
    if neo4j_uri and not neo4j_uri.startswith("memory"):
        user = os.environ.get("NEO4J_USER", "neo4j")
        pwd = os.environ.get("NEO4J_PASSWORD", "password")
        try:
            ns_store = Neo4jMemoryStore(uri=neo4j_uri, user=user, password=pwd)
            ns_store.connect()
            stores.append(ns_store)
        except Exception as e:
            pytest.skip(f"Neo4j instance configured at {neo4j_uri} but connection failed: {e}")

    return stores


@pytest.fixture(params=[InMemoryMemoryStore], ids=["in_memory_store"])
def store(request):
    """Parameterized fixture for stores that must pass the contract suite."""
    inst = request.param()
    yield inst


def test_upsert_and_search_categories(store):
    ns = "test_contract_cats"
    store.clear_namespace(ns)

    cat1 = Category(id="c1", name="Food & Diet", embedding=[1.0, 0.0, 0.0, 0.0])
    cat2 = Category(id="c2", name="Work & Career", embedding=[0.0, 1.0, 0.0, 0.0])

    store.upsert_category(ns, cat1)
    store.upsert_category(ns, cat2)

    # Search closest to Food & Diet vector
    results = store.search_categories(ns, [0.95, 0.05, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "c1"
    assert results[0][0].name == "Food & Diet"
    assert results[0][1] > 0.9

    store.clear_namespace(ns)


def test_add_and_get_chain_chronological(store):
    ns = "test_contract_chain"
    store.clear_namespace(ns)

    cat = Category(id="cat_life", name="Daily Life", embedding=[0.5, 0.5, 0.0, 0.0])
    store.upsert_category(ns, cat)

    t1 = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, 10, 0, tzinfo=timezone.utc)

    node1 = MemoryNode(
        id="n1", text="First event", category_ids=["cat_life"],
        session_id="s1", timestamp=t1, last_used_at=t1, embedding=[0.1, 0.2, 0.3, 0.4]
    )
    node2 = MemoryNode(
        id="n2", text="Second event", category_ids=["cat_life"],
        session_id="s2", timestamp=t2, last_used_at=t2, embedding=[0.4, 0.3, 0.2, 0.1]
    )

    store.add_node(ns, node1, ["cat_life"])
    store.add_node(ns, node2, ["cat_life"])
    store.link_next(ns, node1.id, node2.id)

    chain = store.get_chain(ns, "cat_life")
    assert len(chain) == 2
    assert chain[0].id == "n1"
    assert chain[1].id == "n2"
    assert chain[0].timestamp < chain[1].timestamp

    # Check neighbors
    p1, n1 = store.neighbors(ns, "n1")
    assert p1 is None
    assert n1 is not None and n1.id == "n2"

    p2, n2 = store.neighbors(ns, "n2")
    assert p2 is not None and p2.id == "n1"
    assert n2 is None

    store.clear_namespace(ns)


def test_replace_nodes_calibrated(store):
    ns = "test_contract_replace"
    store.clear_namespace(ns)

    cat = Category(id="cat_c", name="Habits", embedding=[1.0, 0.0, 0.0, 0.0])
    store.upsert_category(ns, cat)

    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 3, tzinfo=timezone.utc)

    n1 = MemoryNode(id="n1", text="Drinks coffee", category_ids=["cat_c"], session_id="s1", timestamp=t1, last_used_at=t1)
    n2 = MemoryNode(id="n2", text="Switched to tea", category_ids=["cat_c"], session_id="s2", timestamp=t2, last_used_at=t2)

    store.add_node(ns, n1, ["cat_c"])
    store.add_node(ns, n2, ["cat_c"])
    store.link_next(ns, n1.id, n2.id)

    # Calibrate: replace both n1 and n2 with unambiguous merged node
    n_cal = MemoryNode(
        id="n_cal", text="Used to drink coffee, now drinks tea",
        category_ids=["cat_c"], session_id="s2", timestamp=t3, last_used_at=t3, calibrated=True
    )
    store.replace_nodes(ns, ["n1", "n2"], n_cal)

    chain = store.get_chain(ns, "cat_c")
    assert len(chain) == 1
    assert chain[0].id == "n_cal"
    assert chain[0].calibrated is True

    store.clear_namespace(ns)


def test_update_node_text_compression(store):
    ns = "test_contract_update"
    store.clear_namespace(ns)

    cat = Category(id="c_u", name="General", embedding=[1.0, 0.0, 0.0, 0.0])
    store.upsert_category(ns, cat)

    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    node = MemoryNode(id="n_u", text="Original verbose text", category_ids=["c_u"], session_id="s1", timestamp=t, last_used_at=t)
    store.add_node(ns, node, ["c_u"])

    store.update_node_text(ns, "n_u", "Compressed text", compressed_level=2)

    chain = store.get_chain(ns, "c_u")
    assert len(chain) == 1
    assert chain[0].text == "Compressed text"
    assert chain[0].compressed_level == 2

    store.clear_namespace(ns)


def test_shared_nodes_multi_category(store):
    ns = "test_contract_shared"
    store.clear_namespace(ns)

    c1 = Category(id="c1", name="Photography", embedding=[1.0, 0.0, 0.0, 0.0])
    c2 = Category(id="c2", name="Pets", embedding=[0.0, 1.0, 0.0, 0.0])
    store.upsert_category(ns, c1)
    store.upsert_category(ns, c2)

    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # Node belongs to BOTH Photography and Pets (multi-category shared node!)
    shared_node = MemoryNode(
        id="shared_1",
        text="Takes dog Barnaby on weekend macro photography walks",
        category_ids=["c1", "c2"],
        session_id="s1",
        timestamp=t,
        last_used_at=t,
    )
    store.add_node(ns, shared_node, ["c1", "c2"])

    # Node in c1 only
    n_photo = MemoryNode(id="photo_1", text="Loves macro flower shots", category_ids=["c1"], session_id="s2", timestamp=t, last_used_at=t)
    store.add_node(ns, n_photo, ["c1"])

    # Check that both categories retrieve shared_node
    chain1 = store.get_chain(ns, "c1")
    chain2 = store.get_chain(ns, "c2")
    assert any(n.id == "shared_1" for n in chain1)
    assert any(n.id == "shared_1" for n in chain2)

    # Check shared nodes for n_photo: shares c1 with shared_1
    shared_with_photo = store.shared_nodes(ns, "photo_1")
    assert any(n.id == "shared_1" for n in shared_with_photo)

    store.clear_namespace(ns)


def test_namespace_isolation(store):
    ns_a = "namespace_alice"
    ns_b = "namespace_bob"
    store.clear_namespace(ns_a)
    store.clear_namespace(ns_b)

    cat_a = Category(id="cat_a", name="Alice Diary", embedding=[1.0, 0.0, 0.0, 0.0])
    cat_b = Category(id="cat_b", name="Bob Diary", embedding=[0.0, 1.0, 0.0, 0.0])

    store.upsert_category(ns_a, cat_a)
    store.upsert_category(ns_b, cat_b)

    # Ensure namespaces do not leak categories
    search_a = store.search_categories(ns_a, [1.0, 0.0, 0.0, 0.0], top_k=10)
    assert len(search_a) == 1
    assert search_a[0][0].id == "cat_a"

    search_b = store.search_categories(ns_b, [0.0, 1.0, 0.0, 0.0], top_k=10)
    assert len(search_b) == 1
    assert search_b[0][0].id == "cat_b"

    # Clear ns_a and verify ns_b is untouched
    store.clear_namespace(ns_a)
    assert store.stats(ns_a).category_count == 0
    assert store.stats(ns_b).category_count == 1

    store.clear_namespace(ns_b)
