import pytest

from search import rrf_fuse


def result(id_):
    return {"id": id_, "file": "x.wav", "speaker": "A", "start_ms": 0, "end_ms": 1,
            "text": str(id_), "score": 0.0, "kw_rank": None, "vec_rank": None}


def test_rrf_scores_and_ranks():
    kw = [result(1), result(2)]
    vec = [result(2), result(3)]
    fused = rrf_fuse(kw, vec, k=10)

    assert [r["id"] for r in fused] == [2, 1, 3]
    top = fused[0]
    assert top["kw_rank"] == 2 and top["vec_rank"] == 1
    assert top["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1]["kw_rank"] == 1 and fused[1]["vec_rank"] is None
    assert fused[2]["kw_rank"] is None and fused[2]["vec_rank"] == 2


def test_rrf_truncates_to_k_and_handles_empty_keyword_arm():
    vec = [result(i) for i in range(1, 6)]
    fused = rrf_fuse([], vec, k=3)
    assert [r["id"] for r in fused] == [1, 2, 3]
    assert all(r["kw_rank"] is None for r in fused)


# --- read-only checks against the live database (skipped if it is down) ---

@pytest.fixture(scope="module")
def db():
    from search import connect, load_model
    try:
        conn = connect()
    except Exception as e:
        pytest.skip(f"database not reachable: {e}")
    if conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0:
        pytest.skip("chunks table is empty; run ingest.py first")
    yield conn, load_model()
    conn.close()


def test_keyword_search_live(db):
    from search import keyword_search
    conn, _ = db
    results = keyword_search(conn, "moon", 5)
    assert 0 < len(results) <= 5
    assert all("moon" in r["text"].lower() for r in results)
    assert keyword_search(conn, "xyzzyplugh", 5) == []


def test_vector_search_returns_exactly_k_live(db):
    from search import vector_search
    conn, model = db
    for k in (5, 50):
        assert len(vector_search(conn, model, "anything at all", k)) == k
