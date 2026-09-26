"""Stage 6: keyword, vector and hybrid (RRF) search over the chunks table.

Usage:
    python pipeline/search.py "why go back to the moon"
    python pipeline/search.py "SLS Orion" --mode all -k 5

Each result: {id, file, speaker, start_ms, end_ms, text, score, kw_rank, vec_rank}
kw_rank / vec_rank are 1-based ranks in each arm, or None if the chunk was not
in that arm's results.
"""

import argparse

import psycopg
from pgvector.psycopg import register_vector

from ingest import database_url, load_model

RRF_K = 60
POOL = 30             # results taken from each arm before fusing
HNSW_EF_SEARCH = 40   # pgvector default; HNSW returns at most this many rows

COLUMNS = "id, file, speaker, start_ms, end_ms, text"

KEYWORD_SQL = f"""
    SELECT {COLUMNS}, ts_rank_cd(tsv, q) AS score
    FROM chunks, websearch_to_tsquery('english', %s) AS q
    WHERE tsv @@ q
    ORDER BY score DESC, id
    LIMIT %s
"""

VECTOR_SQL = f"""
    SELECT {COLUMNS}, 1 - (embedding <=> %s) AS score
    FROM chunks
    ORDER BY embedding <=> %s, id
    LIMIT %s
"""


def connect():
    # autocommit: search only reads, and an idle open transaction would block
    # ingest.py's TRUNCATE while an eval run holds this connection.
    conn = psycopg.connect(database_url(), autocommit=True)
    register_vector(conn)
    return conn


def rows_to_results(rows, rank_field):
    results = []
    for rank, (id_, file, speaker, start_ms, end_ms, text, score) in enumerate(rows, start=1):
        results.append({
            "id": id_,
            "file": file,
            "speaker": speaker,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "score": float(score),
            "kw_rank": rank if rank_field == "kw_rank" else None,
            "vec_rank": rank if rank_field == "vec_rank" else None,
        })
    return results


def keyword_search(conn, query, k):
    """Full-text search. Returns 0..k results (only chunks that match)."""
    rows = conn.execute(KEYWORD_SQL, (query, k)).fetchall()
    return rows_to_results(rows, "kw_rank")


def vector_search(conn, model, query, k):
    """Nearest chunks by cosine distance. Returns exactly k (if the table has k rows)."""
    # HNSW stops after ef_search candidates, so raise it when k is larger.
    ef_search = max(HNSW_EF_SEARCH, k)
    conn.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(ef_search),))

    query_emb = model.encode(query, normalize_embeddings=True)
    rows = conn.execute(VECTOR_SQL, (query_emb, query_emb, k)).fetchall()
    return rows_to_results(rows, "vec_rank")


def rrf_fuse(kw_results, vec_results, k, rrf_k=RRF_K):
    """Reciprocal Rank Fusion: score = sum over arms of 1 / (rrf_k + rank)."""
    fused = {}
    for arm, rank_field in [(kw_results, "kw_rank"), (vec_results, "vec_rank")]:
        for rank, r in enumerate(arm, start=1):
            if r["id"] not in fused:
                fused[r["id"]] = {**r, "score": 0.0, "kw_rank": None, "vec_rank": None}
            fused[r["id"]][rank_field] = rank
            fused[r["id"]]["score"] += 1 / (rrf_k + rank)

    def sort_key(r):
        best_rank = min(x for x in (r["kw_rank"], r["vec_rank"]) if x is not None)
        return (-r["score"], best_rank, r["id"])

    return sorted(fused.values(), key=sort_key)[:k]


def hybrid_search(conn, model, query, k, pool=POOL):
    kw_results = keyword_search(conn, query, pool)
    vec_results = vector_search(conn, model, query, pool)
    return rrf_fuse(kw_results, vec_results, k)


def search(conn, model, query, k, mode):
    if mode == "keyword":
        return keyword_search(conn, query, k)
    if mode == "vector":
        return vector_search(conn, model, query, k)
    if mode == "hybrid":
        return hybrid_search(conn, model, query, k)
    raise ValueError(f"unknown mode: {mode}")


def fmt_time(ms):
    seconds = ms // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def fmt_rank(rank):
    return "-" if rank is None else str(rank)


def print_results(mode, results):
    print(f"\n== {mode} ({len(results)} results) ==")
    for i, r in enumerate(results, start=1):
        print(
            f"{i:2d}. {r['score']:.4f}  kw={fmt_rank(r['kw_rank']):>2} vec={fmt_rank(r['vec_rank']):>2}  "
            f"{r['file']}  {r['speaker']}  {fmt_time(r['start_ms'])}-{fmt_time(r['end_ms'])}"
        )
        print(f"      {r['text'][:140]}")


def main():
    parser = argparse.ArgumentParser(description="Search the audio chunks.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "keyword", "vector", "all"], default="hybrid")
    args = parser.parse_args()

    modes = ["keyword", "vector", "hybrid"] if args.mode == "all" else [args.mode]
    model = load_model()
    with connect() as conn:
        for mode in modes:
            print_results(mode, search(conn, model, args.query, args.k, mode))


if __name__ == "__main__":
    main()
