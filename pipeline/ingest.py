"""Stage 5: load data/chunks/*.json into Postgres with embeddings.

- applies schema.sql (idempotent: IF NOT EXISTS everywhere)
- TRUNCATE chunks RESTART IDENTITY, then inserts every chunk, all in one
  transaction, so re-running always gives the same table and a failed run
  leaves the old data in place
- embeds embed_text (text + other speaker's context); stores clean text
"""

import json
import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path("data/chunks")
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 32

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5434/postgres"

INSERT_SQL = """
    INSERT INTO chunks (file, speaker, start_ms, end_ms, text, n_words, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def database_url():
    load_dotenv()
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def load_model():
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    dim = model.get_embedding_dimension()
    assert dim == EMBED_DIM, f"{EMBED_MODEL} gives {dim} dims, schema expects {EMBED_DIM}"
    return model


def embed(model, texts):
    # Normalized vectors: cosine distance and dot product then agree.
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def load_chunks():
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        with open(path) as f:
            chunks.extend(json.load(f))
    return chunks


def ingest(conn, chunks, embeddings):
    with conn.transaction():
        conn.execute(SCHEMA_PATH.read_text())
        conn.execute("TRUNCATE chunks RESTART IDENTITY")
        rows = []
        for c, emb in zip(chunks, embeddings):
            rows.append((c["file"], c["speaker"], c["start_ms"], c["end_ms"], c["text"], c["n_words"], emb))
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)


def print_report(conn):
    rows = conn.execute(
        "SELECT file, count(*), sum(n_words) FROM chunks GROUP BY file ORDER BY file"
    ).fetchall()
    for file, n_chunks, n_words in rows:
        print(f"  {file}: {n_chunks} chunks, {n_words} words")

    total, no_emb, empty_tsv = conn.execute(
        """
        SELECT count(*),
               count(*) FILTER (WHERE embedding IS NULL),
               count(*) FILTER (WHERE tsv = ''::tsvector)
        FROM chunks
        """
    ).fetchone()
    print(f"  total {total} rows | {no_emb} missing embedding | {empty_tsv} empty tsvector")


def main():
    chunks = load_chunks()
    if not chunks:
        print(f"No chunks in {CHUNKS_DIR}/")
        return
    print(f"loaded {len(chunks)} chunks from {CHUNKS_DIR}/")

    # Embed before opening the transaction so the table is locked only briefly.
    t0 = time.time()
    model = load_model()
    embeddings = embed(model, [c["embed_text"] for c in chunks])
    print(f"embedded {len(chunks)} chunks in {time.time() - t0:.0f}s")

    with psycopg.connect(database_url()) as conn:
        # The vector type must exist before register_vector can look it up.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)

        ingest(conn, chunks, embeddings)
        print("ingested:")
        print_report(conn)


if __name__ == "__main__":
    main()
