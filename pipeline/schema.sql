CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id        bigserial PRIMARY KEY,
  file      text NOT NULL,
  speaker   text NOT NULL,
  start_ms  int  NOT NULL,
  end_ms    int  NOT NULL,
  text      text NOT NULL,
  n_words   int  NOT NULL,
  embedding vector(384),
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx  ON chunks USING gin  (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks (file);
