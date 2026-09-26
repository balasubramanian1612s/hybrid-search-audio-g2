## SYSTEM

Hybrid seach over 2 speaker interview/podcast audio. Input will be a query, and output will be top k related moments with the file, speaker, timestamp, text. It is a search engine, not a Q&A. No LLM in search path.

## PIPELINE

6 stages, each takes files as input, and outputs a files.

transcribe -> diarize -> align -> chunk -> ingest -> search -> eval

## transcribe.py

input: audio/*.wav
output: data/transcripts/<stem>.json
        {file, duration, segments:[{id, text, start, end, words:[{word, start, end}]}]}
- faster-whisper, small, cpu, int8
- beam_size = 5
- language = en
- word_timestamps = True
- vad_filter = True
- min_silence_duration = 500ms
- skip if already output exist for the file

## diarize.py

input: audio/*.wav
output: data/diarization/<stem>.json
        {file, turns:[{speaker, start, end}]}
- pyannote community-1, 4.x output shape (output.speaker_diarization)
- num_speaker = 2
- progress the hook
- print per-speaker time limit

## align.py

input: transcripts/<stem>.json + diarization/<stem>.json
output: data/turns/<stem>.json
        {file, turns:[{speaker, start, end, text, n_words, n_low}]}
- drop diarization spans < 250ms
- split the words at the boundary, multi-token can be split, single token should not be split
- assign speaker: never splits.
  Assign to the span it OVERLAPS MOST (not midpoint — midpoint
  discards the word's duration)
- no overlap -> nearest span, confidence='low', dont drop
- group same speaker words into turns, merge gaps <= 2s
- n_words accumulates len(piece.split()), not += 1
- INVARIANT: total TOKEN count out == total TOKEN count in (assert)
- print first 15 turns with low confidence

## chunk.py
input: data/turns/<stem>.json
output: data/chunks/<stem>.json
        {file, speaker, start_ms, end_ms, text, n_words, embed_text}

- MODEL_LIMIT=180 (MiniLM truncates at 256 words)
- TARGET_MAX=80, MIN_WORDS=20, CONTEXT_WORDS=25
- chunk on turns, not fixed windows
- drop backchannels (yeah/right/mm)
- split > TARGET_MAX at sentence boundaries
- merge < MIN_WORDS into adjacent same speaker chunk
- embed_text = prev speaker's last 25 words + this text
- text stays clean for display

## schema.sql + ingest.py

CREATE TABLE chunks (
  id        PRIMARY KEY,
  file      text NOT NULL,
  speaker   text NOT NULL,
  start_ms  int  NOT NULL,
  end_ms    int  NOT NULL,
  text      text NOT NULL,
  n_words   int  NOT NULL,
  embedding vector(384),
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX chunks_tsv_idx  ON chunks USING gin  (tsv);
CREATE INDEX chunks_emb_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_file_idx ON chunks (file);

TRUNCATE chunks RESTART IDENTITY; (re-ingest idempotency)

## 6. search.py
keyword_search: websearch_to_tsquery + ts_rank_cd. Returns 0 to k.
vector_search:  embedding <=> query. Returns exactly k.
hybrid_search:  RRF, score = sum 1/(60+rank)
- pool 30 per arm before fusing
- show kw_rank / vec_rank into output

## 7. eval.py
golden.json labels are TIME SPANS, not chunk ids (I rechunk while tuning)
relevant if: chunk.start_ms < span.end*1000 AND chunk.end_ms > span.start*1000
- recall@5 and MRR, per query type, all three modes
- load model + connection ONCE

## Success criteria (before results)
lexical R@5 >= 0.80 | semantic >= 0.70 | hybrid beats both on mixed
low-confidence < 5%

## Out of scope
LLM answers, query expansion, cross-file speaker ID

