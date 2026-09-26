# Effective Retrieval from Audio Transcripts: Submission

**G2 AI Hiring Hackathon, Problem Statement 1 (Multimodal AI: Audio search)**

| | |
|---|---|
| How to run it | [README.md](README.md), or just `./init.sh` |
| Golden dataset | [golden.json](golden.json): 63 queries, 117 labelled time spans, 6 recordings |
| Latest eval report | [results/result_20260926_145727.md](results/result_20260926_145727.md) |
| Spec given to the coding agent | [design.md](design.md) |
| Trace of agent collaboration | [prompts.md](prompts.md) |

---

## 1. Summary

- **What it does:** a fully local hybrid search engine over two-speaker podcast audio. A query returns the matching **moments**, each with its **file, speaker, start–end timestamp and text**.
- **How it works:**
  - Keyword search (Postgres full-text) and semantic search (MiniLM embeddings in pgvector) run in parallel.
  - The two rankings are fused with Reciprocal Rank Fusion.
  - There is no LLM anywhere in the search path.
- **Golden dataset:** 6 conversations, 52 minutes in total, each with a different pair of speakers. There are 63 hand-labelled queries of three types (lexical, semantic, mixed), including traps where the same words appear in the wrong file, and 10 documented failure cases.
- **Headline results:**
  - Hybrid is **never worse than the better single search on any query type**, and it clearly beats both on lexical queries.
  - Excluding the 10 deliberate failure cases, hybrid reaches **R@5 = 0.86 and MRR = 0.86** across the 53 remaining queries.
  - With the failure cases included (the official run), the lexical target (0.80) and semantic target (0.70) are missed: 0.75 and 0.66.
- **Main weakness found by the eval:** keyword search requires every word to match, so it returns nothing for 37 of the 63 queries. The fix is known and is described in section 7.

---

## 2. Golden dataset

### Recordings

| file | length | speakers / content |
|---|---|---|
| `behavioral_mock_interview` | 7:12 | interviewer and candidate, a mock consulting behavioural interview |
| `nasa_interview` | 10:00 | TV host and NASA administrator Jared Isaacman |
| `psychology_raj_shamani` | 7:15 | Raj Shamani and Evie, a former US Secret Service agent, on influence strategies |
| `beerbiceps` | 7:33 | BeerBiceps host and a spiritual teacher on the Bhagavad Gita |
| `nikhilkamath` | 9:37 | Nikhil Kamath and Chamath Palihapitiya on investing and AI |
| `y_combinator` | 10:00 | two YC partners on product-market fit |

All audio is converted to 16 kHz mono wav; clips longer than 10 minutes are cut at 10:00. The six conversations were chosen to be deliberately different in domain: interviews, space policy, psychology, spirituality, finance and startups. This makes cross-file mistakes visible.

### Queries

| type | n | meaning | example |
|---|---|---|---|
| lexical | 23 | the exact words are spoken | `rat poison`, `Artemis program` |
| semantic | 22 | a paraphrase; the key words are never spoken | `will AI take away people's jobs` |
| mixed | 18 | a natural question that contains some spoken words | `risk of ruin with large investments` |

### Design choices, and why

- **Labels are time spans, not chunk ids.** A label is `{file, start, end}` in seconds, so the labels survive any change to chunking or chunk sizes.
- **Every relevant moment is labelled,** not just the first mention. 34 queries have between 2 and 8 spans. This makes recall graded rather than just "found / not found".
- **One labelling rule for everything:** a span is any moment a listener searching this query would want to land on. A question and its immediate answer count as one moment.
- **Each span has a `why` field** quoting what is said there, so any label can be checked in seconds.
- **Traps:** some queries share a word across files but are correct in only one. Examples: "leadership experience" (the behavioral interview) vs "American leadership in space" (nasa), and NASA's commercial partners vs the consulting firm's partners.
- **Known failure cases:** 10 queries are marked `expect_fail` with a `fail_reason`. Each one was run against live search before being added, so they are real failures, not guesses. They document what the system cannot do, and act as a regression list: the report shows whether each one "still fails", is "partial", or "now passes".

### Scoring

- A result **hits** a span if it is in the same file and its time range overlaps the span.
- **Recall@5** is counted per span: the share of a query's spans that are hit by at least one of the top 5 results, averaged over queries. Counting chunks instead would reward returning the same moment 5 times.
- **MRR** = 1 / rank of the first hit, looking at the top 10 results.

---

## 3. Engineering design and rationale

```
audio/*.wav -> transcribe -> diarize -> align -> chunk -> ingest -> search -> eval
```

Each stage reads files and writes files (`data/<stage>/<file>.json`), so each stage can be inspected, tested and re-run on its own.

| stage | decision | why |
|---|---|---|
| **Transcribe** | faster-whisper `small`, int8, CPU, beam 5, VAD, **word-level timestamps** | Runs fully locally. Word timestamps are what make exact speaker attribution and exact chunk boundaries possible. |
| **Diarize** | pyannote `speaker-diarization-community-1`, `num_speakers=2`, the **exclusive** (non-overlapping) output | Every conversation has exactly two speakers. The exclusive output has no overlapping turns, so each word has one clear owner. Only 0.1% of words fall between speaker turns and need the low-confidence fallback below. |
| **Align** | Each word goes to the speaker turn it **overlaps most**, not the one containing its midpoint. Words are never split. A word that falls in no turn goes to the nearest one and is marked low-confidence. Turns shorter than 250 ms are dropped. Same-speaker words with a gap under 2 s are merged into one turn. The code asserts that the word count going in equals the word count coming out. | Going by overlap uses the word's full duration. The fallback never drops a word, and the assert guarantees no text is lost between the transcript and the index. |
| **Chunk** | Chunks follow **speaker turns**, not fixed windows. Long turns are split at sentence boundaries into chunks of up to about 80 words. Backchannel-only turns ("yeah", "mm") are dropped. | Each result is one speaker's moment, so the speaker shown is always correct for the whole chunk. Chunks of about 80 words keep timestamps precise and stay well under MiniLM's 256-token limit. |
| | Each chunk is embedded as `embed_text` = **the other speaker's last 25 words + this chunk**. The stored `text` stays clean. | In interviews, answers often don't repeat the question. Adding the question to the embedding makes "why is the moon important" find the *answer*, while the text shown to users and used for keyword search stays exactly what was said. |
| **Store** | One Postgres table holds text, timestamps, speaker, a `vector(384)` with an **HNSW** cosine index, and a generated `tsvector` with a **GIN** index | One database for both kinds of search: no second store to keep in sync, and everything is queryable with SQL. The ingest runs in a single transaction, so re-running it is safe. |
| **Embed** | `all-MiniLM-L6-v2` (384 dimensions, normalized), run locally on CPU | Small and fast: all 192 chunks embed in about 6 s. The tradeoff is weaker semantics (see section 7). |
| **Search** | Keyword: `websearch_to_tsquery` + `ts_rank_cd`. Vector: cosine distance `<=>`. Hybrid: **RRF** with `score = Σ 1/(60 + rank)` over the top 30 from each | Keyword and cosine scores aren't on comparable scales, and RRF only uses ranks, so no score normalization is needed. Each result shows its `kw_rank` and `vec_rank`, so you can see which search found it. |
| **Eval** | `eval.py` runs all 3 modes over the golden set and writes a markdown report with success criteria, a summary by type, per-query results and known failures | Every change is measured the same way, and the reports stay on disk for comparison. |

**Example: fusion finding what each search misses.** For "Artemis program", keyword search found 2 of the 3 relevant moments and vector search found a different 2 of 3. Hybrid found **all 3**. It ranked the 01:20 moment 2nd (keyword rank 1, vector rank 12) and the 09:47 moment 3rd (vector only).

**Tests:**
- `pytest` runs 25 tests: unit tests for each stage and for the scoring functions, plus live-database checks that skip if Postgres is down.
- `eval.py` is the automated recall@k measurement.
- `init.sh` sets up and runs everything with one command.

---

## 4. Success criteria

| # | criterion | why |
|---|---|---|
| 1 | Lexical R@5 (hybrid) ≥ **0.80** | Users must be able to find specific words that were said. |
| 2 | Semantic R@5 (hybrid) ≥ **0.70** | Users must be able to find things phrased differently from how they were said. |
| 3 | Hybrid R@5 ≥ the better of keyword and vector, **for every query type** | Fusion must never lose a moment that either search found on its own. |

**How the criteria changed:** criterion 3 was originally "hybrid beats both searches on mixed queries". The eval showed that this can't pass as written. When keyword search returns nothing, the RRF ranking becomes exactly the vector ranking, so hybrid ties vector by construction. That happened on 10 of the 18 mixed queries. On the other 8, keyword only found moments vector had already found. The criterion was really measuring keyword search, not fusion, so it was replaced with the fairer "never worse than the best single search".

---

## 5. Results

**Official run:** all 63 queries, including the 10 known failures.

| type | n | keyword R@5 | vector R@5 | **hybrid R@5** | hybrid MRR |
|---|---|---|---|---|---|
| lexical | 23 | 0.62 | 0.60 | **0.75** | 0.77 |
| semantic | 22 | 0.00 | 0.66 | **0.66** | 0.61 |
| mixed | 18 | 0.37 | 0.78 | **0.78** | 0.87 |
| all | 63 | 0.33 | 0.67 | **0.73** | 0.74 |

**Excluding the 10 known failures** (53 queries):

| type | n | keyword R@5 | vector R@5 | **hybrid R@5** | hybrid MRR |
|---|---|---|---|---|---|
| lexical | 18 | 0.80 | 0.77 | **0.95** | 0.97 |
| semantic | 17 | 0.00 | 0.83 | **0.83** | 0.73 |
| mixed | 18 | 0.37 | 0.78 | **0.78** | 0.87 |
| all | 53 | 0.40 | 0.79 | **0.86** | 0.86 |

### Achievement against the criteria

| criterion | target | official (63) | excluding known failures (53) |
|---|---|---|---|
| 1. lexical R@5 | ≥ 0.80 | 0.75 **FAIL** | 0.95 PASS |
| 2. semantic R@5 | ≥ 0.70 | 0.66 **FAIL** | 0.83 PASS |
| 3. hybrid ≥ best single search, lexical | ≥ 0.62 | 0.75 PASS | 0.95 ≥ 0.80 PASS |
| 3. hybrid ≥ best single search, semantic | ≥ 0.66 | 0.66 PASS (tie) | 0.83 PASS (tie) |
| 3. hybrid ≥ best single search, mixed | ≥ 0.78 | 0.78 PASS (tie) | 0.78 PASS (tie) |

**How to read this:**
- **Criteria 1 and 2 are missed in the official run, and the misses are entirely due to the deliberate failure cases:** 5 lexical and 4 semantic queries that were added *because* they fail, plus 1 partial. On normal queries both targets pass with margin.
- **Hybrid adds the most on lexical queries:** +0.15 over the best single search (0.95 vs 0.80). Keyword and vector search each find moments the other misses.
- **On semantic queries hybrid equals vector,** because keyword search can't match paraphrases. That's expected.
- **On mixed queries hybrid ties vector on recall but ranks better** (MRR 0.87 vs 0.84). It can't gain recall there while keyword search returns nothing for most mixed queries (section 7, limitation 1).
- **Latency:** p50 query time was 0.3 ms for keyword, 4.1 ms for vector (including embedding the query) and 4.0 ms for hybrid, on an Apple M4 laptop CPU with 192 chunks.

---

## 6. Tuning after the baseline

Two changes, made one at a time and each motivated by a v1 finding. Both were measured on the same 63 queries with the same criteria; there was no held-out set.

| run | change | lexical R@5 | semantic R@5 | mixed R@5 | all R@5 | all MRR | report |
|---|---|---|---|---|---|---|---|
| v1 | baseline (section 5) | 0.75 | 0.66 | 0.78 | 0.73 | 0.74 | [v1](results/result_20260926_145727.md) |
| v2 | join split tokens (`4 .5 %` becomes `4.5%`) | **0.79** | 0.66 | 0.78 | 0.74 | 0.76 | [v2](results/result_20260926_161009.md) |
| v3 | chunk size 80 to 100 words | 0.79 | **0.75** | **0.85** | **0.79** | **0.80** | [v3](results/result_20260926_161113.md) |

- **Split tokens:** only "4.5%" changed, from not found to rank 1. No other query got worse.
- **100-word chunks, the trade-off:** bigger chunks help keyword matching and cover more of each moment, but they dilute the embedding. 3 queries got worse, and 2 known failures now pass only because a chunk grew to include the answer; negation and the ASR error are not fixed.

---

## 7. Limitations

**Retrieval**
1. **Keyword search requires every word to match.** `websearch_to_tsquery` ANDs every term, so one unmatched word ("describe", "plan") empties the result. It returned nothing for 37 of the 63 queries, which caps what hybrid can add on mixed queries. The next step is to OR the terms and let `ts_rank_cd` reward chunks that match more of them, then re-run the eval.
2. **Split tokens.** Whisper outputs `4.5%` as three words: `4`, `.5`, `%`. The same happens to `helium-3`, `C-suite`, `U.S.` and `single-minded`. The pipeline joins words with spaces, so the stored text reads `4 .5 %` and can't match. The fix: attach pieces that start with `.`, `-` or `%` to the previous word in `align.py`.
3. **Limits of a small embedding model:**
   - negation: "internships that were *not* technical" returns the technical ones
   - vocabulary gaps: "less money" doesn't connect to "affordably"
   - aliases: "Y Combinator" doesn't connect to "YC"
   - typos in names: "artemus" instead of "artemis"
4. **Search returns single moments.** It can't combine facts spread across a conversation, such as "what jobs has Evie had", whose answer is spread across the whole talk.

**Transcription and speakers**

5. **Speech-recognition errors are unrecoverable downstream:** "Metta" (Meta), "copper and content" (copyrighted content), "PV quote" (PG quote), "parallel linguistics" (paralinguistics), a garbled Sanskrit verse, and "using correctly" where the speaker probably said "incorrectly". The `small` Whisper model mishears names and rare words.
6. **Speakers are anonymous:**
   - Labels are `SPEAKER_00` / `SPEAKER_01` per file, with no names or roles, so you can't search "what did the interviewer ask" (that query is a known partial failure).
   - Diarization makes occasional errors. For example, a host question at 7:45 in nikhilkamath is attributed to the guest.
   - `num_speakers=2` is hard-coded, so show intros and outros are forced into one of the two speakers.

**Evaluation and dataset**

7. **The golden set is small and labelled by one annotator.** Labels were drafted by the coding agent from the transcripts, following my labelling rules. There was no second annotator and no agreement measure. With about 20 queries per type, one query moves a type's average by about 0.05.
8. **Labels inherit the ASR's timing**, because spans were located using the same transcripts that search runs over.
9. **The hit rule is lenient:** any overlap counts, so long spans are easy to hit.
10. **Three recordings are under the requested 8–10 minutes:** 7:12, 7:15 and 7:33. The source recordings are that long. The other three are 9:37 to 10:00.

**Scale**

11. **Scale is untested.** The system was only run on 192 chunks; HNSW has no measurable effect at this size, and latency numbers won't carry over. Ingest also truncates and reloads the whole table rather than updating it incrementally.

---

## 8. What would change at scale

The problem statement asks which diarization, database, embedding and indexing strategies would give "ideal" retrieval quality at scale. These are recommendations, not what was built:

| area | now | at scale |
|---|---|---|
| **Transcription** | Whisper `small` on CPU | `large-v3` / `turbo` on GPU, or a hosted API; prime the model with a vocabulary of names and domain terms (e.g. Whisper's `initial_prompt`) to cut name errors; store per-word ASR confidence |
| **Diarization** | community-1, exclusive output, 2 speakers | keep the exclusive output; detect the speaker count instead of fixing it at 2; store speaker embeddings so the same person can be recognised across files and given a name (turning `SPEAKER_00` into "Chamath") |
| **Keyword search** | `ts_rank_cd`, AND semantics | BM25 ranking (e.g. ParadeDB `pg_search`) with OR semantics; `pg_trgm` fuzzy matching for typos; a synonym and alias dictionary ("YC" = "Y Combinator") |
| **Embeddings** | MiniLM, 384 dimensions | a stronger retrieval model (e.g. bge / e5 / gte, around 768 dimensions); `halfvec` to halve storage; a **cross-encoder reranker** on the top ~50 fused results to fix negation and precision |
| **Indexing** | HNSW with defaults, one table | tune HNSW `m` / `ef_construction` / `ef_search` against recall; partition by tenant or collection; pgvector 0.8 iterative index scans for filtered searches (by file, speaker, date); incremental upserts keyed on `(file, start_ms)` instead of a full reload |
| **Fusion** | plain RRF with k = 60 | weighted RRF tuned per query type, or a lightweight classifier that shifts weight toward keyword search for short, name-like queries |

---

## 9. Metrics for production and what makes an effective evaluation

**Retrieval quality (offline, on every change):**
- Recall@k and MRR, broken down by query type and by file, as done here.
- nDCG@k with graded labels: "the answer" vs "mentioned in passing".
- **Zero-result rate for each search.** This was the most useful single number in this project: it exposed the AND problem immediately.
- Pass/fail status of every known failure case, used as a regression list.

**Upstream quality**, because retrieval can't recover what transcription loses:
- Word error rate on a sampled, hand-corrected set of transcripts, with a separate error rate for **named entities**.
- Diarization error rate, plus the speaker-attribution accuracy of the results actually returned.
- The share of words assigned to a speaker with low confidence.

**System:**
- p50 / p95 latency for each search and end to end.
- Ingest throughput (audio hours processed per hour), time from upload to searchable, cost per audio hour, index size.

**What makes the evaluation effective:**
- labels as time spans, so they survive re-chunking
- every relevant moment labelled, so recall is graded
- separate query types
- trap queries in other files
- a maintained list of known failures
- a report written on every run so changes can be compared

In production, add multiple annotators with an agreement measure, and grow the query set from real (anonymized) user queries.

---

## 10. Coding agent disclosure

**How I directed it:**
1. **Spec first.** Before any code, I gave the agent [design.md](design.md). It sets out, for each stage: input and output schemas, parameters, rules and invariants (e.g. "assign to the span it overlaps most, not the midpoint", "token count out == token count in"), plus success criteria and what's out of scope.
2. **Working rules in the first prompt:**
   - build **one module at a time**, in pipeline order, with no scaffolding of the whole project
   - the stack is fixed (Python 3.11, faster-whisper, pyannote 4.x, sentence-transformers, psycopg3, pytest, dotenv)
   - **if you think a design decision is wrong, say so and why; do not silently implement something different**
   - write boring, readable code

**Full trace:** [prompts.md](prompts.md) records every prompt, every question the agent asked, and what I decided and why.
