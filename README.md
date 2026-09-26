# Hybrid Search over Podcast Audio

Search two-speaker interviews and podcasts by what was said. Type a query and get back the exact **moments**: file, speaker, timestamp and text.

Every query runs through keyword search (Postgres full-text) and semantic search (MiniLM embeddings), and the two rankings are fused with Reciprocal Rank Fusion. It is a search engine, not a Q&A bot: there is **no LLM in the search path**.

```
$ python pipeline/search.py "Artemis program" -k 3

== hybrid (3 results) ==
 1. 0.0323  kw= 2 vec= 2  nasa_interview.wav  SPEAKER_00  04:44-05:05
      Well, I'd say first and foremost, you know, the SLS architecture that supports Artemis ...
 2. 0.0303  kw= 1 vec=12  nasa_interview.wav  SPEAKER_00  01:20-01:46
      That's a really good question. First, I'd say it's fulfilling a promise that presidents ...
 3. 0.0164  kw= - vec= 1  nasa_interview.wav  SPEAKER_00  09:47-09:59
      They were able to do a number of tests, Comchecks, Eclips, Life Support Checks ...
```

`kw` and `vec` show where each result ranked in the keyword and vector searches, with `-` meaning that search didn't return it. Result 2 was ranked 12th by vector search and result 3 was never found by keyword search. Hybrid ranks both in the top 3.

---

## How it works

```
audio/*.wav -> transcribe -> diarize -> align -> chunk -> ingest -> search -> eval
```

Each stage reads files and writes files, so any stage can be re-run on its own.

| stage | what it does | output |
|---|---|---|
| `transcribe.py` | faster-whisper (small, CPU, int8) with word-level timestamps | `data/transcripts/` |
| `diarize.py` | pyannote community-1, 2 speakers, uses the version with no overlapping turns | `data/diarization/` |
| `align.py` | gives each word to the speaker span it overlaps most, then groups words into speaker turns | `data/turns/` |
| `chunk.py` | splits turns at sentence boundaries into chunks of up to about 80 words, and adds the other speaker's last 25 words (usually the question) as extra embedding context | `data/chunks/` |
| `ingest.py` | embeds the chunks with all-MiniLM-L6-v2 and loads them into Postgres (pgvector + tsvector) | `chunks` table |
| `search.py` | keyword (`websearch_to_tsquery`), vector (cosine) and hybrid (RRF, k=60, top 30 from each) | results |
| `eval.py` | scores all three modes against `golden.json` | `results/result_<timestamp>.md` |

The full spec is in [design.md](design.md). Every decision made while building it, and why, is logged in [prompts.md](prompts.md).

---

## Golden dataset

[golden.json](golden.json) has **63 hand-labelled queries** over 6 recordings (52 minutes in total).

| file | length | content |
|---|---|---|
| `behavioral_mock_interview` | 7:12 | mock consulting interview with behavioural questions |
| `nasa_interview` | 10:00 | TV interview with the new NASA administrator |
| `psychology_raj_shamani` | 7:15 | a former law-enforcement guest on influence strategies |
| `beerbiceps` | 7:33 | the Bhagavad Gita and the "blueprint of reality" |
| `nikhilkamath` | 9:37 | Chamath Palihapitiya on investing and AI |
| `y_combinator` | 10:00 | YC partners on product-market fit |

**Query types** (about 10 per file):

| type | meaning | example |
|---|---|---|
| lexical | the exact words are spoken | `rat poison` |
| semantic | a paraphrase; the key words are never spoken | `will AI take away people's jobs` |
| mixed | a natural question that contains some spoken words | `risk of ruin with large investments` |

**Labels are time spans, not chunk ids**, so re-chunking never breaks them. Each span has a `why` field so a reviewer can check it quickly.

```json
{"query": "crypto", "type": "lexical",
 "spans": [
   {"file": "nikhilkamath.wav", "start": 173.0, "end": 187.5, "why": "Bitcoin in 2012, people were upset, vitriol on CNBC"},
   {"file": "nikhilkamath.wav", "start": 397.0, "end": 421.5, "why": "private crypto projects could go to zero or infinity"}
 ]}
```

**Labelling rules:**
- A span is any moment a listener searching this query would want to land on.
- Every such moment is labelled; 32 queries have 2 to 4 spans.
- A question and its immediate answer count as one moment.
- Some queries are traps with a correct answer in only one file. For example, "leadership experience" (behavioral) shares a word with "American leadership in space" (nasa).

**Scoring:**
- A result **hits** a span if it's from the same file and its time range overlaps the span at all.
- **Recall@5** counts spans: the share of a query's spans hit by at least one of the top 5 results. It is then averaged across queries.
- **MRR** = 1 / rank of the first hit, looking at the top 10 results. A miss counts as 0.

**Known failure cases:** 10 queries are marked `expect_fail`. They are kept on purpose to document what the system cannot do, and they are included in the averages.

| query | why it fails |
|---|---|
| `4.5%` | Whisper splits it into `4` `.5` `%`, so the stored text never matches (a fixable bug) |
| `Meta`, `copyrighted content`, `Sanskrit shloka` | speech-recognition errors ("Metta", "copper and content", garbled Sanskrit) |
| `Y Combinator` | the speakers only ever say "YC" |
| `artemus program` | typo in a name |
| `internships that were not technical` | embeddings ignore "not" |
| `how can we go to moon with less money` | vocabulary gap: the speaker says "affordably", never "money" |
| `what jobs has Evie had` | the answer is spread across the talk; search returns single moments and can't combine them |
| `questions the interviewer asked the candidate` | search can't filter by speaker, and there are 8 relevant moments but only 5 results (partial score) |

---

## Results

Latest run: [results/result_20260926_145727.md](results/result_20260926_145727.md)

| type | n | keyword R@5 | vector R@5 | **hybrid R@5** | hybrid MRR |
|---|---|---|---|---|---|
| lexical | 23 | 0.62 | 0.60 | **0.75** | 0.77 |
| semantic | 22 | 0.00 | 0.66 | **0.66** | 0.61 |
| mixed | 18 | 0.37 | 0.78 | **0.78** | 0.87 |
| all | 63 | 0.33 | 0.67 | **0.73** | 0.74 |

- **Hybrid is never worse than the better single search, for any query type.** On lexical queries it clearly beats both (0.75 against 0.62 and 0.60), because each search finds moments the other misses.
- **The targets are not met: lexical 0.75 against 0.80, and semantic 0.66 against 0.70.** Both averages include the 10 deliberate failure cases.
- **Biggest weakness:** keyword search requires every word to match, so it returns nothing for 37 of the 63 queries. On those queries hybrid can only equal vector search. Switching to OR matching is the next change to try.

---

## Run it on your computer

### Prerequisites
- Python 3.11
- Docker, for Postgres with pgvector
- ffmpeg, to convert audio (`brew install ffmpeg` on macOS)
- A Hugging Face account that has accepted the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). Only `diarize.py` needs this.

### 1. Install
```bash
git clone <repo-url> && cd hybrid-search-audio-g2
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Postgres with pgvector (port 5434)
```bash
docker run -d --name hybrid-audio-search -e POSTGRES_PASSWORD=postgres \
  -p 5434:5432 -v hybrid-audio-search:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```
`ingest.py` creates the `vector` extension, the table and the indexes itself.

### 3. Create `.env` in the repo root
```
HF_TOKEN=hf_your_token_here
# optional, this is the default:
DATABASE_URL=postgresql://postgres:postgres@localhost:5434/postgres
```

### 4. Add audio
Put 16 kHz mono `.wav` files in `audio/`. To convert an mp3:
```bash
ffmpeg -i raw_audio/nasa_interview.mp3 -t 600 -ac 1 -ar 16000 audio/nasa_interview.wav
```

### 5. Run the pipeline
Run everything from the repo root, in this order:
```bash
python pipeline/transcribe.py   # slow on CPU; skips files already transcribed
python pipeline/diarize.py      # slow on CPU; skips files already diarized
python pipeline/align.py
python pipeline/chunk.py
python pipeline/ingest.py       # truncates and reloads the table, safe to re-run
```
If `data/` already contains transcripts and diarization, you can skip the first two steps.

### 6. Search
```bash
python pipeline/search.py "why invest alone" -k 5              # hybrid (default)
python pipeline/search.py "Artemis program" --mode all -k 5    # keyword, vector and hybrid side by side
```

### 7. Evaluate
```bash
python pipeline/eval.py
```
It prints the report and saves it to `results/result_<timestamp>.md`.

### 8. Tests
```bash
pytest
```
25 tests. The two tests that query the live database skip themselves if Postgres is not running.

---

## Repo layout
```
audio/          16 kHz mono wav input
raw_audio/      original mp3s
audio_notes/    first hand-written notes the golden set started from
data/           output of each stage (transcripts, diarization, turns, chunks)
pipeline/       the 7 stage scripts + schema.sql
tests/          pytest unit tests
golden.json     labelled evaluation queries
results/        eval reports
design.md       system spec
prompts.md      build log: questions, decisions, and why
```
