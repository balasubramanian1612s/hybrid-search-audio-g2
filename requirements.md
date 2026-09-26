Model built over audio from 2 speakers.

Input: Query text
Output: Top k matches

Thought process:

How do I search over the audio?

Hybrid search: hence lexical + semantic search.

1. Transcribe

Options: faster-whisper, NVIDIA Parakeet TDT for CPU
Considered: faster-whisper
Rationale: Easy to plug in, faster, broader language support, same openai whisper model but faster

output will be sentence with the timestamps, and each word with the timestamp


2. Diarization

Options: pyannote, NVIDIA NeMo Sortformer
Considered: pyannote
Rationale: Easy setup, with very higher accuracy (open-source)

output will be texts with the speaker named to it

3. Aligning the Transcribed and Diarized texts together: Aligning

My own algorithm to align and form a sentence with proper time stamps, and with the speaker attached to it.

4. Chunking

Embedding: condidered local CPU friendy model all-MiniLM-L6-v2
Dimensions: 384.
max token: 256
Considering maximum words: 180
Chunk size matters for good search. Hence considering 80 words target for a chunk.

5. Ingest

Considering postgreSQL and pgvector. both in one, not separate db for vector, and storage.

Will be storing, timestamps, speaker, vector (semantic), text, tsvector (lexical)

GIN index for tsvector
HSNW index for pgvector

6. Search

Need to have 3 types of search, keyword, semantic, hybrid (keyword + semantic)
Gets the query, searches and gives me the top 10 with the ranks.

For hybrid search using Reciprocal Rank Fusion (RRF)

Rationale: The semantic search and lexical search has its own parameter, and output. it is not related to each other. Hence to normalize and give priority to both using RRF.

7. eval

Evaluation based on Recall@K again dataset

Need to think through - How I build it up at scale. What params to consider!