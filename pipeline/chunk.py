"""Stage 4: turn aligned speaker turns into search chunks.

input:  data/turns/<stem>.json
output: data/chunks/<stem>.json
        [{file, speaker, start_ms, end_ms, text, n_words, embed_text}]

Rules:
- chunk on speaker turns, not fixed windows
- drop turns that are only backchannels ("yeah", "right", "mm", ...)
- split turns longer than TARGET_MAX words at sentence boundaries
- merge chunks shorter than MIN_WORDS into an adjacent same-speaker chunk;
  if there is none, keep them as-is
- embed_text = last CONTEXT_WORDS words of the other speaker's most recent
  turn + this chunk's text. `text` stays clean for display.
"""

import json
import re
from pathlib import Path

TURNS_DIR = Path("data/turns")
OUT_DIR = Path("data/chunks")

MODEL_LIMIT = 180   # MiniLM truncates at 256 wordpiece tokens; 180 words is a safe margin
TARGET_MAX = 100
MIN_WORDS = 20
CONTEXT_WORDS = 25

BACKCHANNELS = {
    "yeah", "yes", "yep", "yup", "right", "okay", "ok", "sure",
    "mm", "mhm", "mm-hmm", "hmm", "uh-huh", "uh", "um", "oh", "wow",
}

SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")


def count_tokens(text):
    return len(text.split())


def words_n(words):
    return sum(count_tokens(w["word"]) for w in words)


def normalize(token):
    return re.sub(r"[^\w-]", "", token.lower())


def is_backchannel(turn):
    tokens = [normalize(t) for t in turn["text"].split()]
    tokens = [t for t in tokens if t]
    return len(tokens) > 0 and all(t in BACKCHANNELS for t in tokens)


def split_sentences(words):
    """Group words into sentences, ending a sentence at . ? or !"""
    sentences = []
    current = []
    for w in words:
        current.append(w)
        if SENTENCE_END.search(w["word"]):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def split_turn(words, target_max=TARGET_MAX):
    """Pack whole sentences into pieces of at most target_max words.

    A single sentence longer than target_max is cut every target_max words.
    """
    pieces = []
    current = []
    for sentence in split_sentences(words):
        if words_n(sentence) > target_max:
            if current:
                pieces.append(current)
                current = []
            for i in range(0, len(sentence), target_max):
                pieces.append(sentence[i:i + target_max])
            continue

        if current and words_n(current) + words_n(sentence) > target_max:
            pieces.append(current)
            current = []
        current = current + sentence
    if current:
        pieces.append(current)
    return pieces


def last_words(text, n):
    return " ".join(text.split()[-n:])


def context_for(turns, i, n=CONTEXT_WORDS):
    """Last n words of the most recent turn before turns[i] by another speaker."""
    for j in range(i - 1, -1, -1):
        if turns[j]["speaker"] != turns[i]["speaker"]:
            return last_words(turns[j]["text"], n)
    return ""


def can_merge(a, b):
    if a["speaker"] != b["speaker"]:
        return False
    return words_n(a["words"]) + words_n(b["words"]) + CONTEXT_WORDS <= MODEL_LIMIT


def merge(first, second):
    # The earlier chunk's context is the right one: both are the same speaker
    # with no other speaker in between.
    return {
        "speaker": first["speaker"],
        "words": first["words"] + second["words"],
        "context": first["context"],
    }


def merge_small(raw_chunks, min_words=MIN_WORDS):
    """Merge chunks under min_words into the previous same-speaker chunk,
    else into the next one; if neither exists, keep the chunk as-is."""
    chunks = list(raw_chunks)
    out = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if words_n(c["words"]) < min_words:
            if out and can_merge(out[-1], c):
                out[-1] = merge(out[-1], c)
                i += 1
                continue
            if i + 1 < len(chunks) and can_merge(c, chunks[i + 1]):
                chunks[i + 1] = merge(c, chunks[i + 1])
                i += 1
                continue
        out.append(c)
        i += 1
    return out


def to_output(file, c):
    text = " ".join(w["word"] for w in c["words"])
    embed_text = f"{c['context']} {text}" if c["context"] else text
    return {
        "file": file,
        "speaker": c["speaker"],
        "start_ms": round(c["words"][0]["start"] * 1000),
        "end_ms": round(max(w["end"] for w in c["words"]) * 1000),
        "text": text,
        "n_words": words_n(c["words"]),
        "embed_text": embed_text,
    }


def chunk(turns_data):
    turns = [t for t in turns_data["turns"] if not is_backchannel(t)]

    raw_chunks = []
    for i, turn in enumerate(turns):
        context = context_for(turns, i)
        for piece in split_turn(turn["words"]):
            raw_chunks.append({"speaker": turn["speaker"], "words": piece, "context": context})

    chunks = [to_output(turns_data["file"], c) for c in merge_small(raw_chunks)]

    for c in chunks:
        n_embed = count_tokens(c["embed_text"])
        assert n_embed <= MODEL_LIMIT, f"embed_text has {n_embed} words > {MODEL_LIMIT}: {c['text'][:60]}"
    return chunks


def write_json(data, out_path):
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(out_path)


def print_report(stem, turns_data, chunks):
    n_backchannel = sum(1 for t in turns_data["turns"] if is_backchannel(t))
    sizes = sorted(c["n_words"] for c in chunks)
    n_small = sum(1 for s in sizes if s < MIN_WORDS)
    max_embed = max(count_tokens(c["embed_text"]) for c in chunks)
    print(
        f"{stem}: {len(chunks)} chunks from {len(turns_data['turns'])} turns "
        f"(dropped {n_backchannel} backchannel turns) | "
        f"words min {sizes[0]} / median {sizes[len(sizes) // 2]} / max {sizes[-1]} | "
        f"{n_small} kept under {MIN_WORDS} | max embed_text {max_embed} words"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    turns_paths = sorted(TURNS_DIR.glob("*.json"))
    if not turns_paths:
        print(f"No turns in {TURNS_DIR}/")
        return

    for turns_path in turns_paths:
        with open(turns_path) as f:
            turns_data = json.load(f)
        chunks = chunk(turns_data)
        write_json(chunks, OUT_DIR / f"{turns_path.stem}.json")
        print_report(turns_path.stem, turns_data, chunks)


if __name__ == "__main__":
    main()
