"""Stage 3: align transcript words with diarization spans.

input:  data/transcripts/<stem>.json + data/diarization/<stem>.json
output: data/turns/<stem>.json
        {file, turns: [{speaker, start, end, text, n_words, n_low,
                        words: [{word, start, end}]}]}

Rules:
- whisper splits some tokens into pieces ('4' '.5' '%', 'helium' '-3');
  a piece starting with '-', '.' or '%' is joined back onto the word before it
- drop diarization spans shorter than MIN_SPAN_SEC
- a word is never split; it goes to the span it overlaps most
- a word with no overlap goes to the nearest span and is marked low-confidence
- consecutive same-speaker words form a turn; gaps <= MAX_GAP_SEC are merged
- total token count out == total token count in (asserted)
- every character of the transcript survives into the turns (asserted)
"""

import json
from pathlib import Path

TRANSCRIPT_DIR = Path("data/transcripts")
DIARIZATION_DIR = Path("data/diarization")
OUT_DIR = Path("data/turns")

MIN_SPAN_SEC = 0.25
MAX_GAP_SEC = 2.0
N_LOW_TURNS_TO_PRINT = 15

# A word starting with one of these continues the previous word.
# '$' is deliberately not here: in 'over $100' the '$' starts a new word.
JOIN_PREFIXES = ("-", ".", "%")


def count_tokens(text):
    return len(text.split())


def load_words(transcript):
    words = []
    for seg in transcript["segments"]:
        for w in seg["words"]:
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
    return words


def join_split_tokens(words):
    """Join pieces whisper split off a word: '4' '.5' '%' -> '4.5%'."""
    joined = []
    for w in words:
        if joined and w["word"].startswith(JOIN_PREFIXES):
            prev = joined[-1]
            prev["word"] += w["word"]
            prev["end"] = max(prev["end"], w["end"])
        else:
            joined.append(dict(w))
    return joined


def non_space_chars(text):
    return "".join(text.split())


def filter_spans(spans, min_dur=MIN_SPAN_SEC):
    return [s for s in spans if s["end"] - s["start"] >= min_dur]


def overlap(word, span):
    return min(word["end"], span["end"]) - max(word["start"], span["start"])


def distance(word, span):
    """Gap in seconds between word and span; 0 if they touch or overlap."""
    if word["end"] < span["start"]:
        return span["start"] - word["end"]
    if word["start"] > span["end"]:
        return word["start"] - span["end"]
    return 0.0


def assign_speaker(word, spans):
    """Return (speaker, is_low_confidence) for one word."""
    if not spans:
        raise ValueError("no diarization spans left to assign words to")

    # 1. The span with the largest positive overlap wins.
    best_span = None
    best_overlap = 0.0
    for span in spans:
        ov = overlap(word, span)
        if ov > best_overlap:
            best_overlap = ov
            best_span = span
    if best_span is not None:
        return best_span["speaker"], False

    # 2. Zero-duration word (whisper emits a few): if it sits inside a span,
    #    that is a real match, not a guess.
    if word["end"] <= word["start"]:
        for span in spans:
            if span["start"] <= word["start"] <= span["end"]:
                return span["speaker"], False

    # 3. No overlap: fall back to the nearest span, flagged low-confidence.
    nearest = min(spans, key=lambda span: distance(word, span))
    return nearest["speaker"], True


def group_into_turns(words, max_gap=MAX_GAP_SEC):
    """words: [{word, start, end, speaker, low}] in time order."""
    turns = []
    current = None
    for w in words:
        same_turn = (
            current is not None
            and w["speaker"] == current["speaker"]
            and w["start"] - current["end"] <= max_gap
        )
        if not same_turn:
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "words": [],
                "n_words": 0,
                "n_low": 0,
            }
            turns.append(current)

        current["end"] = max(current["end"], w["end"])
        current["words"].append({"word": w["word"], "start": w["start"], "end": w["end"]})
        current["n_words"] += count_tokens(w["word"])
        if w["low"]:
            current["n_low"] += 1

    out = []
    for t in turns:
        out.append({
            "speaker": t["speaker"],
            "start": t["start"],
            "end": t["end"],
            "text": " ".join(w["word"] for w in t["words"]),
            "n_words": t["n_words"],
            "n_low": t["n_low"],
            # word timings let chunk.py give exact times when it splits a turn
            "words": t["words"],
        })
    return out


def align(transcript, diarization):
    raw_words = load_words(transcript)
    words = join_split_tokens(raw_words)
    spans = filter_spans(diarization["turns"])

    assigned = []
    for w in words:
        speaker, low = assign_speaker(w, spans)
        assigned.append({**w, "speaker": speaker, "low": low})

    turns = group_into_turns(assigned)

    tokens_in = sum(count_tokens(w["word"]) for w in words)
    tokens_out = sum(t["n_words"] for t in turns)
    assert tokens_in == tokens_out, f"token count changed: in={tokens_in} out={tokens_out}"

    chars_in = non_space_chars("".join(w["word"] for w in raw_words))
    chars_out = non_space_chars("".join(t["text"] for t in turns))
    assert chars_in == chars_out, "transcript text changed during alignment"

    return {"file": transcript["file"], "turns": turns}


def write_json(data, out_path):
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(out_path)


def print_report(stem, diarization, data):
    turns = data["turns"]
    n_spans = len(diarization["turns"])
    n_dropped = n_spans - len(filter_spans(diarization["turns"]))
    n_words = sum(t["n_words"] for t in turns)
    n_low = sum(t["n_low"] for t in turns)
    low_pct = 100 * n_low / n_words if n_words else 0

    print(
        f"{stem}: {len(turns)} turns, {n_words} words, "
        f"{n_low} low-confidence ({low_pct:.1f}%), "
        f"dropped {n_dropped}/{n_spans} spans < {MIN_SPAN_SEC}s"
    )

    low_turns = [t for t in turns if t["n_low"] > 0]
    for t in low_turns[:N_LOW_TURNS_TO_PRINT]:
        print(
            f"  [{t['start']:7.1f}-{t['end']:7.1f}] {t['speaker']} "
            f"low {t['n_low']}/{t['n_words']}: {t['text'][:80]}"
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_paths = sorted(TRANSCRIPT_DIR.glob("*.json"))
    if not transcript_paths:
        print(f"No transcripts in {TRANSCRIPT_DIR}/")
        return

    for transcript_path in transcript_paths:
        stem = transcript_path.stem
        diarization_path = DIARIZATION_DIR / f"{stem}.json"
        if not diarization_path.exists():
            print(f"skip {stem} (no diarization at {diarization_path})")
            continue

        with open(transcript_path) as f:
            transcript = json.load(f)
        with open(diarization_path) as f:
            diarization = json.load(f)

        data = align(transcript, diarization)
        write_json(data, OUT_DIR / f"{stem}.json")
        print_report(stem, diarization, data)


if __name__ == "__main__":
    main()
