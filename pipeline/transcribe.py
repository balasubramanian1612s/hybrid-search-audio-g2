"""Stage 1: transcribe audio/*.wav -> data/transcripts/<stem>.json

Output shape:
    {file, duration, segments: [{id, text, start, end, words: [{word, start, end}]}]}
"""

import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

AUDIO_DIR = Path("audio")
OUT_DIR = Path("data/transcripts")

MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
LANGUAGE = "en"
MIN_SILENCE_MS = 500


def segment_to_dict(seg):
    words = []
    for w in seg.words or []:
        words.append({
            "word": w.word.strip(),
            "start": round(w.start, 3),
            "end": round(w.end, 3),
        })
    return {
        "id": seg.id,
        "text": seg.text.strip(),
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "words": words,
    }


def transcribe_file(model, wav_path):
    segments, info = model.transcribe(
        str(wav_path),
        beam_size=BEAM_SIZE,
        language=LANGUAGE,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": MIN_SILENCE_MS},
    )
    # `segments` is a lazy generator; transcription happens while we iterate.
    out_segments = []
    for seg in segments:
        out_segments.append(segment_to_dict(seg))
        print(f"  [{seg.start:7.1f}s / {info.duration:7.1f}s] {seg.text.strip()[:70]}")

    return {
        "file": wav_path.name,
        "duration": round(info.duration, 3),
        "segments": out_segments,
    }


def write_json(data, out_path):
    # Write to a temp file then rename, so a crash never leaves a partial
    # file that the skip-if-exists check would treat as done.
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_paths:
        print(f"No .wav files in {AUDIO_DIR}/")
        return

    model = None  # load lazily, only if there is work to do
    for wav_path in wav_paths:
        out_path = OUT_DIR / f"{wav_path.stem}.json"
        if out_path.exists():
            print(f"skip {wav_path.name} (exists: {out_path})")
            continue

        if model is None:
            print(f"loading faster-whisper {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})")
            model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

        print(f"transcribing {wav_path.name}")
        t0 = time.time()
        data = transcribe_file(model, wav_path)
        write_json(data, out_path)

        n_words = sum(len(s["words"]) for s in data["segments"])
        print(
            f"done {wav_path.name}: {len(data['segments'])} segments, "
            f"{n_words} words, {data['duration']:.0f}s audio in {time.time() - t0:.0f}s"
        )


if __name__ == "__main__":
    main()
