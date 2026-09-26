"""Stage 2: diarize audio/*.wav -> data/diarization/<stem>.json

Output shape:
    {file, turns: [{speaker, start, end}]}

Needs HF_TOKEN in .env (accept the model terms on Hugging Face first).
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

AUDIO_DIR = Path("audio")
OUT_DIR = Path("data/diarization")

MODEL_ID = "pyannote/speaker-diarization-community-1"
NUM_SPEAKERS = 2


def annotation_to_turns(annotation):
    turns = []
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "speaker": speaker,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
        })
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


def speaker_totals(turns):
    totals = defaultdict(float)
    for t in turns:
        totals[t["speaker"]] += t["end"] - t["start"]
    return dict(totals)


def diarize_file(pipeline, wav_path):
    with ProgressHook() as hook:
        output = pipeline(str(wav_path), num_speakers=NUM_SPEAKERS, hook=hook)
    # pyannote 4.x returns a DiarizeOutput, not an Annotation.
    # The exclusive version has no overlapping turns, which is what align.py wants.
    turns = annotation_to_turns(output.exclusive_speaker_diarization)
    return {"file": wav_path.name, "turns": turns}


def write_json(data, out_path):
    # Temp file + rename, so a crash never leaves a partial file behind.
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(out_path)


def print_speaker_totals(turns):
    totals = speaker_totals(turns)
    grand_total = sum(totals.values())
    for speaker in sorted(totals):
        seconds = totals[speaker]
        share = 100 * seconds / grand_total if grand_total else 0
        print(f"  {speaker}: {seconds:7.1f}s ({share:4.1f}%)")


def main():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set. Add it to .env")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_paths:
        print(f"No .wav files in {AUDIO_DIR}/")
        return

    pipeline = None  # load lazily, only if there is work to do
    for wav_path in wav_paths:
        out_path = OUT_DIR / f"{wav_path.stem}.json"
        if out_path.exists():
            print(f"skip {wav_path.name} (exists: {out_path})")
            continue

        if pipeline is None:
            print(f"loading {MODEL_ID}")
            pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)

        print(f"diarizing {wav_path.name}")
        t0 = time.time()
        data = diarize_file(pipeline, wav_path)
        write_json(data, out_path)

        print(f"done {wav_path.name}: {len(data['turns'])} turns in {time.time() - t0:.0f}s")
        print_speaker_totals(data["turns"])


if __name__ == "__main__":
    main()
