#!/usr/bin/env bash
# One command to set up and run the whole project:
#   Python env -> Postgres (Docker) -> audio -> pipeline -> eval -> tests -> demo search
#
# Usage:  ./init.sh
#
# Safe to re-run. Transcribe and diarize skip files that are already done,
# ingest reloads the table from scratch, eval writes a new results file.

set -euo pipefail
cd "$(dirname "$0")"

CONTAINER="hybrid-audio-search"
DB_PORT=5434
PY=".venv/bin/python"

step() { echo; echo "==> $*"; }
fail() { echo; echo "ERROR: $*" >&2; exit 1; }


step "1/8 Checking prerequisites"
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON311="python3.11"
elif python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
    PYTHON311="python3"
else
    fail "Python 3.11 not found. Install it (macOS: brew install python@3.11)."
fi
command -v docker >/dev/null 2>&1 || fail "Docker not found. Install Docker Desktop."
docker info >/dev/null 2>&1 || fail "Docker is installed but not running. Start Docker Desktop and re-run."
echo "python: $($PYTHON311 --version) | docker: running"


step "2/8 Python environment (.venv)"
if [ ! -x "$PY" ]; then
    "$PYTHON311" -m venv .venv
fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt
echo "dependencies installed"


step "3/8 Postgres + pgvector on port $DB_PORT"
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker start "$CONTAINER" >/dev/null
else
    docker run -d --name "$CONTAINER" \
        -e POSTGRES_PASSWORD=postgres \
        -p "$DB_PORT:5432" \
        -v hybrid-audio-search:/var/lib/postgresql/data \
        --restart unless-stopped \
        pgvector/pgvector:pg17 >/dev/null
fi
# Check over TCP: on first start the image runs a temporary init server that
# only listens on a unix socket, so a socket check can report ready too early.
ready="no"
for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" pg_isready -h 127.0.0.1 -U postgres >/dev/null 2>&1; then
        ready="yes"
        break
    fi
    sleep 1
done
[ "$ready" = "yes" ] || fail "Postgres did not start within 60s. Check: docker logs $CONTAINER"
echo "postgres ready"


step "4/8 Audio (raw_audio/*.mp3 -> audio/*.wav, 16 kHz mono, first 10 min)"
mkdir -p audio
shopt -s nullglob
for mp3 in raw_audio/*.mp3; do
    wav="audio/$(basename "${mp3%.mp3}").wav"
    if [ -f "$wav" ]; then
        continue
    fi
    command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is needed to convert $mp3 (macOS: brew install ffmpeg)."
    echo "converting $mp3"
    ffmpeg -loglevel error -i "$mp3" -t 600 -ac 1 -ar 16000 "$wav"
done
wavs=(audio/*.wav)
shopt -u nullglob
[ "${#wavs[@]}" -gt 0 ] || fail "No .wav files in audio/ and nothing to convert in raw_audio/."
echo "${#wavs[@]} audio files"


step "5/8 Pipeline: transcribe -> diarize -> align -> chunk -> ingest"
# transcribe/diarize are slow on CPU but skip files already in data/.
# diarize only needs HF_TOKEN in .env if a file still has to be diarized.
"$PY" pipeline/transcribe.py
"$PY" pipeline/diarize.py
"$PY" pipeline/align.py
"$PY" pipeline/chunk.py
"$PY" pipeline/ingest.py


step "6/8 Evaluation against golden.json"
"$PY" pipeline/eval.py


step "7/8 Tests"
"$PY" -m pytest -q


step "8/8 Demo search"
"$PY" pipeline/search.py "Artemis program" --mode all -k 3


echo
echo "=========================================================="
echo " Done. Everything is set up."
echo
echo " Search:   .venv/bin/python pipeline/search.py \"your query\" -k 5"
echo "           add --mode all to compare keyword, vector and hybrid"
echo " Eval:     .venv/bin/python pipeline/eval.py   (report in results/)"
echo "=========================================================="
