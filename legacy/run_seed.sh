#!/usr/bin/env bash
# Chunked baseline A for one seed:  ./run_seed.sh SEED [CHUNKS] [JOBS] [THRESH]
# Needs user_stats.json (python stats.py) and detect_chunk.py (make_detect_chunk.py).
set -euo pipefail
cd "$(dirname "$0")"
SEED="$1"; CHUNKS="${2:-8}"; JOBS="${3:-2}"; THRESH="${4:-0.6}"
DATA="${DATA:-../data/clue.zip}"
PY="${PY:-python3}"
STATS="${STATS:-user_stats.json}"
RUN="${RUNS:-runs}/seed${SEED}"
mkdir -p "$RUN"

"$PY" inject_split.py --data "$DATA" --stats "$STATS" --outdir "$RUN" --chunks "$CHUNKS" -s "$SEED" 2>&1 | tee "$RUN/inject.log"

seq 0 $((CHUNKS - 1)) | xargs -P "$JOBS" -I{} sh -c \
  "\"$PY\" detect_chunk.py -t $THRESH --input $RUN/chunk_{}.json --labels $RUN/labels.txt --result $RUN/result_{}.json > $RUN/detect_{}.log 2>&1"

"$PY" merge.py "$RUN" | tee "$RUN/metrics.txt"
[ "${KEEP_CHUNKS:-0}" = 1 ] || rm -f "$RUN"/chunk_*.json
