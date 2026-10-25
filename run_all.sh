#!/usr/bin/env bash
# Reproduce every experiment, in order. Usage:
#   ./run_all.sh            # all steps
#   ./run_all.sh 3          # start from step 3
# Paths: DATA_DIR (clue.zip), WORK_DIR, RESULTS_DIR (see src/config.py).
# Total time on 2 CPU / 8 GB RAM: about 4-5 hours (K3 window steps are the longest).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
DATA_DIR="${DATA_DIR:-$PWD/data}"; WORK_DIR="${WORK_DIR:-$PWD/work}"; RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"
export DATA_DIR WORK_DIR RESULTS_DIR
START="${1:-0}"
step() { local n="$1"; shift; if [ "$n" -ge "$START" ]; then echo "=== step $n: $*"; "$@"; fi; }

# 0. data: CLUE-LDS from Zenodo (635 MB, MD5 9e318370f96b68077667e9cdc05f26a5)
if [ "$START" -le 0 ]; then
  mkdir -p "$DATA_DIR"
  [ -f "$DATA_DIR/clue.zip" ] || wget -O "$DATA_DIR/clue.zip" https://zenodo.org/record/7119953/files/clue.zip
  echo "9e318370f96b68077667e9cdc05f26a5  $DATA_DIR/clue.zip" | md5sum -c -
fi
# 1. zip -> parquet (keeps original line order as column rn), ~7 min
step 1 $PY src/to_parquet.py --zip "$DATA_DIR/clue.zip" --out "$WORK_DIR/events.parquet" --work "$WORK_DIR"
# 2. per-user statistics for the swap pair selection
step 2 $PY src/user_stats.py
# 3. data checks: V1 field coverage, V2 aggregation; IP/geolocation coverage
step 3 $PY src/v1_v2.py "$WORK_DIR/events.parquet" "$RESULTS_DIR/v1_v2.json"
step 3 $PY src/ip_geo_coverage.py --data "$DATA_DIR/clue.zip" --out "$RESULTS_DIR/ip_geo_coverage.json"
# 4. per-user-day base tables for the models
step 4 $PY src/base_daily.py "$WORK_DIR/events.parquet" "$WORK_DIR/base"
# 5. baseline A (Landauer) — seed 1 with per-day dump (V4), seeds 0, 2-9 for reference
step 5 $PY src/duck_seed.py --parquet "$WORK_DIR/events.parquet" --stats "$WORK_DIR/user_stats.json" \
    --outdir "$WORK_DIR/A/seed1" -s 1 --dump "$WORK_DIR/A/seed1_t0.6.jsonl"
if [ 5 -ge "$START" ]; then
  for s in 0 2 3 4 5 6 7 8 9; do
    $PY src/duck_seed.py --parquet "$WORK_DIR/events.parquet" --stats "$WORK_DIR/user_stats.json" --outdir "$WORK_DIR/A/seed$s" -s $s > /dev/null
  done
  $PY src/summarize_seeds.py "$WORK_DIR/A" 3 4 5 6 7 8 9
fi
# 6. evaluator check: our evaluator on A's decisions must give TP 13 / FP 3337 (V4)
step 6 $PY src/check_evaluator.py
# 7. pengujian 1 (uid swap): all scenarios of Skenario_Pengujian.xlsx + V5-V7, ~70 min
step 7 $PY -W ignore src/run_all.py
# 8. exploratory (post-hoc): hybrid Landauer + context, relative features + calibration (X4), detector benchmark
step 8 $PY -W ignore src/hybrid.py
step 8 $PY -W ignore src/explore_rel.py
step 8 $PY -W ignore src/bench_detectors.py
# 9. pengujian 2: K3 session hijack, daily (seed 1, 3 intensities)
step 9 $PY -W ignore src/k3_day.py
# 10. pengujian 3: K3 per 15/30-min window
step 10 $PY -W ignore src/k3_window.py
# 11. pengujian 3b: improved window detector, 30/60 min, seeds 0-9
step 11 $PY -W ignore src/k3_window2.py
echo "done; compare $RESULTS_DIR with expected_results/"
