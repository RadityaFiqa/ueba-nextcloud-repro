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
step 1 $PY src/step01_data_to_parquet.py --zip "$DATA_DIR/clue.zip" --out "$WORK_DIR/events.parquet" --work "$WORK_DIR"
# 2. per-user statistics for the swap pair selection
step 2 $PY src/step02_user_stats.py
# 3. data checks: V1 field coverage, V2 aggregation; IP/geolocation coverage
step 3 $PY src/step03_check_v1_v2_data.py "$WORK_DIR/events.parquet" "$RESULTS_DIR/v1_v2.json"
step 3 $PY src/step03_check_ip_geo_coverage.py --data "$DATA_DIR/clue.zip" --out "$RESULTS_DIR/ip_geo_coverage.json"
# 4. per-user-day base tables for the models
step 4 $PY src/step04_daily_tables.py "$WORK_DIR/events.parquet" "$WORK_DIR/base"
# 5. MODEL A = baseline Landauer — seed 1 with per-day dump (V4), seeds 0, 2-9 for reference
step 5 $PY src/step05_model_A_landauer.py --parquet "$WORK_DIR/events.parquet" --stats "$WORK_DIR/user_stats.json" \
    --outdir "$WORK_DIR/A/seed1" -s 1 --dump "$WORK_DIR/A/seed1_t0.6.jsonl"
if [ 5 -ge "$START" ]; then
  for s in 0 2 3 4 5 6 7 8 9; do
    $PY src/step05_model_A_landauer.py --parquet "$WORK_DIR/events.parquet" --stats "$WORK_DIR/user_stats.json" --outdir "$WORK_DIR/A/seed$s" -s $s > /dev/null
  done
  $PY src/step05_summarize_A_seeds.py "$WORK_DIR/A" 3 4 5 6 7 8 9
fi
# 6. evaluator check: our evaluator on A's decisions must give TP 13 / FP 3337 (V4)
step 6 $PY src/step06_check_evaluator_vs_A.py
# 7. Test 1 (uid swap): models B, B+RS, C, D + all scenarios of Skenario_Pengujian.xlsx, ~70 min
step 7 $PY -W ignore src/step07_test1_uidswap_models_B_C_D.py
# 8. exploration (post-hoc): model Hybrid, models X1-X4, detector benchmark
step 8 $PY -W ignore src/step08_explore_model_hybrid.py
step 8 $PY -W ignore src/step08_explore_model_X4_relative.py
step 8 $PY -W ignore src/step08_explore_detectors_on_X4.py
# 9. Test 2: K3 session hijack per day, models A, Hybrid, D, X4
step 9 $PY -W ignore src/step09_test2_k3_daily_A_D_X4_hybrid.py
# 10. Test 3: K3 per 15/30-min window, models A-win, X4-win
step 10 $PY -W ignore src/step10_test3_k3_window_Awin_X4win.py
# 11. Test 3b: K3 per 30/60-min window, models X5 vs A-win, X4-win
step 11 $PY -W ignore src/step11_test3b_k3_window_X5.py
echo "done; compare $RESULTS_DIR with expected_results/"
