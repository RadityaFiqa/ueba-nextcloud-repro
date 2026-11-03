#!/bin/bash
# STEP 12 - FINAL experiment (numbers of the thesis and the paper).
#   1. transitions table (order of activities), once
#   2. data preparation in two parallel parts (one per CPU), ~45 min on 2 CPUs / 8 GB
#   3. evaluation -> results/final.json and results/final_curve.csv, ~10 min
# Run from the repository root after steps 01-04:  bash src/step12_final.sh
set -e
PY=${PY:-python}
RES=${RESULTS_DIR:-results}
mkdir -p "$RES"
$PY -W ignore src/step12_final_prep.py T
$PY -W ignore src/step12_final_prep.py A > "$RES/final_prep_A.log" 2>&1 &
$PY -W ignore src/step12_final_prep.py B > "$RES/final_prep_B.log" 2>&1 &
wait %1 && wait %2  # stop here if a preparation part failed
$PY -W ignore src/step12_final_eval.py | tee "$RES/final_eval.log"
