"""STEP 6 - CHECK V4: our evaluator (lib_models.evaluate) must reproduce model A exactly.
  Input : work/base, work/user_stats.json, work/A/seed1_t0.6.jsonl, work/A/seed1/labels.txt
  Output: stdout, ends with "EVALUATOR CHECK PASS"
  Run   : python src/step06_check_evaluator_vs_A.py

------------------------------------------------------------------------
Check lib.evaluate against Landauer: feed baseline A's own per-day decisions
(seed 1, theta 0.6) and require identical statuses and TP/FP/TN/FN/train.
Also times load_base / relabel / compute_features."""
import config  # data / work / results paths (see config.py)
import json
import os
import time

import numpy as np

import lib_models as lib  # features F1-F8, models B/C/D/X4, Landauer detector, evaluator

t = time.time()
base, types = lib.load_base(config.BASE)
print(f'load_base {time.time() - t:.1f}s, {sum(len(v) for v in base.values()):,} user-days, {len(types)} types', flush=True)
pairs, ct = lib.seed_swaps(config.STATS, 1)
t = time.time()
recs = lib.relabel(base, pairs, ct)
feat = lib.compute_features(recs, W=30, M=5)
print(f'relabel+features {time.time() - t:.1f}s, rows {len(feat["d"]):,}', flush=True)
labels = lib.labels_from_swaps(pairs, ct)
assert labels == lib.read_labels(config.A_LABELS)

A = {}
for line in open(config.A_DUMP):
    u, d, *det = json.loads(line)
    A[(u, d)] = det
alert = np.array([A.get((u, d), ['', ''])[-1] == 'anomalous' for u, d in zip(feat['uid'], feat['d'])])
avail = np.ones(len(alert), bool)
status, dec, m = lib.evaluate(feat, alert, avail, labels, r=1)
mism = sum(1 for i, (u, d) in enumerate(zip(feat['uid'], feat['d']))
           if status[i] and A.get((u, d)) != [status[i], dec[i]])
missing = sum(1 for i in range(len(status)) if status[i] and (feat['uid'][i], feat['d'][i]) not in A)
extra = len(A) - sum(1 for s in status if s)
print({k: m[k] for k in ('TP', 'FN', 'FP', 'TN', 'train', 'TPR', 'FPR', 'R')})
print(f'per-day status/decision mismatches: {mism}, missing: {missing}, extra in A: {extra}')
ok = (m['TP'], m['FN'], m['FP'], m['TN'], m['train']) == (13, 7, 3337, 69112, 5289) and mism == 0 and extra == 0
print('EVALUATOR CHECK', 'PASS' if ok else 'FAIL')
print('F NaN share on non-first days:', float(np.isnan(feat['F'][~feat['first']]).mean()))
print('F summary (non-first):', np.round(np.nanpercentile(feat['F'][~feat['first']], [50, 95, 99], axis=0), 3).tolist())
