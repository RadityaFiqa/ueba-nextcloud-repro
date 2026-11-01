"""Robustness check (post-hoc), uid swap (Test 1) on seeds 0-9: model X4 and its two refinements,
all at the fixed threshold R >= 0.60, against Landauer's per-seed results (runs of step05).
  X4                 R >= 0.60
  X4+strong          R >= 0.60 AND (k2 OR k4)
  X4+strong+d0.5     R >= 0.60 AND (k2 OR k4) AND nearest-day Landauer distance (last 30 days) >= 0.5
Seeds 0 and 2 were used to choose the refinements; seeds 3-9 were never used for any choice.
  Output: results/x4_seeds.json
  Run   : python -W ignore tools/x4_seeds.py [landauer_runs_dir]      (~15 min)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

TH, W_FEAT, W_W, LAM = 0.60, 14, (0.5, 0.4, 0.1), 0.3
A_RUNS = sys.argv[1] if len(sys.argv) > 1 else config.A_DIR
T0 = time.time()


def nn_dist(feat, N=30):
    A = feat['A'].astype(np.float64)
    out = np.full(len(A), np.nan)
    starts = np.flatnonzero(feat['first'])
    for s, e in zip(starts, list(starts[1:]) + [len(A)]):
        for i in range(s + 1, e):
            Q = A[max(s, i - N):i]
            den = np.maximum(Q, A[i]).sum(1)
            out[i] = np.min(np.where(den > 0, np.abs(Q - A[i]).sum(1) / np.where(den > 0, den, 1), 0.0))
    return out


BASE, _ = lib.load_base(config.BASE)
OUT = {'meta': __doc__.strip(), 'seeds': {}}
for s in range(10):
    t = time.time()
    pairs, ct = lib.seed_swaps(config.STATS, s)
    lab = lib.labels_from_swaps(pairs, ct)
    f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)
    X = np.hstack([lib.relative_A(f, W=W_FEAT), f['F']]).astype(np.float32)
    a_cal = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    ks, c = lib.flags_c(f['F'])
    R = np.nan_to_num(lib.risk(f, a_cal, c, W_W, LAM), nan=-1)
    avail = ~np.isnan(a_cal) & ~f['first']
    strong = ks['k2'] | ks['k4']
    far = np.nan_to_num(nn_dist(f), nan=0.0) >= 0.5
    res = {}
    for name, al in (('X4', R >= TH), ('X4+strong', (R >= TH) & strong), ('X4+strong+d0.5', (R >= TH) & strong & far)):
        m = lib.evaluate(f, al, avail, lab, r=1)[2]
        res[name] = {k: (round(m[k], 4) if isinstance(m[k], float) else m[k])
                     for k in ('TP', 'FN', 'FP', 'TN', 'TP_adj', 'TPR', 'FPR', 'alerts_per_day')}
    mp = os.path.join(A_RUNS, f'seed{s}', 'metrics.json')
    if os.path.exists(mp):
        a = json.load(open(mp))
        res['Landauer'] = {'TP': a['TP'], 'FN': a['FN'], 'FP': a['FP'], 'TN': a['TN'], 'TP_adj': a['TP_adj'],
                           'TPR': round(a['TPR'], 4), 'FPR': round(a['FPR'], 4),
                           'alerts_per_day': round((a['TP'] + a['FP']) / 1910, 4)}
    OUT['seeds'][s] = res
    print(f'[{time.time() - T0:5.0f}s] seed {s}: ' + ' | '.join(f"{k} {v['TP']}/20 FPR={v['FPR']:.4f}" for k, v in res.items()), flush=True)
    json.dump(OUT, open(os.path.join(config.RESULTS, 'x4_seeds.json'), 'w'), indent=1)
print('DONE', flush=True)
