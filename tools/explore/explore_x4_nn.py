"""Additional exploration (post-hoc), uid swap (Test 1): model X4 with the relative Set A computed
against the MOST SIMILAR stored day (Landauer style) instead of the median of the last 14 days.

  X4        : rel = log1p(x_t) - log1p(median of the previous 14 active days)       (as step08)
  X4-NN14   : rel = log1p(x_t) - log1p(x_nn), x_nn = the previous day (of the last 14 active days)
              with the smallest Landauer distance  sum|x - y| / sum max(x, y)
  X4-NN30   : same, memory of the last 30 active days (Landauer's memory size)
Everything else as X4: + F1-F8 (W = 14), global IF (200 trees, 60-day window, weekly retrain),
per-user calibration (30 days), R = 0.5a + 0.4c + 0.1h, lambda = 0.3, retraining mode r = 1.
Thresholds chosen on seeds 0+2, reported on seed 1, with two rules:
  rule_tpr    : lowest FPR with TPR >= 0.65 (the rule used for every model so far)
  rule_budget : highest TPR with false alarms <= 1.75 per day (Landauer's budget)
  Output: results/explore_x4_nn.json
  Run   : python -W ignore tools/explore_x4_nn.py      (~60 min)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

VAL, TEST, DAYS = [0, 2], 1, 1910
W_FEAT, W_W, LAM, BUDGET = 14, (0.5, 0.4, 0.1), 0.3, 1.75
TH = [round(x, 2) for x in np.arange(0.30, 0.995, 0.01)] + [0.995, 0.999]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def relative_A_nn(feat, N):
    """rel[t] = log1p(A_t) - log1p(A_nn), A_nn = the user's previous day (last N active days)
    nearest to day t by Landauer's distance. NaN on the user's first day."""
    A = feat['A'].astype(np.float64)
    out = np.full_like(A, np.nan)
    L = np.log1p(A)
    starts = np.flatnonzero(feat['first'])
    for s, e in zip(starts, list(starts[1:]) + [len(A)]):
        for i in range(s + 1, e):
            Q = A[max(s, i - N):i]
            den = np.maximum(Q, A[i]).sum(1)
            d = np.where(den > 0, np.abs(Q - A[i]).sum(1) / np.where(den > 0, den, 1), 0.0)
            out[i] = L[i] - L[max(s, i - N) + int(np.argmin(d))]
    return out


BASE, TYPES = lib.load_base(config.BASE)
VARS = {'X4': ('median', 14), 'X4-NN14': ('nn', 14), 'X4-NN30': ('nn', 30)}
DATA = {}
for s in VAL + [TEST]:
    t = time.time()
    pairs, ct = lib.seed_swaps(config.STATS, s)
    f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)
    _, c = lib.flags_c(f['F'])
    DATA[s] = {'f': f, 'lab': lib.labels_from_swaps(pairs, ct), 'R': {}}
    for v, (kind, N) in VARS.items():
        rel = lib.relative_A(f, W=N) if kind == 'median' else relative_A_nn(f, N)
        X = np.hstack([rel, f['F']]).astype(np.float32)
        a_cal = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
        DATA[s]['R'][v] = (lib.risk(f, a_cal, c, W_W, LAM), ~np.isnan(a_cal) & ~f['first'])
    log(f'seed {s}: {time.time() - t:.0f}s')


def run(s, v, th):
    d = DATA[s]
    R, avail = d['R'][v]
    return lib.evaluate(d['f'], np.nan_to_num(R, nan=-1) >= th, avail, d['lab'], r=1)[2]


def pooled(v, th):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        m = run(s, v, th)
        for k in t:
            t[k] += m[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    t['fp_per_day'] = t['FP'] / (len(VAL) * DAYS)
    return t


def pick(v):
    c = [(th, pooled(v, th)) for th in TH]
    ok = [x for x in c if x[1]['TPR'] >= 0.65]
    r1 = min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(c, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    inb = [x for x in c if x[1]['fp_per_day'] <= BUDGET]
    r2 = max(inb, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    return r1, r2


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k])
            for k in ('TP', 'FN', 'FP', 'TN', 'TP_adj', 'TPR', 'TPR_adj', 'FPR', 'alerts_per_day')}


OUT = {'meta': __doc__.strip(), 'variants': {}}
for v in VARS:
    r1, r2 = pick(v)
    res = {}
    for rule, (th, val) in (('rule_tpr', r1), ('rule_budget', r2)):
        m = run(TEST, v, th)
        res[rule] = {'theta': th, 'validation': {k: round(x, 4) if isinstance(x, float) else x for k, x in val.items()},
                     'test_seed1': short(m)}
        log(f"{v:8s} {rule:11s} theta={th}: val TPR={val['TPR']:.3f} FP/day={val['fp_per_day']:.2f} | "
            f"TEST TP={m['TP']}/20 FPR={m['FPR']:.4f} alerts/day={m['alerts_per_day']:.2f}")
    res['test_curve_seed1'] = {t: short(run(TEST, v, t)) for t in TH[::5]}
    OUT['variants'][v] = res
json.dump(OUT, open(os.path.join(config.RESULTS, 'explore_x4_nn.json'), 'w'), indent=1, default=str)
log('DONE')
