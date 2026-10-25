"""Exploratory (post-hoc) benchmark: which anomaly detector works best in the X4 setup?

X4 setup (explore_rel.py): relative Set A + F1-F8 (57 columns), one global model per
60-day window retrained every 7 days, per-user calibration of the score (mid-rank among
the user's previous 30 scored days), risk scoring R = 0.5a + 0.4c + 0.1h (lambda 0.3).
Only the detector changes: IF (lib.iforest_a, identical to X4), ECOD, COPOD, HBOS, LOF
(lib.rolling_a: same schedule; windows with < 25 training days are skipped).
Theta chosen on seeds 0+2, reported on seed 1. Does not replace H1.
"""
import config  # data / work / results paths (see config.py)
import json
import os
import time

import numpy as np
from scipy.stats import binomtest

import lib

STATS = config.STATS
VAL, TEST = [0, 2], 1
W_FEAT, W_W, LAM = 14, (0.5, 0.4, 0.1), 0.3
TH = [round(x, 2) for x in np.arange(0.30, 0.995, 0.01)] + [0.995, 0.999]
DETECTORS = ['IF', 'ECOD', 'COPOD', 'HBOS', 'LOF']
OUT = os.path.join(config.RESULTS, 'bench_detectors.json')
LOG = open(os.path.join(config.RESULTS, 'bench_detectors.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


T0 = time.time()
BASE, _ = lib.load_base(config.BASE)
DATA = {}
for s in VAL + [TEST]:
    pairs, ct = lib.seed_swaps(STATS, s)
    f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)
    X = np.hstack([lib.relative_A(f, W=W_FEAT), f['F']]).astype(np.float64)
    _, c = lib.flags_c(f['F'])
    DATA[s] = dict(f=f, X=X, c=c, lab=lib.labels_from_swaps(pairs, ct), a={}, sec={})
    for det in DETECTORS:
        t = time.time()
        if det == 'IF':
            a = lib.iforest_a(f, X.astype(np.float32), schedule='rolling')
        else:
            a = lib.rolling_a(f, X, lib.make_detector(det))
        a_cal = lib.calibrate_per_user(f, a)
        R = lib.risk(f, a_cal, c, W_W, LAM)
        DATA[s]['a'][det] = (a_cal, R)
        DATA[s]['sec'][det] = round(time.time() - t, 1)
        log(f'seed {s} {det}: {time.time() - t:.0f}s')


def run(s, det, th):
    d = DATA[s]
    a_cal, R = d['a'][det]
    avail = ~np.isnan(a_cal) & ~d['f']['first']
    return lib.evaluate(d['f'], np.nan_to_num(R, nan=-1) >= th, avail, d['lab'], r=1)


def pooled(det, th):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        m = run(s, det, th)[2]
        for k in t:
            t[k] += m[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    return t


def choose(c):
    ok = [x for x in c if x[1]['TPR'] >= 0.65]
    return min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(c, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))


f1, lab1 = DATA[TEST]['f'], DATA[TEST]['lab']
A = {}
for line in open(config.A_DUMP):
    u, d, *det = json.loads(line)
    A[(u, d)] = det
DEC = {'A': (np.array([A.get((u, d), ['', ''])[0] for u, d in zip(f1['uid'], f1['d'])], dtype=object),
             np.array([A.get((u, d), ['', ''])[-1] for u, d in zip(f1['uid'], f1['d'])], dtype=object))}


def compare(x, y='A', B=2000):
    sx, dx = DEC[x]; sy, dy = DEC[y]
    b = cc = 0
    per = {}
    for i in range(len(f1['d'])):
        u = f1['uid'][i]
        neg = lab1.get(u) != f1['d'][i]
        if sx[i] == 'detection' and sy[i] == 'detection' and neg:
            ax, ay = dx[i] == 'anomalous', dy[i] == 'anomalous'
            b += ax and not ay
            cc += ay and not ax
        for k, (s_, d_) in enumerate(((sx, dx), (sy, dy))):
            if s_[i] == 'detection' and neg:
                e = per.setdefault(u, [0, 0, 0, 0])
                e[2 * k] += d_[i] == 'anomalous'; e[2 * k + 1] += 1
    M = np.array(list(per.values()), float)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(B):
        S = M[rng.integers(0, len(M), len(M))].sum(0)
        diffs.append(S[0] / S[1] - S[2] / S[3])
    S = M.sum(0)
    return {'p_value_exact': binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0,
            'dFPR': S[0] / S[1] - S[2] / S[3], 'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


res = {'meta': __doc__.strip()}
for det in DETECTORS:
    th, val = choose([(th, pooled(det, th)) for th in TH])
    st, dc, m = run(TEST, det, th)
    m['PR_AUC'] = lib.pr_auc(f1, DATA[TEST]['a'][det][1], lab1)
    DEC[det] = (st, dc)
    res[det] = {'theta': th, 'validation': val, 'test_seed1': m, 'vs_A': compare(det),
                'seconds_per_seed': DATA[TEST]['sec'][det],
                'test_curve_seed1': {t: run(TEST, det, t)[2] for t in TH[::5]}}
    v = res[det]['vs_A']
    log(f"{det}: val theta={th} TPR={val['TPR']:.3f} FPR={val['FPR']:.4f} | TEST TP={m['TP']} FP={m['FP']} "
        f"TPR={m['TPR']:.3f} TPR_adj={m['TPR_adj']:.3f} FPR={m['FPR']:.4f} R={m['R']:.4f} alerts/day={m['alerts_per_day']:.2f} "
        f"PR-AUC={m['PR_AUC']:.4f} | dFPR vs A={v['dFPR']:+.4f} [{v['ci95'][0]:+.4f};{v['ci95'][1]:+.4f}] p={v['p_value_exact']:.1e}")
json.dump(res, open(OUT, 'w'), indent=1, default=str)
log(f'BENCH_DONE in {time.time() - T0:.0f}s')
