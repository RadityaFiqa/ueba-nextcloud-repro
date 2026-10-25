"""STEP 8 - EXPLORATION (post-hoc), uid swap: model Hybrid
(Landauer's per-user distance as score a, combined with context c and history h).
  Output: results/hybrid.json
  Run   : python -W ignore src/step08_explore_model_hybrid.py

------------------------------------------------------------------------
Exploratory (post-hoc) experiment: hybrid of Landauer's per-user score with
context (c) and history (h). Designed after the main results were known, so it
does not replace H1. Tuning on seeds 0 and 2, reported on seed 1.

Steps
  0. sanity: w=(1,0,0), theta=0.6 must reproduce baseline A on seed 1 exactly;
  1. H-hybrid: weights from the P2 set, lambda = 0.3 (locked D), theta grid;
  2. H-gate: alert iff a > theta_a and c >= theta_c;
  3. seed-1 test, McNemar + bootstrap vs A, theta curve on seed 1 (reporting only).
"""
import config  # data / work / results paths (see config.py)
import json
import os
import time

import numpy as np
from scipy.stats import binomtest

import lib_models as lib  # features F1-F8, models B/C/D/X4, Landauer detector, evaluator

STATS = config.STATS
VAL, TEST = [0, 2], 1
W_FEAT, LAM = 14, 0.3                      # locked in P1 / P3
TH_R = [round(x, 2) for x in np.arange(0.30, 0.951, 0.01)]
TH_A = [round(x, 2) for x in np.arange(0.30, 0.951, 0.02)]
TH_C = [k / 8 for k in range(0, 9)]
WEIGHTS = [(0.6, 0.3, 0.1), (0.5, 0.4, 0.1), (0.7, 0.2, 0.1), (0.5, 0.3, 0.2)]
OUT = os.path.join(config.RESULTS, 'hybrid.json')
LOG = open(os.path.join(config.RESULTS, 'hybrid.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


T0 = time.time()
BASE, _ = lib.load_base(config.BASE)
DATA = {}
for s in VAL + [TEST]:
    pairs, ct = lib.seed_swaps(STATS, s)
    f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)
    _, c = lib.flags_c(f['F'])
    DATA[s] = (f, c, lib.labels_from_swaps(pairs, ct))
log(f'data ready {time.time() - T0:.0f}s')


def run(s, **kw):
    f, c, lab = DATA[s]
    return lib.hybrid_detect(f, c, labels=lab, lam=LAM, **kw)


def pooled(**kw):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        m = run(s, **kw)[2]
        for k in t:
            t[k] += m[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    return t


def _pooled_kw(kw):
    return pooled(**kw)


def pooled_many(kws):
    """Evaluate many configs on both validation seeds using 2 forked workers (DATA is shared)."""
    import multiprocessing as mp
    with mp.get_context('fork').Pool(2) as pool:
        return pool.map(_pooled_kw, kws, chunksize=4)


def choose(c):
    ok = [x for x in c if x[1]['TPR'] >= 0.65]
    return min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(c, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))


res = {'meta': 'exploratory/post-hoc; tuning seeds 0+2, report seed 1; features W=14, lambda=0.3; strict > threshold'}

# 0. sanity
t = time.time()
st0, dc0, m0, _, _ = run(TEST, w=(1, 0, 0), theta=0.6)
res['sanity_w100_t0.6'] = m0
log(f"sanity (1,0,0) theta 0.6: TP={m0['TP']} FN={m0['FN']} FP={m0['FP']} TN={m0['TN']} train={m0['train']} "
    f"({time.time() - t:.1f}s per run)")
assert (m0['TP'], m0['FN'], m0['FP'], m0['TN'], m0['train']) == (13, 7, 3337, 69112, 5289), 'hybrid != detect.py'
A_dec = {}
for line in open(config.A_DUMP):
    u, d, *det = json.loads(line)
    A_dec[(u, d)] = det
f1 = DATA[TEST][0]
mism = sum(1 for i, (u, d) in enumerate(zip(f1['uid'], f1['d'])) if st0[i] and A_dec.get((u, d)) != [st0[i], dc0[i]])
res['sanity_per_day_mismatches'] = mism
log('sanity per-day mismatches vs detect.py:', mism)

# 1. H-hybrid: weights x theta on validation
cands = []
for w in WEIGHTS:
    kws = [dict(w=w, theta=th) for th in TH_R]
    cw = [((w, kw['theta']), m) for kw, m in zip(kws, pooled_many(kws))]
    best = choose(cw)
    log(f'  val w={w}: theta={best[0][1]} TPR={best[1]["TPR"]:.3f} FPR={best[1]["FPR"]:.4f}')
    cands.append(best)
(w_h, th_h), val_h = choose(cands)
res['H_hybrid'] = {'weights': w_h, 'theta': th_h, 'validation': val_h,
                   'candidates': {str(k[0]): {'theta': k[1], **m} for k, m in cands}}
log(f'H-hybrid locked w={w_h} theta={th_h}')

# 2. H-gate
kws = [dict(gate=(ta, tc)) for ta in TH_A for tc in TH_C]
cg = [(kw['gate'], m) for kw, m in zip(kws, pooled_many(kws))]
(ta, tc), val_g = choose(cg)
res['H_gate'] = {'theta_a': ta, 'theta_c': tc, 'validation': val_g}
log(f'H-gate locked a>{ta} c>={tc}: val TPR={val_g["TPR"]:.3f} FPR={val_g["FPR"]:.4f}')

# 3. test on seed 1
DEC = {'A': (st0, dc0)}
for name, kw in (('H_hybrid', dict(w=w_h, theta=th_h)), ('H_gate', dict(gate=(ta, tc)))):
    st, dc, m, _, _ = run(TEST, **kw)
    res[name]['test_seed1'] = m
    DEC[name] = (st, dc)
    log(f"TEST {name}: TP={m['TP']} FN={m['FN']} FP={m['FP']} TN={m['TN']} TPR={m['TPR']:.3f} "
        f"TPR_adj={m['TPR_adj']:.3f} FPR={m['FPR']:.4f} R={m['R']:.4f} alerts/day={m['alerts_per_day']:.2f}")
res['H_hybrid']['test_curve_seed1'] = {th: run(TEST, w=w_h, theta=th)[2] for th in TH_R[::5]}

lab1 = DATA[TEST][2]


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
    return {f'fp_only_{x}': b, f'fp_only_{y}': cc, 'p_value_exact': binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0,
            'dFPR': S[0] / S[1] - S[2] / S[3], 'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


res['vs_A'] = {x: compare(x) for x in ('H_hybrid', 'H_gate')}
for x, v in res['vs_A'].items():
    log(f'  {x} vs A:', v)
json.dump(res, open(OUT, 'w'), indent=1, default=str)
log(f'HYBRID_DONE in {time.time() - T0:.0f}s')
