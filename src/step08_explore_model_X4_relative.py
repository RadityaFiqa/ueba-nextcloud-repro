"""STEP 8 - EXPLORATION (post-hoc), uid swap: models X1-X4
(features relative to the user's own history + per-user score calibration).
X4 is reused in Tests 2 and 3.
  Output: results/explore_rel.json
  Run   : python -W ignore src/step08_explore_model_X4_relative.py

------------------------------------------------------------------------
Exploratory (post-hoc) experiment: one global Isolation Forest on user-relative
features (relative Set A + F1-F8) with per-user score calibration. Designed after
the main results were known, so it does not replace H1. Tuning on seeds 0 and 2,
reported on seed 1. Same IF settings and schedule as D (200 trees, 256 samples,
60-day window, weekly retrain) -> ~273 models per run.

Variants
  X1 C-rel      : IF on relative features, alert a >= theta            (option 2)
  X2 C-rel-cal  : same IF, per-user calibrated a_cal >= theta           (option 2+3)
  X3 D-rel      : risk R = 0.5a + 0.4c + 0.1h with a (lambda 0.3)       (option 2 + scoring)
  X4 D-rel-cal  : risk with a_cal                                        (option 2+3 + scoring)
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
W_FEAT, W_W, LAM = 14, (0.5, 0.4, 0.1), 0.3        # locked in P1-P3
TH = [round(x, 2) for x in np.arange(0.30, 0.995, 0.01)] + [0.995, 0.999]
OUT = os.path.join(config.RESULTS, 'explore_rel.json')
LOG = open(os.path.join(config.RESULTS, 'explore_rel.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


T0 = time.time()
BASE, TYPES = lib.load_base(config.BASE)
DATA = {}
for s in VAL + [TEST]:
    t = time.time()
    pairs, ct = lib.seed_swaps(STATS, s)
    f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)
    X = np.hstack([lib.relative_A(f, W=W_FEAT), f['F']]).astype(np.float32)
    a = lib.iforest_a(f, X, schedule='rolling')
    a_cal = lib.calibrate_per_user(f, a)
    _, c = lib.flags_c(f['F'])
    DATA[s] = dict(f=f, a=a, a_cal=a_cal, c=c, lab=lib.labels_from_swaps(pairs, ct))
    log(f'seed {s}: features + IF + calibration {time.time() - t:.0f}s')

VARIANTS = {
    'X1_C-rel': dict(score='a', risk=False),
    'X2_C-rel-cal': dict(score='a_cal', risk=False),
    'X3_D-rel': dict(score='a', risk=True),
    'X4_D-rel-cal': dict(score='a_cal', risk=True),
}
CACHE = {}


def score(s, v):
    key = (s, v['score'], v['risk'])
    if key not in CACHE:
        d = DATA[s]
        a = d[v['score']]
        sc = lib.risk(d['f'], a, d['c'], W_W, LAM) if v['risk'] else a
        CACHE[key] = (sc, ~np.isnan(a) & ~d['f']['first'])
    return CACHE[key]


def run(s, v, th):
    d = DATA[s]
    sc, avail = score(s, v)
    return lib.evaluate(d['f'], np.nan_to_num(sc, nan=-1) >= th, avail, d['lab'], r=1)


def pooled(v, th):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        m = run(s, v, th)[2]
        for k in t:
            t[k] += m[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    return t


def choose(c):
    ok = [x for x in c if x[1]['TPR'] >= 0.65]
    return min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(c, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))


res = {'meta': 'exploratory/post-hoc; tuning seeds 0+2, report seed 1; IF settings as D; '
               'relative Set A = log1p(x) - log1p(median of previous 14 active days); calibration = mid-rank '
               'among previous 30 scored days of the user (fallback to global a with < 5)'}
DEC = {}
for name, v in VARIANTS.items():
    th, val = choose([(th, pooled(v, th)) for th in TH])
    st, dc, m = run(TEST, v, th)
    sc, _ = score(TEST, v)
    m['PR_AUC'] = lib.pr_auc(DATA[TEST]['f'], sc, DATA[TEST]['lab'])
    res[name] = {'theta': th, 'validation': val, 'test_seed1': m,
                 'test_curve_seed1': {t: run(TEST, v, t)[2] for t in TH[::5]}}
    DEC[name] = (st, dc)
    log(f"{name}: val theta={th} TPR={val['TPR']:.3f} FPR={val['FPR']:.4f} | TEST TP={m['TP']} FN={m['FN']} "
        f"FP={m['FP']} TPR={m['TPR']:.3f} TPR_adj={m['TPR_adj']:.3f} FPR={m['FPR']:.4f} R={m['R']:.4f} "
        f"alerts/day={m['alerts_per_day']:.2f} PR-AUC={m['PR_AUC']:.4f}")

# comparison with A (baseline decisions dumped from detect.py reimplementation)
f1, lab1 = DATA[TEST]['f'], DATA[TEST]['lab']
A = {}
for line in open(config.A_DUMP):
    u, d, *det = json.loads(line)
    A[(u, d)] = det
DEC['A'] = (np.array([A.get((u, d), ['', ''])[0] for u, d in zip(f1['uid'], f1['d'])], dtype=object),
            np.array([A.get((u, d), ['', ''])[-1] for u, d in zip(f1['uid'], f1['d'])], dtype=object))


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


res['vs_A'] = {x: compare(x) for x in VARIANTS}
for x, v in res['vs_A'].items():
    log(f'  {x} vs A:', v)
json.dump(res, open(OUT, 'w'), indent=1, default=str)
log(f'EXPLORE_REL_DONE in {time.time() - T0:.0f}s')
