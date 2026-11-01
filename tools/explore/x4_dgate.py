"""Additional exploration (post-hoc): X4 + Landauer distance gate only (no k2/k4 requirement).
Alarm rule: R >= 0.60 AND Landauer's nearest-day distance (last 30 active days) >= t_d.
t_d in {none, 0.2, 0.3, 0.4, 0.5, 0.6} chosen on seeds 0+2 (Test 1) with the usual rule (lowest FPR
with TPR >= 0.65), reported on seed 1 for Test 1 and Test 2 (no re-tuning), then on seeds 3-9 (Test 1).
  Output: results/x4_dgate.json
  Run   : python -W ignore tools/x4_dgate.py      (~50 min)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

VAL, TEST, ROBUST, TH = [0, 2], 1, [3, 4, 5, 6, 7, 8, 9], 0.60
W_FEAT, W_W, LAM = 14, (0.5, 0.4, 0.1), 0.3
DG = [None, 0.2, 0.3, 0.4, 0.5, 0.6]
OUTF = os.path.join(config.RESULTS, 'x4_dgate.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}


def nn_dist(feat, N=30):
    """Landauer distance of each day to the nearest of the user's previous N active days (NaN first day)."""
    A = feat['A'].astype(np.float64)
    out = np.full(len(A), np.nan)
    starts = np.flatnonzero(feat['first'])
    for s, e in zip(starts, list(starts[1:]) + [len(A)]):
        for i in range(s + 1, e):
            Q = A[max(s, i - N):i]
            den = np.maximum(Q, A[i]).sum(1)
            d = np.where(den > 0, np.abs(Q - A[i]).sum(1) / np.where(den > 0, den, 1), 0.0)
            out[i] = d.min()
    return out


def prepare(f):  # X4 exactly as published: relative Set A + F1-F8, per-user calibration, original flags
    X = np.hstack([lib.relative_A(f, W=W_FEAT), f['F']]).astype(np.float32)
    a_cal = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    F = f['F']
    g = lambda i: np.nan_to_num(F[:, i], nan=-np.inf)
    k1, k2, k3_, k4 = g(0) >= 3, (g(1) >= 0.5) | (g(2) >= 0.2), g(3) >= 0.3, g(5) >= 0.5
    k5 = (g(6) >= 0.1) | (g(7) >= 3)
    c = (2 * k1 + 2 * k2 + k3_ + 2 * k4 + k5) / 8.0
    return {'f': f, 'avail': ~np.isnan(a_cal) & ~f['first'], 'dist': np.nan_to_num(nn_dist(f), nan=0.0),
            'R': np.nan_to_num(lib.risk(f, a_cal, c, W_W, LAM), nan=-1)}


def name(t):
    return 'X4' if t is None else f'X4+d>={t}'


def run(d, lab, t):
    al = d['R'] >= TH
    if t is not None:
        al &= d['dist'] >= t
    return lib.evaluate(d['f'], al, d['avail'], lab, r=1)[2]


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


def test1(s):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    return prepare(lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)), lib.labels_from_swaps(pairs, ct)


def save(out):
    json.dump(out, open(OUTF, 'w'), indent=1, default=str)


# ---------------------------------------------------------------- validation (seeds 0+2) and Test 1 seed 1
T1 = {}
for s in VAL + [TEST]:
    t = time.time()
    T1[s] = test1(s)
    log(f'test1 seed {s}: {time.time() - t:.0f}s')

VALRES = {}
for t in DG:
    m = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        r = run(*T1[s], t)
        for k in m:
            m[k] += r[k]
    m['TPR'], m['FPR'] = m['TP'] / (m['TP'] + m['FN']), m['FP'] / (m['FP'] + m['TN'])
    VALRES[t] = m
ok = [t for t in DG if VALRES[t]['TPR'] >= 0.65]
best = min(ok, key=lambda t: (VALRES[t]['FPR'], -VALRES[t]['TPR']))
log('validation best:', name(best))
OUT = {'meta': __doc__.strip(), 'threshold': TH, 'chosen_on_validation': name(best),
       'validation_all': {name(t): {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()} for t, m in VALRES.items()},
       'test1_seed1': {name(t): short(run(*T1[TEST], t)) for t in DG}}
save(OUT)
del T1

# ---------------------------------------------------------------- Test 2, seed 1 (as fpr_reduction2)
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(TEST, BASE, W30, ELIG)
CLEAN = lib.relabel(BASE, {}, {})


def inject(rec, events):  # as step09
    n = lib.Rec()
    n.d = rec.d
    n.types, n.hist = rec.types.copy(), rec.hist.copy()
    n.total, n.sens, n.pathcnt = rec.total, rec.sens, rec.pathcnt
    n.first, n.last, n.pref = rec.first, rec.last, dict(rec.pref)
    for ts, typ, pref in events:
        mi = int((ts - rec.d * 86400) // 60)
        n.types[TIX[typ]] += 1
        n.hist[mi // 5] += 1
        n.total += 1
        n.first, n.last = min(n.first, mi), max(n.last, mi)
        if typ in SENSITIVE:
            n.sens += 1
        if pref is not None:
            n.pathcnt += 1
            n.pref[pref] = n.pref.get(pref, 0) + 1
    n.peak10 = float(n.hist.reshape(-1, 2).sum(1).max())
    return n


T2 = {name(t): {'TP': 0, 'per': {}, 'apd': []} for t in DG}
for level in k3.LEVELS:
    t0 = time.time()
    recs = dict(CLEAN)
    for p in PLANS:
        ev = k3.attack_events(p, level, TEST, BASE, ALL_PREFIX)
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
    labels = {p['victim']: p['day'] for p in PLANS}
    d = prepare(lib.compute_features(recs, W=W_FEAT))
    for t in DG:
        m = run(d, labels, t)
        T2[name(t)]['TP'] += m['TP']; T2[name(t)]['per'][level] = m['TP']; T2[name(t)]['apd'].append(m['alerts_per_day'])
    del d, recs
    log(f'test2 {level}: {time.time() - t0:.0f}s')
OUT['test2_seed1'] = {k: {'TP_of_60': v['TP'], 'per': v['per'], 'alerts_per_day': round(float(np.mean(v['apd'])), 3)} for k, v in T2.items()}
save(OUT)
for k in OUT['test1_seed1']:
    a, b = OUT['test1_seed1'][k], OUT['test2_seed1'][k]
    log(f"{k:12s} T1 TP={a['TP']}/20 FPR={a['FPR']:.4f} alerts/day={a['alerts_per_day']:.2f} | "
        f"T2 TP={b['TP_of_60']}/60 alerts/day={b['alerts_per_day']:.2f}")

# ---------------------------------------------------------------- robustness: Test 1 on seeds 3-9
OUT['test1_seeds3_9'] = {}
for s in ROBUST:
    t0 = time.time()
    d, lab = test1(s)
    OUT['test1_seeds3_9'][s] = {name(t): short(run(d, lab, t)) for t in (None, best)}
    del d
    save(OUT)
    log(f'seed {s}: ' + ' | '.join(f"{k} TP={v['TP']} FPR={v['FPR']:.4f}" for k, v in OUT['test1_seeds3_9'][s].items())
        + f' ({time.time() - t0:.0f}s)')
log('DONE')
