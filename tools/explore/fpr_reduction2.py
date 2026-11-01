"""Additional exploration (post-hoc): further refinements of "X4 + strong flag" with the alarm
threshold FIXED at R >= 0.60 (no threshold tuning). Base rule: R >= 0.60 AND (k2 OR k4).
Refinements, combined factorially and chosen on seeds 0+2 (Test 1) with the usual rule
(lowest FPR with TPR >= 0.65), reported on seed 1 for Test 1 and Test 2 (no re-tuning):
  k1two   k1 fires on |F1| >= 3 (a sharp DROP in volume counts too; swaps often look like that)
  k5tight k5 fires on F7 >= 0.2 OR F8 >= 5 (instead of 0.1 / 3; k5 sits in 69% of X4's false alarms)
  r2      two adaptation days after an alarm instead of one (repeat alarms on the same user)
  dgate   also require Landauer's nearest-day distance (last 30 active days) >= t_d, t_d in 0.2..0.5
  Output: results/fpr_reduction2.json
  Run   : python -W ignore tools/fpr_reduction2.py      (~25 min)
"""
import itertools
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

VAL, TEST, TH = [0, 2], 1, 0.60
W_FEAT, W_W, LAM = 14, (0.5, 0.4, 0.1), 0.3
DG = [None, 0.2, 0.3, 0.4, 0.5]
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


def flags(F, k1two, k5tight):
    g = lambda i: np.nan_to_num(F[:, i], nan=-np.inf)
    k1 = (np.abs(np.nan_to_num(F[:, 0], nan=0)) >= 3) if k1two else (g(0) >= 3)
    k2 = (g(1) >= 0.5) | (g(2) >= 0.2)
    k3 = g(3) >= 0.3
    k4 = g(5) >= 0.5
    k5 = ((g(6) >= 0.2) | (g(7) >= 5)) if k5tight else ((g(6) >= 0.1) | (g(7) >= 3))
    return k2 | k4, (2 * k1 + 2 * k2 + k3 + 2 * k4 + k5) / 8.0


def prepare(f):
    X = np.hstack([lib.relative_A(f, W=W_FEAT), f['F']]).astype(np.float32)
    a_cal = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    d = {'f': f, 'avail': ~np.isnan(a_cal) & ~f['first'], 'dist': np.nan_to_num(nn_dist(f), nan=0.0), 'R': {}}
    for k1two, k5tight in itertools.product([False, True], repeat=2):
        strong, c = flags(f['F'], k1two, k5tight)
        d['R'][(k1two, k5tight)] = (np.nan_to_num(lib.risk(f, a_cal, c, W_W, LAM), nan=-1), strong)
    return d


CONFIGS = [dict(k1two=a, k5tight=b, r=r, dgate=g) for a, b, r, g in itertools.product([False, True], [False, True], [1, 2], DG)]


def name(cfg):
    parts = [p for p, on in (('k1two', cfg['k1two']), ('k5tight', cfg['k5tight']), ('r2', cfg['r'] == 2)) if on]
    if cfg['dgate'] is not None:
        parts.append(f"d>={cfg['dgate']}")
    return 'strong' + ('+' + '+'.join(parts) if parts else '')


def run(d, lab, cfg):
    R, strong = d['R'][(cfg['k1two'], cfg['k5tight'])]
    al = (R >= TH) & strong
    if cfg['dgate'] is not None:
        al &= d['dist'] >= cfg['dgate']
    return lib.evaluate(d['f'], al, d['avail'], lab, r=cfg['r'])[2]


def run_plain(d, lab):  # X4 as published: R >= 0.60 with the original flags
    R, _ = d['R'][(False, False)]
    return lib.evaluate(d['f'], R >= TH, d['avail'], lab, r=1)[2]


T1 = {}
for s in VAL + [TEST]:
    t = time.time()
    pairs, ct = lib.seed_swaps(config.STATS, s)
    T1[s] = (prepare(lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)), lib.labels_from_swaps(pairs, ct))
    log(f'test1 seed {s}: {time.time() - t:.0f}s')


def pooled(cfg):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        m = run(*T1[s], cfg)
        for k in t:
            t[k] += m[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    return t


VALRES = [(cfg, pooled(cfg)) for cfg in CONFIGS]
ok = [x for x in VALRES if x[1]['TPR'] >= 0.65]
best = min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR']))[0] if ok else None
log('validation best:', name(best) if best else None)
SINGLE = [dict(k1two=False, k5tight=False, r=1, dgate=None), dict(k1two=True, k5tight=False, r=1, dgate=None),
          dict(k1two=False, k5tight=True, r=1, dgate=None), dict(k1two=False, k5tight=False, r=2, dgate=None),
          dict(k1two=False, k5tight=False, r=1, dgate=0.3)]
REPORT = SINGLE + ([best] if best and best not in SINGLE else [])


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


OUT = {'meta': __doc__.strip(), 'threshold': TH, 'chosen_on_validation': name(best) if best else None,
       'validation_all': {name(c): {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()} for c, m in VALRES},
       'test1_seed1': {'X4 (R>=0.60)': short(run_plain(*T1[TEST]))}}
for cfg in REPORT:
    OUT['test1_seed1'][name(cfg)] = short(run(*T1[TEST], cfg))
del T1

# ---------------------------------------------------------------- Test 2, seed 1
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


T2 = {k: {'TP': 0, 'apd': []} for k in ['X4 (R>=0.60)'] + [name(c) for c in REPORT]}
for level in k3.LEVELS:
    t = time.time()
    recs = dict(CLEAN)
    for p in PLANS:
        ev = k3.attack_events(p, level, TEST, BASE, ALL_PREFIX)
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
    labels = {p['victim']: p['day'] for p in PLANS}
    d = prepare(lib.compute_features(recs, W=W_FEAT))
    for key, m in [('X4 (R>=0.60)', run_plain(d, labels))] + [(name(c), run(d, labels, c)) for c in REPORT]:
        T2[key]['TP'] += m['TP']; T2[key]['apd'].append(m['alerts_per_day'])
    log(f'test2 {level}: {time.time() - t:.0f}s')
OUT['test2_seed1'] = {k: {'TP_of_60': v['TP'], 'alerts_per_day': round(float(np.mean(v['apd'])), 3)} for k, v in T2.items()}
json.dump(OUT, open(os.path.join(config.RESULTS, 'fpr_reduction2.json'), 'w'), indent=1, default=str)
for k in OUT['test1_seed1']:
    a, b = OUT['test1_seed1'][k], OUT['test2_seed1'][k]
    log(f"{k:32s} T1 TP={a['TP']}/20 FPR={a['FPR']:.4f} alerts/day={a['alerts_per_day']:.2f} | "
        f"T2 TP={b['TP_of_60']}/60 alerts/day={b['alerts_per_day']:.2f}")
log('DONE')
