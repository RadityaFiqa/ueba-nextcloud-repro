"""Exploration (post-hoc), pre-specified: models whose Isolation Forest uses ONLY the assessed user's
own history (one IF per user, trained on the user's previous 30 active days, retrained every 7 active
days; a = share of the user's training days that look more normal than today).

Models
  IF Personal             alarm if a >= theta
  Model Usulan Personal   R = wa*a + wc*c + wh*h (flags k1..k5 as Model Usulan, lambda 0.3), alarm if R >= theta
Days with fewer than 5 earlier feature days have no personal model; they are still scored, with a = 0.
Choices made on seeds 0+2 only (Test 1), rule: lowest FPR with TPR >= 0.65 (else highest TPR):
  X        log1p counts + F1..F8   or   relative counts (log1p(x) - log1p(median 14 days)) + F1..F8
  w        (.5,.4,.1), (.6,.3,.1), (.4,.5,.1), (.3,.6,.1)        (Model Usulan Personal)
  theta    0.40 .. 0.99
Locked choices are then reported on seed 1 for Test 1 (uid swap) and Test 2 (session hijack, 60 sessions).
Per-user scores are cached by (user, hash of the user's feature rows): across seeds only swapped/injected
users change, so later data sets cost little.
  Output: results/peruser_models.json
  Run   : python -W ignore tools/peruser_models.py <landauer_runs_dir>      (~45 min)
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

A_RUNS = sys.argv[1]
W, N, M, STEP, LAM, DAYS = 14, 30, 5, 7, 0.3, 1910
VAL, TEST = [0, 2], 1
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
WEIGHTS = [(0.5, 0.4, 0.1), (0.6, 0.3, 0.1), (0.4, 0.5, 0.1), (0.3, 0.6, 0.1)]
XSETS = ['log', 'rel']
OUTF = os.path.join(config.RESULTS, 'peruser_models.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
CACHE = {xs: {} for xs in XSETS}


def design(f, xs):
    A = lib.relative_A(f, W=W) if xs == 'rel' else np.log1p(f['A'].astype(np.float64))
    return np.nan_to_num(np.hstack([A, f['F']]).astype(np.float32))


def peruser_a(f, X, xs):
    """As lib.iforest_a_peruser, with a per-user cache; a = 0 where no personal model exists yet."""
    starts = np.flatnonzero(f['first'])
    bounds = list(zip(starts, list(starts[1:]) + [len(X)]))
    a = np.zeros(len(X))
    todo = []
    for s, e in bounds:
        if e - s <= 1 + M:
            continue
        key = (f['uid'][s], hashlib.md5(X[s:e].tobytes()).hexdigest())
        if key in CACHE[xs]:
            a[s:e] = CACHE[xs][key]
        else:
            todo.append((s, e, key))
    outs = Parallel(n_jobs=2, batch_size=16)(delayed(lib._peruser_one)(X[s:e], N, M, STEP, 0) for s, e, _ in todo)
    for (s, e, key), part in zip(todo, outs):
        part = np.nan_to_num(part, nan=0.0)
        CACHE[xs][key] = part
        a[s:e] = part
    return a, len(todo)


def prepare(recs):
    f = lib.compute_features(recs, W=W)
    _, c = lib.flags_c(f['F'])
    out = {'f': f, 'avail': ~f['first'], 'a': {}, 'R': {}}
    for xs in XSETS:
        a, n = peruser_a(f, design(f, xs), xs)
        out['a'][xs] = a
        for w in WEIGHTS:
            out['R'][(xs, w)] = np.nan_to_num(lib.risk(f, a, c, w, LAM), nan=-1)
        log(f'  {xs}: {n} users scored')
    return out


def run(d, lab, model, xs, w, th):
    s = d['a'][xs] if model == 'IF Personal' else d['R'][(xs, w)]
    return lib.evaluate(d['f'], s >= th, d['avail'], lab, r=1)[2]


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


def save(o):
    json.dump(o, open(OUTF, 'w'), indent=1, default=str)


# ---------------------------------------------------------------- 1. choose on seeds 0+2
V = {}
for s in VAL:
    t = time.time()
    pairs, ct = lib.seed_swaps(config.STATS, s)
    V[s] = (prepare(lib.relabel(BASE, pairs, ct)), lib.labels_from_swaps(pairs, ct))
    log(f'validation seed {s}: {time.time() - t:.0f}s')


def pooled(model, xs, w, th):
    m = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        r = run(*V[s], model, xs, w, th)
        for k in m:
            m[k] += r[k]
    m['TPR'], m['FPR'] = m['TP'] / (m['TP'] + m['FN']), m['FP'] / (m['FP'] + m['TN'])
    m['fp_per_day'] = m['FP'] / (len(VAL) * DAYS)
    return m


LOCK = {}
OUT = {'meta': __doc__.strip(), 'validation': {}}
for model in ['IF Personal', 'Model Usulan Personal']:
    grid = [(xs, w, th) for xs in XSETS for w in ([None] if model == 'IF Personal' else WEIGHTS) for th in TH]
    res = [(g, pooled(model, *g)) for g in grid]
    ok = [x for x in res if x[1]['TPR'] >= 0.65]
    best = min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(res, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    LOCK[model] = best[0]
    OUT['validation'][model] = {'X': best[0][0], 'w': best[0][1], 'theta': best[0][2],
                                **{k: round(v, 4) if isinstance(v, float) else v for k, v in best[1].items()}}
    for xs in XSETS:  # best per feature set, for the record
        sub = [x for x in res if x[0][0] == xs]
        okx = [x for x in sub if x[1]['TPR'] >= 0.65]
        bx = min(okx, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if okx else max(sub, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
        OUT['validation'][f'{model} [{xs}]'] = {'w': bx[0][1], 'theta': bx[0][2], 'TP': bx[1]['TP'], 'TPR': round(bx[1]['TPR'], 4),
                                                 'FPR': round(bx[1]['FPR'], 4), 'fp_per_day': round(bx[1]['fp_per_day'], 3)}
    log(f'LOCKED {model}: {OUT["validation"][model]}')
save(OUT)
del V

# ---------------------------------------------------------------- 2. Test 1, seed 1
L = json.load(open(os.path.join(A_RUNS, f'seed{TEST}', 'metrics.json')))
pairs, ct = lib.seed_swaps(config.STATS, TEST)
d = prepare(lib.relabel(BASE, pairs, ct))
lab = lib.labels_from_swaps(pairs, ct)
OUT['test1_seed1'] = {'Landauer': {'TP': L['TP'], 'FP': L['FP'], 'FPR': round(L['FP'] / (L['FP'] + L['TN']), 4),
                                   'alerts_per_day': round(L['FP'] / DAYS, 3)}}
for model, (xs, w, th) in LOCK.items():
    OUT['test1_seed1'][model] = short(run(d, lab, model, xs, w, th))
save(OUT)
log('test1 seed 1:', OUT['test1_seed1'])
del d

# ---------------------------------------------------------------- 3. Test 2, seed 1
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


T2 = {m: {'TP': 0, 'per': {}, 'apd': []} for m in LOCK}
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
    d = prepare(recs)
    for model, (xs, w, th) in LOCK.items():
        m = run(d, labels, model, xs, w, th)
        T2[model]['TP'] += m['TP']; T2[model]['per'][level] = m['TP']; T2[model]['apd'].append(m['alerts_per_day'])
    del d, recs
    log(f'test2 {level}: {time.time() - t:.0f}s')
OUT['test2_seed1'] = {k: {'TP_of_60': v['TP'], 'per': v['per'], 'alerts_per_day': round(float(np.mean(v['apd'])), 3)} for k, v in T2.items()}
save(OUT)
log('test2:', OUT['test2_seed1'])
log('DONE')
