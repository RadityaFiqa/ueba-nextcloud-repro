"""Exploration (post-hoc), pre-specified: history length W of the user-relative model X4.
W in {7, 14, 25} active days (25 = Landauer's learning period) is used for BOTH the relative counts
log1p(x) - log1p(median of the previous W days) and the contextual features F1..F8.
Models (one global IF on user-relative inputs, as X4; a ranked against the user's own previous 30 scored days):
  IF Relatif Personal     alarm if a_cal >= theta                       (= X2)
  Model Usulan Personal   R = .5 a_cal + .4 c + .1 h, lambda 0.3, alarm if R >= theta   (= X4)
Choices on seeds 0+2 only (Test 1), rule: lowest FPR with TPR >= 0.65 (else highest TPR):
  theta 0.40..0.99 per (model, W); then W is locked per model by the same rule.
Reported on seed 1 for Test 1 (uid swap) and Test 2 (session hijack, 60 sessions), for every W.
  Output: results/x4_window.json
  Run   : python -W ignore tools/x4_window.py <landauer_runs_dir>      (~30 min)
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

A_RUNS = sys.argv[1]
WS, VAL, TEST, DAYS = [7, 14, 25], [0, 2], 1, 1910
W_W, LAM = (0.5, 0.4, 0.1), 0.3
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
MODELS = ['IF Relatif Personal', 'Model Usulan Personal']
OUTF = os.path.join(config.RESULTS, 'x4_window.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}


def prepare(recs, W):
    f = lib.compute_features(recs, W=W)
    X = np.nan_to_num(np.hstack([lib.relative_A(f, W=W), f['F']]).astype(np.float32))
    a_cal = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    _, c = lib.flags_c(f['F'])
    return {'f': f, 'avail': ~np.isnan(a_cal) & ~f['first'],
            'IF Relatif Personal': np.nan_to_num(a_cal, nan=-1),
            'Model Usulan Personal': np.nan_to_num(lib.risk(f, a_cal, c, W_W, LAM), nan=-1)}


def run(d, lab, model, th):
    return lib.evaluate(d['f'], d[model] >= th, d['avail'], lab, r=1)[2]


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


def pick(cands):
    ok = [x for x in cands if x[1]['TPR'] >= 0.65]
    return min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(cands, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))


def save(o):
    json.dump(o, open(OUTF, 'w'), indent=1, default=str)


OUT = {'meta': __doc__.strip(), 'validation': {}, 'test1_seed1': {}, 'test2_seed1': {}}
THETA = {}
# ---------------------------------------------------------------- 1. validation per W (seeds 0+2)
for W in WS:
    V = {}
    for s in VAL:
        t = time.time()
        pairs, ct = lib.seed_swaps(config.STATS, s)
        V[s] = (prepare(lib.relabel(BASE, pairs, ct), W), lib.labels_from_swaps(pairs, ct))
        log(f'W={W} validation seed {s}: {time.time() - t:.0f}s')
    for model in MODELS:
        cands = []
        for th in TH:
            m = dict(TP=0, FN=0, FP=0, TN=0)
            for s in VAL:
                r = run(*V[s], model, th)
                for k in m:
                    m[k] += r[k]
            m['TPR'], m['FPR'] = m['TP'] / (m['TP'] + m['FN']), m['FP'] / (m['FP'] + m['TN'])
            m['fp_per_day'] = m['FP'] / (len(VAL) * DAYS)
            cands.append((th, m))
        th, m = pick(cands)
        THETA[(model, W)] = (th, m)
        OUT['validation'][f'{model} W={W}'] = {'theta': th, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
        log(f'W={W} {model}: theta={th} val TP={m["TP"]}/40 TPR={m["TPR"]:.3f} FPR={m["FPR"]:.4f}')
    del V
LOCK = {model: pick([((W, THETA[(model, W)][0]), THETA[(model, W)][1]) for W in WS])[0] for model in MODELS}
OUT['locked'] = {model: {'W': W, 'theta': th} for model, (W, th) in LOCK.items()}
log('LOCKED', OUT['locked'])
save(OUT)

# ---------------------------------------------------------------- 2. Test 1, seed 1 (every W)
L = json.load(open(os.path.join(A_RUNS, f'seed{TEST}', 'metrics.json')))
OUT['test1_seed1']['Landauer'] = {'TP': L['TP'], 'FP': L['FP'], 'FPR': round(L['FP'] / (L['FP'] + L['TN']), 4), 'alerts_per_day': round(L['FP'] / DAYS, 3)}
pairs, ct = lib.seed_swaps(config.STATS, TEST)
lab = lib.labels_from_swaps(pairs, ct)
for W in WS:
    d = prepare(lib.relabel(BASE, pairs, ct), W)
    for model in MODELS:
        OUT['test1_seed1'][f'{model} W={W}'] = {'theta': THETA[(model, W)][0], **short(run(d, lab, model, THETA[(model, W)][0]))}
    del d
    save(OUT)
    log(f'test1 W={W}:', {k: (v['TP'], v['FPR']) for k, v in OUT['test1_seed1'].items() if k.endswith(f'W={W}')})

# ---------------------------------------------------------------- 3. Test 2, seed 1 (every W)
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


T2 = {f'{m} W={W}': {'TP': 0, 'per': {}, 'apd': []} for m in MODELS for W in WS}
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
    for W in WS:
        d = prepare(recs, W)
        for model in MODELS:
            k = f'{model} W={W}'
            m = run(d, labels, model, THETA[(model, W)][0])
            T2[k]['TP'] += m['TP']; T2[k]['per'][level] = m['TP']; T2[k]['apd'].append(m['alerts_per_day'])
        del d
    del recs
    log(f'test2 {level}: {time.time() - t:.0f}s')
OUT['test2_seed1'] = {k: {'TP_of_60': v['TP'], 'per': v['per'], 'alerts_per_day': round(float(np.mean(v['apd'])), 3)} for k, v in T2.items()}
save(OUT)
log('test2:', {k: v['TP_of_60'] for k, v in OUT['test2_seed1'].items()})
log('DONE')
