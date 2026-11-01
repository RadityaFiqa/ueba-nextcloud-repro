"""Additional analysis: every model with ONE uniform threshold instead of a threshold tuned per model.

Single-score models alarm at 0.6 (Landauer distance, IF score a, risk R); the accumulative
window models (sum of R over K windows) alarm at 1.8. Everything else (features, weights,
lambda = 0.3, history W = 14, retraining mode, attack sessions) is the locked configuration.
Data: seed 1 (the reported run), all four tests.
  Sanity: each test also reruns the tuned threshold of one model, which must match expected_results.
  Output: results/uniform_threshold.json
  Run   : python -W ignore tools/uniform_threshold.py      (~45 min)
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from lib_k3_window import GlobalIF, calib_risk, inject as inject_w, load_windows, user_features, user_features_v2  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

SEED, TH, TH_ACC, DAYS = 1, 0.6, 1.8, 1910
W_RISK, LAM = (0.5, 0.4, 0.1), 0.3
T0 = time.time()
OUT = {'thresholds': {'single_score': TH, 'accumulative': TH_ACC}}


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def pick(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FN', 'FP', 'TN', 'TPR', 'FPR', 'R', 'alerts_per_day')}


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}


def daily_models(f, labels, th):
    """Daily models on one feature table at one threshold (Landauer keeps its own 0.6)."""
    _, c = lib.flags_c(f['F'])
    first = f['first']
    aB = lib.iforest_a(f, lib.design_matrix(f, 'A'))
    aC = lib.iforest_a(f, lib.design_matrix(f, 'AF'))
    Xr = np.hstack([lib.relative_A(f, W=14), f['F']]).astype(np.float32)
    a4 = lib.calibrate_per_user(f, lib.iforest_a(f, Xr))
    avB, avC, av4 = ~np.isnan(aB) & ~first, ~np.isnan(aC) & ~first, ~np.isnan(a4) & ~first
    ev = lambda s, av, t: lib.evaluate(f, np.nan_to_num(s, nan=-1) >= t, av, labels)[2]
    scores = dict(B=(aB, avB), B_RS=(lib.risk(f, aB, c, W_RISK, LAM), avB), C=(aC, avC),
                  D=(lib.risk(f, aC, c, W_RISK, LAM), avC), X4=(lib.risk(f, a4, c, W_RISK, LAM), av4))
    return scores, c, ev


# ------------------------------------------------------------------ Test 1: uid swap (seed 1)
pairs, ct = lib.seed_swaps(config.STATS, SEED)
labels = lib.labels_from_swaps(pairs, ct)
f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=14, M=5)
scores, c, ev = daily_models(f, labels, TH)
t1 = {'A_landauer': pick(lib.hybrid_detect(f, c, w=(1, 0, 0), theta=TH, labels=labels)[2])}
for k, (s, av) in scores.items():
    t1[k] = pick(ev(s, av, TH))
t1['Hybrid'] = pick(lib.hybrid_detect(f, c, w=(0.5, 0.3, 0.2), lam=LAM, theta=TH, labels=labels)[2])
sanity = {'D@0.55 (expect TP 11, FPR 0.0988)': pick(ev(*scores['D'], 0.55)),
          'C@0.80 (expect TP 11, FPR 0.1279)': pick(ev(*scores['C'], 0.80)),
          'Hybrid@0.40 (expect TP 12, FPR ~0.061)': pick(lib.hybrid_detect(f, c, w=(0.5, 0.3, 0.2), lam=LAM, theta=0.40, labels=labels)[2])}
OUT['test1_uid_swap'] = {'uniform': t1, 'sanity_tuned': sanity}
log('test 1', {k: (v['TP'], v['FPR']) for k, v in t1.items()})
log('sanity 1', {k: (v['TP'], v['FPR']) for k, v in sanity.items()})

# ------------------------------------------------------------------ Test 2: session hijack per day (seed 1)
ELIG = k3.eligible_users()
W30C = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(SEED, BASE, W30C, ELIG)
EVENTS = {lv: {p['victim']: k3.attack_events(p, lv, SEED, BASE, ALL_PREFIX) for p in PLANS} for lv in k3.LEVELS}
CLEAN = lib.relabel(BASE, {}, {})


def inject_day(rec, events):
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


t2 = {}
for lv in k3.LEVELS:
    recs = dict(CLEAN)
    for p in PLANS:
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject_day(lst[i], EVENTS[lv][p['victim']])
        recs[p['victim']] = lst
    lab = {p['victim']: p['day'] for p in PLANS}
    f2 = lib.compute_features(recs, W=14)
    _, c2 = lib.flags_c(f2['F'])
    first = f2['first']
    aC = lib.iforest_a(f2, lib.design_matrix(f2, 'AF'))
    Xr = np.hstack([lib.relative_A(f2, W=14), f2['F']]).astype(np.float32)
    a4 = lib.calibrate_per_user(f2, lib.iforest_a(f2, Xr))
    RD, R4 = lib.risk(f2, aC, c2, W_RISK, LAM), lib.risk(f2, a4, c2, W_RISK, LAM)
    ev2 = lambda s, av, t: pick(lib.evaluate(f2, np.nan_to_num(s, nan=-1) >= t, av, lab)[2])
    t2[lv] = {'A_landauer': pick(lib.hybrid_detect(f2, c2, w=(1, 0, 0), theta=TH, labels=lab)[2]),
              'Hybrid': pick(lib.hybrid_detect(f2, c2, w=(0.5, 0.3, 0.2), lam=LAM, theta=TH, labels=lab)[2]),
              'D': ev2(RD, ~np.isnan(aC) & ~first, TH),
              'X4': ev2(R4, ~np.isnan(a4) & ~first, TH),
              'sanity D@0.55': ev2(RD, ~np.isnan(aC) & ~first, 0.55)}
    log('test 2', lv, {k: v['TP'] for k, v in t2[lv].items()})
OUT['test2_hijack_daily'] = t2


# ------------------------------------------------------------------ Tests 3 / 3b: per window (seed 1, 174 users)
def accumulate(ws, s, K):
    ok = ~np.isnan(s)
    if K <= 1:
        return s
    v = np.where(ok, s, 0.0)
    acc = np.empty(len(v))
    j, run = 0, 0.0
    for i in range(len(v)):
        run += v[i]
        while ws[j] <= ws[i] - K:
            run -= v[j]; j += 1
        acc[i] = run
    acc[~ok] = np.nan
    return acc


def emitted(ws, v, theta, strict, sup):
    ok = ~np.isnan(v)
    idx = np.flatnonzero(ok & ((v > theta) if strict else (v >= theta)))
    if sup <= 0 or len(idx) == 0:
        return idx
    out, last = [], None
    for i in idx:
        if last is None or ws[i] - last >= sup:
            out.append(i); last = ws[i]
    return np.asarray(out, dtype=int)


def evaluate(V, WS, W, theta, strict, K, sup):
    att = {p['victim']: (int(p['start'] // (60 * W)), int((p['start'] + p['dur'] - 1e-9) // (60 * W)), p['start']) for p in PLANS}
    fp, det = 0, {}
    for u, v in V.items():
        ws = WS[u]
        idx = emitted(ws, v, theta, strict, sup)
        if u in att:
            w0, w1, st = att[u]
            m = (ws[idx] >= w0) & (ws[idx] <= w1 + K - 1)
            fp += int(np.sum(~m))
            hits = idx[m]
            det[u] = (((ws[hits[0]] + 1) * 60 * W - st) / 60) if len(hits) else None
        else:
            fp += len(idx)
    return det, fp / DAYS


t3 = {}
for W in (15, 30, 60):
    K, SUP = {15: (1, 0), 30: (3, 8), 60: (2, 4)}[W]
    WINS = load_windows(ELIG, W, TYPES)
    users = sorted(WINS)
    rows_t = np.concatenate([[x.w for x in WINS[u]] for u in users])
    F4 = {u: user_features(WINS[u]) for u in users}
    X = np.vstack([F4[u][0] for u in users]); g4 = GlobalIF(rows_t, X, W); a4 = g4.score(rows_t, X); del X
    CL = {'A': {}, 'X4': {}}
    WS0, off = {}, 0
    for u in users:
        n = len(WINS[u]); WS0[u] = np.array([x.w for x in WINS[u]])
        CL['A'][u] = F4[u][2]; CL['X4'][u] = calib_risk(a4[off:off + n], F4[u][1]); off += n
    del F4
    if W != 15:
        F5 = {u: user_features_v2(WINS[u]) for u in users}
        X = np.vstack([F5[u][0] for u in users]); g5 = GlobalIF(rows_t, X, W); a5 = g5.score(rows_t, X); del X
        CL['X5'], off = {}, 0
        for u in users:
            n = len(WINS[u]); CL['X5'][u] = calib_risk(a5[off:off + n], F5[u][1]); off += n
        del F5
        CL['X5acc'] = {u: accumulate(WS0[u], CL['X5'][u], K) for u in users}
    log(f'W={W}: clean scores ready')
    runs = {}
    for lv in k3.LEVELS:
        ov = {key: {} for key in CL}; ov['WS'] = {}
        for v, ev_ in EVENTS[lv].items():
            wl = inject_w(WINS[v], ev_, W, TIX)
            ws = np.array([x.w for x in wl])
            X4v, c4v, aLv = user_features(wl)
            ov['A'][v] = aLv; ov['X4'][v] = calib_risk(g4.score(ws, X4v), c4v); ov['WS'][v] = ws
            if W != 15:
                X5v, c5v = user_features_v2(wl)
                ov['X5'][v] = calib_risk(g5.score(ws, X5v), c5v); ov['X5acc'][v] = accumulate(ws, ov['X5'][v], K)
        runs[lv] = ov
    models = {'A_win': ('A', True, 1, 0, TH), 'X4_win': ('X4', False, 1, 0, TH)}
    if W != 15:
        models.update({'X5_base': ('X5', False, 1, 0, TH), 'X5_acc': ('X5acc', False, K, 0, TH_ACC), 'X5_full': ('X5acc', False, K, SUP, TH_ACC)})
    tuned = {15: ('X4_win', 0.74), 30: ('X4_win', 0.73), 60: ('X4_win', 0.72)}[W]  # expect 35, 32, 27 of 60
    res = {}
    for name, (key, strict, KK, sup, th) in list(models.items()) + [('sanity_' + tuned[0] + f'@{tuned[1]}', models[tuned[0]][:4] + (tuned[1],))]:
        dets, lats, fas = 0, [], []
        for lv in k3.LEVELS:
            V = dict(CL[key]); V.update(runs[lv][key])
            WS = dict(WS0); WS.update(runs[lv]['WS'])
            d, fa = evaluate(V, WS, W, th, strict, KK, sup)
            dets += sum(x is not None for x in d.values()); lats += [x for x in d.values() if x is not None]; fas.append(fa)
        res[name] = {'threshold': th, 'detected_of_60': dets, 'latency_median_min': round(float(np.median(lats)), 1) if lats else None,
                     'false_alarms_per_day': round(float(np.mean(fas)), 2)}
    t3[f'W{W}'] = res
    log(f'W={W}', {k: (v['detected_of_60'], v['false_alarms_per_day']) for k, v in res.items()})
    del WINS, CL, runs
OUT['test3_windows'] = t3

os.makedirs(config.RESULTS, exist_ok=True)
json.dump(OUT, open(os.path.join(config.RESULTS, 'uniform_threshold.json'), 'w'), indent=1)
log('done')
