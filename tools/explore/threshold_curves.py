"""Threshold curves of every model on seed 1, for the interactive model-comparison page.

For each test and model: attacks detected and false alarms at many thresholds, so a reader can
move a threshold and see the trade-off. Thresholds marked "official" are the ones chosen on the
validation seeds (0 and 2) and reported in the thesis; choosing a threshold from these seed-1
curves is exploration only.
  Test 1 (uid swap, per day)       : A, B, B+RS, C, D, Hybrid, X1-X4, X4 with ECOD/COPOD/HBOS/LOF
  Test 2 (session hijack, per day) : A, Hybrid, D, X4 (3 intensities x 20 sessions)
  Test 3/3b (per 15/30/60-min window, 174 users): A-win, X4-win, X5-base, X5-acc, X5-full
  Output: results/threshold_curves.json
  Run   : python -W ignore tools/threshold_curves.py      (~1.5 h)
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
from lib_k3_window import GlobalIF, calib_risk, inject as inject_w, load_windows, user_features, user_features_v2  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

SEED, DAYS = 1, 1910
W_RISK, LAM = (0.5, 0.4, 0.1), 0.3
OFFICIAL_DAILY = {'A': 0.6, 'B': 0.52, 'B_RS': 0.50, 'C': 0.80, 'D': 0.55, 'Hybrid': 0.40, 'X1': 0.87, 'X2': 0.83,
                  'X3': 0.61, 'X4': 0.60, 'X4_ECOD': 0.58, 'X4_COPOD': 0.57, 'X4_HBOS': 0.62, 'X4_LOF': 0.51}
OFFICIAL_WIN = {15: {'A_win': 0.999, 'X4_win': 0.74},
                30: {'A_win': 0.93, 'X4_win': 0.73, 'X5_base': 0.74, 'X5_acc': 1.92, 'X5_full': 1.8},
                60: {'A_win': 0.89, 'X4_win': 0.72, 'X5_base': 0.73, 'X5_acc': 1.295, 'X5_full': 1.257}}
GRID_DAY = sorted({round(x, 3) for x in np.arange(0.30, 0.985, 0.02)} | {0.99, 0.995, 0.999} | set(OFFICIAL_DAILY.values()))
GRID_WIN = [round(x, 3) for x in np.arange(0.30, 0.995, 0.01)] + [0.995, 0.999]
T0 = time.time()
OUT = {'official_daily': OFFICIAL_DAILY, 'official_window': {str(k): v for k, v in OFFICIAL_WIN.items()}}


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def point(th, m):
    return {'th': th, 'TP': m['TP'], 'FPR': round(m['FPR'], 5), 'apd': round(m['alerts_per_day'], 3), 'TP_adj': m['TP_adj']}


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}


def daily_scores(f, detectors):
    """Score arrays (and availability) of the IF-based daily models on one feature table."""
    _, c = lib.flags_c(f['F'])
    first = f['first']
    aB = lib.iforest_a(f, lib.design_matrix(f, 'A'))
    aC = lib.iforest_a(f, lib.design_matrix(f, 'AF'))
    Xr = np.hstack([lib.relative_A(f, W=14), f['F']])
    aR = lib.iforest_a(f, Xr.astype(np.float32))
    aRc = lib.calibrate_per_user(f, aR)
    av = lambda a: ~np.isnan(a) & ~first
    risk = lambda a: lib.risk(f, a, c, W_RISK, LAM)
    S = {'B': (aB, av(aB)), 'B_RS': (risk(aB), av(aB)), 'C': (aC, av(aC)), 'D': (risk(aC), av(aC)),
         'X1': (aR, av(aR)), 'X2': (aRc, av(aRc)), 'X3': (risk(aR), av(aR)), 'X4': (risk(aRc), av(aRc))}
    for det in detectors:
        a = lib.calibrate_per_user(f, lib.rolling_a(f, Xr, lib.make_detector(det)))
        S[f'X4_{det}'] = (risk(a), av(a))
        log('  detector', det)
    return S, c


def daily_curves(f, labels, models, detectors=()):
    S, c = daily_scores(f, detectors)
    out = {}
    for name in models:
        if name in ('A', 'Hybrid'):
            w = (1, 0, 0) if name == 'A' else (0.5, 0.3, 0.2)
            out[name] = [point(th, lib.hybrid_detect(f, c, w=w, lam=LAM, theta=th, labels=labels)[2]) for th in GRID_DAY]
        else:
            s, avail = S[name]
            out[name] = [point(th, lib.evaluate(f, np.nan_to_num(s, nan=-1) >= th, avail, labels)[2]) for th in GRID_DAY]
        log('  curve', name)
    return out


# ------------------------------------------------------------------ Test 1: uid swap
pairs, ct = lib.seed_swaps(config.STATS, SEED)
labels = lib.labels_from_swaps(pairs, ct)
f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=14, M=5)
OUT['test1'] = daily_curves(f, labels, list(OFFICIAL_DAILY), detectors=('ECOD', 'COPOD', 'HBOS', 'LOF'))
log('test 1 done')

# ------------------------------------------------------------------ Test 2: session hijack per day
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


OUT['test2'] = {}
for lv in k3.LEVELS:
    recs = dict(CLEAN)
    for p in PLANS:
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject_day(lst[i], EVENTS[lv][p['victim']])
        recs[p['victim']] = lst
    lab = {p['victim']: p['day'] for p in PLANS}
    f2 = lib.compute_features(recs, W=14)
    OUT['test2'][lv] = daily_curves(f2, lab, ['A', 'Hybrid', 'D', 'X4'])
    log('test 2', lv, 'done')


# ------------------------------------------------------------------ Tests 3 / 3b: per window
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


OUT['test3'] = {}
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
    views = {}
    for key in CL:
        for lv in k3.LEVELS:
            V = dict(CL[key]); V.update(runs[lv][key])
            WS = dict(WS0); WS.update(runs[lv]['WS'])
            views[(key, lv)] = (V, WS)
    models = {'A_win': ('A', True, 1, 0), 'X4_win': ('X4', False, 1, 0)}
    if W != 15:
        models.update({'X5_base': ('X5', False, 1, 0), 'X5_acc': ('X5acc', False, K, 0), 'X5_full': ('X5acc', False, K, SUP)})
    res = {}
    for name, (key, strict, KK, sup) in models.items():
        grid = GRID_WIN if KK == 1 else [round(x, 3) for x in np.linspace(0.30, float(K), 136)]
        grid = sorted(set(grid) | {OFFICIAL_WIN[W][name], 0.6 if KK == 1 else 1.8})
        pts = []
        for th in grid:
            per, lats, fas = {}, [], []
            for lv in k3.LEVELS:
                V, WS = views[(key, lv)]
                d, fa = evaluate(V, WS, W, th, strict, KK, sup)
                per[lv] = sum(x is not None for x in d.values()); lats += [x for x in d.values() if x is not None]; fas.append(fa)
            pts.append({'th': th, 'det': sum(per.values()), 'per': per, 'fa': round(float(np.mean(fas)), 3),
                        'lat': round(float(np.median(lats)), 1) if lats else None})
        res[name] = pts
        o = next(p for p in pts if p['th'] == OFFICIAL_WIN[W][name])
        log(f'W={W} {name}: official th {o["th"]} -> {o["det"]}/60, fa {o["fa"]}')
    OUT['test3'][str(W)] = res
    del WINS, CL, runs, views

os.makedirs(config.RESULTS, exist_ok=True)
json.dump(OUT, open(os.path.join(config.RESULTS, 'threshold_curves.json'), 'w'), separators=(',', ':'))
log('done')
