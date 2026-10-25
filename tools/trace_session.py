"""Trace every K3 hijack session of seed 1 (20 victims x 3 intensities = 60) through every model.

For the interactive simulator: writes, per session, every intermediate number of
  * per day : model A (Landauer distance, nearest known day), Hybrid (Landauer distance
              + c + h), model D (a, F1-F8, flags k1-k5, c, h, R) and model X4 (relative
              volume, a_cal, h, R), each with its status (detection / training);
  * per 30-min window: A-win (Landauer distance), X4-win (c1-c4, a, a_cal, h, R) and
              X5 (c1-c4, a, a_cal, h, R, sum over K = 3 windows, 4-hour suppression).
Thresholds are the locked ones of the thesis (A 0.6, Hybrid 0.4, D 0.55, X4 0.60,
A-win 0.93, X4-win 0.73, X5-full 1.8 with K = 3 and suppression 8 windows).
  Output: results/trace_sessions.json
  Run   : python -W ignore tools/trace_session.py      (~15 min)
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
from lib_k3_window import GlobalIF, inject as inject_w, load_windows, user_features, user_features_v2  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

SEED, W, K, SUP = 1, 30, 3, 8
TH = dict(A=0.6, Hybrid=0.4, D=0.55, X4=0.60, A_win=0.93, X4_win=0.73, X5=1.8)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(SEED, BASE, W30, ELIG)
CLEAN = lib.relabel(BASE, {}, {})


def inject_day(rec, events):
    """Same as step09's inject: add attack events to the victim's day summary."""
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


def vec(x):
    return {TYPES[j]: int(x[j]) for j in np.flatnonzero(x)}


def nearest_known(A, s, t, r=1, queue=30):
    """Model A's memory for one user up to row t: the nearest known day and the memory size."""
    Q, wait = [], r
    for i in range(s, t + 1):
        x = A[i]
        a, mi = None, None
        if Q:
            M = np.vstack(Q)
            dist = np.abs(M - x).sum(1) / np.maximum(M, x).sum(1)
            mi = int(np.argmin(dist)); a = float(dist[mi])
        if i == t:
            return (vec(Q[mi]) if mi is not None else None), len(Q)
        if a is not None and a > TH['A']:
            wait = r
        else:
            wait -= 1
            if mi is not None:
                Q.append(Q.pop(mi))
        if wait >= 0:
            Q.append(x)
        if len(Q) > queue:
            Q.pop(0)


def risk_parts(a, c, first):
    """lib.risk with (0.5, 0.4, 0.1), lambda 0.3, also returning h."""
    R, H, h = np.full(len(a), np.nan), np.full(len(a), np.nan), 0.0
    for i in range(len(a)):
        if first[i]:
            h = 0.0
            continue
        if np.isnan(a[i]):
            continue
        H[i] = h
        R[i] = 0.5 * a[i] + 0.4 * c[i] + 0.1 * h
        h = 0.3 * h + 0.7 * R[i]
    return R, H


def calib_parts(a, c, w=(0.5, 0.4, 0.1)):
    """lib_k3_window.calib_risk, also returning a_cal and h."""
    a_cal, hist, h = a.copy(), [], 0.0
    R, H = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for i in range(len(a)):
        if np.isnan(a[i]):
            continue
        if len(hist) >= 5:
            q = np.asarray(hist[-30:])
            a_cal[i] = (np.sum(q < a[i]) + 0.5 * np.sum(q == a[i])) / len(q)
        hist.append(a[i])
        H[i] = h
        R[i] = w[0] * a_cal[i] + w[1] * c[i] + w[2] * h
        h = 0.3 * h + 0.7 * R[i]
    return a_cal, H, R


# ------------------------------------------------------------------ window models: trained once on clean data
WINS = load_windows(ELIG, W, TYPES)
users = sorted(WINS)
rows_t = np.concatenate([[x.w for x in WINS[u]] for u in users])
X = np.vstack([user_features(WINS[u])[0] for u in users]); g4 = GlobalIF(rows_t, X, W); del X
X = np.vstack([user_features_v2(WINS[u])[0] for u in users]); g5 = GlobalIF(rows_t, X, W); del X
log('window IFs trained')

out = dict(seed=SEED, window_min=W, K=K, suppress_windows=SUP, thresholds=TH, types=TYPES, sessions=[])
for level in k3.LEVELS:
    events = {p['victim']: k3.attack_events(p, level, SEED, BASE, ALL_PREFIX) for p in PLANS}
    labels = {p['victim']: p['day'] for p in PLANS}
    # ---------------------------------------------------------- per day
    recs = dict(CLEAN)
    for p in PLANS:
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject_day(lst[i], events[p['victim']])
        recs[p['victim']] = lst
    f = lib.compute_features(recs, W=14)
    ks, c = lib.flags_c(f['F'])
    stA, dcA, _, aL, _ = lib.hybrid_detect(f, c, w=(1, 0, 0), theta=TH['A'], labels=labels)
    stH, dcH, _, aH, RH = lib.hybrid_detect(f, c, w=(0.5, 0.3, 0.2), lam=0.3, theta=TH['Hybrid'], labels=labels)
    aD = lib.iforest_a(f, lib.design_matrix(f, 'AF'))
    RD, HD = risk_parts(aD, c, f['first'])
    stD, dcD, _ = lib.evaluate(f, np.nan_to_num(RD, nan=-1) >= TH['D'], ~np.isnan(aD) & ~f['first'], labels)
    Xr = np.hstack([lib.relative_A(f, W=14), f['F']]).astype(np.float32)
    a4raw = lib.iforest_a(f, Xr)
    a4 = lib.calibrate_per_user(f, a4raw)
    R4, H4 = risk_parts(a4, c, f['first'])
    st4, dc4, _ = lib.evaluate(f, np.nan_to_num(R4, nan=-1) >= TH['X4'], ~np.isnan(a4) & ~f['first'], labels)
    log(level, 'daily models done')
    starts = np.flatnonzero(f['first'])
    bounds = {f['uid'][s]: s for s in starts}
    idx = {(u, d): i for i, (u, d) in enumerate(zip(f['uid'], f['d']))}
    Afl = f['A'].astype(np.float64)
    for p in PLANS:
        v, ev = p['victim'], events[p['victim']]
        t, s = idx[(v, p['day'])], bounds[v]
        prev = [x for x in recs[v] if x.d < p['day']][-14:]
        tot_prev = [x.total for x in prev]
        lt_prev = [math.log1p(x) for x in tot_prev]
        med_lt = float(np.median(lt_prev)); mad_lt = float(np.median(np.abs(np.asarray(lt_prev) - med_lt)))
        today = recs[v][t - s]
        seen = set()
        for x in prev:
            seen.update(x.pref)
        nearest, nknown = nearest_known(Afl, s, t)
        hH = (RH[t] - 0.5 * aH[t] - 0.3 * c[t]) / 0.2 if not np.isnan(RH[t]) else None
        day = dict(
            totals=dict(today=int(today.total), median_prev14=float(np.median(tot_prev))),
            today=vec(Afl[t]), nearest_day=nearest, known_days=nknown,
            F1_inputs=dict(log_today=r4(math.log1p(today.total)), median_log=r4(med_lt), mad_log=r4(mad_lt)),
            F6_inputs=dict(path_events=int(sum(today.pref.values())),
                           in_new_folders=int(sum(cn for pf, cn in today.pref.items() if pf not in seen))),
            F=[r4(x) for x in f['F'][t]], flags={k: bool(ks[k][t]) for k in ks}, c=r4(c[t]),
            A=dict(distance=r4(aL[t]), status=stA[t], decision=dcA[t]),
            Hybrid=dict(distance=r4(aH[t]), h=r4(hH), R=r4(RH[t]), status=stH[t], decision=dcH[t]),
            D=dict(a=r4(aD[t]), h=r4(HD[t]), R=r4(RD[t]), status=stD[t], decision=dcD[t]),
            X4=dict(rel_volume=r4(math.log1p(today.total) - math.log1p(float(np.median(tot_prev)))),
                    a_raw=r4(a4raw[t]), a_cal=r4(a4[t]), h=r4(H4[t]), R=r4(R4[t]), status=st4[t], decision=dc4[t]),
            prev_day=dict(total=int(prev[-1].total), D_R=r4(RD[t - 1]), X4_R=r4(R4[t - 1])),
        )
        # ------------------------------------------------------ per 30-min window
        wl = inject_w(WINS[v], ev, W, TIX)
        ws = np.array([x.w for x in wl])
        X4v, c4v, aLw = user_features(wl)
        X5v, c5v = user_features_v2(wl)
        acal4, H4w, R4w = calib_parts(g4.score(ws, X4v), c4v)
        a5 = g5.score(ws, X5v)
        acal5, H5w, R5w = calib_parts(a5, c5v)
        a4w = g4.score(ws, X4v)
        acc, j, run = np.full(len(R5w), np.nan), 0, 0.0
        vv = np.where(np.isnan(R5w), 0.0, R5w)
        for i in range(len(vv)):
            run += vv[i]
            while ws[j] <= ws[i] - K:
                run -= vv[j]; j += 1
            acc[i] = run
        w0 = int(p['start'] // (60 * W)); w1 = int((p['start'] + p['dur'] - 1e-9) // (60 * W))
        inj = {}
        for ts_, typ, pf in ev:
            inj[int(ts_ // (60 * W))] = inj.get(int(ts_ // (60 * W)), 0) + 1
        # X5 alarms over the user's whole stream (suppression needs the history before the session)
        last, x5_alarm = None, np.zeros(len(ws), bool)
        for i in range(len(ws)):
            if not np.isnan(R5w[i]) and acc[i] >= TH['X5'] and (last is None or ws[i] - last >= SUP):
                x5_alarm[i] = True; last = ws[i]
        sel = np.flatnonzero((ws >= w0 - 3) & (ws <= w1 + K + 1))
        rows = [dict(w=int(ws[i]), total=int(wl[i].total), injected=inj.get(int(ws[i]), 0),
                     A_win=dict(distance=r4(aLw[i])),
                     X4_win=dict(c1_c4=[r4(x) for x in X4v[i, -4:]], c=r4(c4v[i]), a=r4(a4w[i]), a_cal=r4(acal4[i]),
                                 h=r4(H4w[i]), R=r4(R4w[i])),
                     X5=dict(c1_c4=[r4(x) for x in X5v[i, -4:]], c=r4(c5v[i]), a=r4(a5[i]), a_cal=r4(acal5[i]),
                             h=r4(H5w[i]), R=r4(R5w[i]), acc=r4(acc[i]), alarm=bool(x5_alarm[i])))
                for i in sel]
        in_att = (ws >= w0) & (ws <= w1)
        in_att_k = (ws >= w0) & (ws <= w1 + K - 1)

        def first_lat(mask):
            hit = np.flatnonzero(mask)
            return None if len(hit) == 0 else round(((ws[hit[0]] + 1) * 60 * W - p['start']) / 60, 1)
        window = dict(w_start=w0, w_end=w1, rows=rows,
                      latency_min=dict(A_win=first_lat(in_att & (np.nan_to_num(aLw, nan=-1) > TH['A_win'])),
                                       X4_win=first_lat(in_att & (np.nan_to_num(R4w, nan=-1) >= TH['X4_win'])),
                                       X5=first_lat(in_att_k & x5_alarm)))
        det = dict(A=day['A']['status'] == 'detection' and day['A']['decision'] == 'anomalous',
                   Hybrid=day['Hybrid']['status'] == 'detection' and day['Hybrid']['decision'] == 'anomalous',
                   D=day['D']['status'] == 'detection' and day['D']['decision'] == 'anomalous',
                   X4=day['X4']['status'] == 'detection' and day['X4']['decision'] == 'anomalous',
                   A_win=window['latency_min']['A_win'] is not None, X4_win=window['latency_min']['X4_win'] is not None,
                   X5=window['latency_min']['X5'] is not None)
        out['sessions'].append(dict(level=level, victim=v, day=p['day'], start=p['start'], dur_min=round(p['dur'] / 60, 1),
                                    med30=p['med30'], attack_events={t_: sum(1 for e in ev if e[1] == t_) for t_ in sorted({e[1] for e in ev})},
                                    detected=det, daily=day, window=window))
    log(level, {m: sum(s['detected'][m] for s in out['sessions'] if s['level'] == level) for m in out['sessions'][-1]['detected']})

os.makedirs(config.RESULTS, exist_ok=True)
json.dump(out, open(os.path.join(config.RESULTS, 'trace_sessions.json'), 'w'), separators=(',', ':'), default=float)
log('done')
