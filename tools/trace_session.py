"""Trace one real K3 hijack session (seed 1) through every model, step by step.

For the slides "simulasi perhitungan": picks, at one intensity, a victim that model A
misses and the proposed models catch, and writes every intermediate number:
  * per day : model A (Landauer distance, nearest known day), model D (a, F1-F8,
              flags k1-k5, c, h, R) and model X4 (relative volume, a_cal, R);
  * per 30-min window: A-win distance and X5 (c1-c4, a, a_cal, c, h, R, sum over
              K = 3 windows, alarm with 4-hour suppression).
Thresholds are the locked ones of the thesis (D 0.55, X4 0.60, A 0.6, A-win 0.93, X5-full 1.8).
  Output: results/trace_session.json
  Run   : python -W ignore tools/trace_session.py [level]     (level: rendah | sedang | tinggi)
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

SEED, LEVEL = 1, (sys.argv[1] if len(sys.argv) > 1 else 'sedang')
TH_A, TH_D, TH_X4, TH_AW, TH_X5, K, SUP, W = 0.6, 0.55, 0.60, 0.93, 1.8, 3, 8, 30
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(SEED, BASE, W30, ELIG)
EVENTS = {p['victim']: k3.attack_events(p, LEVEL, SEED, BASE, ALL_PREFIX) for p in PLANS}
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


def landauer_trace(A, s, e, t, r=1, queue=30):
    """Model A for one user (rows s..e-1) up to row t; returns the decision at t and the nearest known day."""
    Q, wait = [], r
    for i in range(s, t + 1):
        x = A[i]
        a, mi = None, None
        if Q:
            M = np.vstack(Q)
            dist = np.abs(M - x).sum(1) / np.maximum(M, x).sum(1)
            mi = int(np.argmin(dist)); a = float(dist[mi])
        status = 'detection' if wait <= 0 else 'training'
        if i == t:
            return dict(distance=a, status=status, anomalous=a is not None and a > TH_A, known_days=len(Q),
                        nearest_day=vec(Q[mi]) if mi is not None else None, today=vec(x))
        if a is not None and a > TH_A:
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


# ------------------------------------------------------------------ per day: A, D, X4
recs = dict(CLEAN)
for p in PLANS:
    lst = list(recs[p['victim']])
    i = next(j for j, r in enumerate(lst) if r.d == p['day'])
    lst[i] = inject_day(lst[i], EVENTS[p['victim']])
    recs[p['victim']] = lst
f = lib.compute_features(recs, W=14)
ks, c = lib.flags_c(f['F'])
aD = lib.iforest_a(f, lib.design_matrix(f, 'AF'))
RD, HD = risk_parts(aD, c, f['first'])
X = np.hstack([lib.relative_A(f, W=14), f['F']]).astype(np.float32)
a4raw = lib.iforest_a(f, X)
a4 = lib.calibrate_per_user(f, a4raw)
R4, H4 = risk_parts(a4, c, f['first'])
log('daily models done')
starts = np.flatnonzero(f['first'])
bounds = {f['uid'][s]: (s, e) for s, e in zip(starts, list(starts[1:]) + [len(f['uid'])])}
idx = {(u, d): i for i, (u, d) in enumerate(zip(f['uid'], f['d']))}
Afl = f['A'].astype(np.float64)

day_out = {}
for p in PLANS:
    v = p['victim']
    t = idx[(v, p['day'])]
    s, e = bounds[v]
    prev = [x for x in recs[v] if x.d < p['day']][-14:]
    tot_prev = [x.total for x in prev]
    lt_prev = [math.log1p(x) for x in tot_prev]
    med_lt = float(np.median(lt_prev)); mad_lt = float(np.median(np.abs(np.asarray(lt_prev) - med_lt)))
    today = recs[v][t - s]
    seen = set()
    for x in prev:
        seen.update(x.pref)
    novel = sum(cn for pf, cn in today.pref.items() if pf not in seen)
    day_out[v] = dict(
        plan=dict(day=p['day'], start=p['start'], dur_min=p['dur'] / 60, med30=p['med30']),
        attack_events={typ: sum(1 for ev in EVENTS[v] if ev[1] == typ) for typ in {ev[1] for ev in EVENTS[v]}},
        A=landauer_trace(Afl, s, e, t),
        totals=dict(today=int(today.total), median_prev14=float(np.median(tot_prev)), prev_day=int(prev[-1].total)),
        F1_inputs=dict(log_today=math.log1p(today.total), median_log=med_lt, mad_log=mad_lt),
        F6_inputs=dict(path_events=int(sum(today.pref.values())), in_new_folders=int(novel)),
        F7_inputs=dict(sensitive=int(today.sens), total=int(today.total)),
        F=[None if np.isnan(x) else float(x) for x in f['F'][t]],
        flags={k: bool(ks[k][t]) for k in ks}, c=float(c[t]),
        D=dict(a=float(aD[t]), h=float(HD[t]), R=float(RD[t]), alarm=bool(RD[t] >= TH_D)),
        D_prev_day=dict(a=float(aD[t - 1]), c=float(c[t - 1]), R=float(RD[t - 1])),
        X4=dict(rel_volume=math.log1p(today.total) - math.log1p(float(np.median(tot_prev))),
                a_raw=float(a4raw[t]), a_cal=float(a4[t]), h=float(H4[t]), R=float(R4[t]), alarm=bool(R4[t] >= TH_X4)),
    )
log('daily traces done')

# ------------------------------------------------------------------ per 30-min window: A-win, X5
WINS = load_windows(ELIG, W, TYPES)
users = sorted(WINS)
F5 = {u: user_features_v2(WINS[u]) for u in users}
rows_t = np.concatenate([[x.w for x in WINS[u]] for u in users])
X5 = np.vstack([F5[u][0] for u in users])
g5 = GlobalIF(rows_t, X5, W)
del X5, F5
log('window IF trained')

win_out = {}
for p in PLANS:
    v = p['victim']
    wl = inject_w(WINS[v], EVENTS[v], W, TIX)
    ws = np.array([x.w for x in wl])
    Xv, cv = user_features_v2(wl)
    _, _, aL = user_features(wl)
    a = g5.score(ws, Xv)
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
        R[i] = 0.5 * a_cal[i] + 0.4 * cv[i] + 0.1 * h
        h = 0.3 * h + 0.7 * R[i]
    acc, j, run = np.full(len(R), np.nan), 0, 0.0
    vv = np.where(np.isnan(R), 0.0, R)
    for i in range(len(vv)):
        run += vv[i]
        while ws[j] <= ws[i] - K:
            run -= vv[j]; j += 1
        acc[i] = run
    w0 = int(p['start'] // (60 * W)); w1 = int((p['start'] + p['dur'] - 1e-9) // (60 * W))
    inj = {}
    for ts, typ, pf in EVENTS[v]:
        inj[int(ts // (60 * W))] = inj.get(int(ts // (60 * W)), 0) + 1
    rows, last_alarm, first_hit = [], None, None
    for i in np.flatnonzero((ws >= w0 - 4) & (ws <= w1 + K + 1)):
        alarm = bool(acc[i] >= TH_X5) and (last_alarm is None or ws[i] - last_alarm >= SUP)
        if alarm:
            last_alarm = ws[i]
            if first_hit is None and w0 <= ws[i] <= w1 + K - 1:
                first_hit = int(ws[i])
        rows.append(dict(w=int(ws[i]), total=int(wl[i].total), injected=inj.get(int(ws[i]), 0),
                         c1_c4=[None if np.isnan(x) else float(x) for x in Xv[i, -4:]], c=float(cv[i]),
                         a=None if np.isnan(a[i]) else float(a[i]), a_cal=None if np.isnan(a_cal[i]) else float(a_cal[i]),
                         h=None if np.isnan(H[i]) else float(H[i]), R=None if np.isnan(R[i]) else float(R[i]),
                         acc=float(acc[i]), alarm_X5=alarm,
                         A_win_distance=None if np.isnan(aL[i]) else float(aL[i]), alarm_A_win=bool(aL[i] > TH_AW)))
    win_out[v] = dict(w_start=w0, w_end=w1, rows=rows,
                      X5_latency_min=None if first_hit is None else ((first_hit + 1) * 60 * W - p['start']) / 60)
log('window traces done')

summary = {v: dict(A=day_out[v]['A']['anomalous'] and day_out[v]['A']['status'] == 'detection',
                   D=day_out[v]['D']['alarm'], X4=day_out[v]['X4']['alarm'],
                   A_win=any(r['alarm_A_win'] and win_out[v]['w_start'] <= r['w'] <= win_out[v]['w_end'] for r in win_out[v]['rows']),
                   X5=win_out[v]['X5_latency_min'] is not None, total_today=day_out[v]['totals']['today'])
           for v in day_out}
out = dict(seed=SEED, level=LEVEL, thresholds=dict(A=TH_A, D=TH_D, X4=TH_X4, A_win=TH_AW, X5_full=TH_X5, K=K, suppress=SUP),
           summary=summary, day=day_out, window=win_out)
os.makedirs(config.RESULTS, exist_ok=True)
json.dump(out, open(os.path.join(config.RESULTS, 'trace_session.json'), 'w'), indent=1, default=float)
for v, s_ in summary.items():
    log(v, s_)
log('done')
