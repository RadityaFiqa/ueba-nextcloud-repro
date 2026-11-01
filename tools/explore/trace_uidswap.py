"""Trace the 20 uid-swap positives of seed 1 (Test 1) through every daily model, step by step.

For each swapped account on its first active day after the swap: the day's activity, Landauer's
nearest known day, features F1-F8 with their inputs, flags k1-k5, context score c, and for every
model its score (a, a_cal, h, R), status (detection / training) and decision at the official
threshold. Companion of tools/trace_session.py (which covers the 60 session hijacks).
  Output: results/trace_uidswap.json
  Run   : python -W ignore tools/trace_uidswap.py      (~10 min)
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

SEED = 1
W_RISK, LAM = (0.5, 0.4, 0.1), 0.3
TH = {'A': 0.6, 'B': 0.52, 'B_RS': 0.50, 'C': 0.80, 'D': 0.55, 'Hybrid': 0.40, 'X1': 0.87, 'X2': 0.83,
      'X3': 0.61, 'X4': 0.60, 'X4_ECOD': 0.58, 'X4_COPOD': 0.57, 'X4_HBOS': 0.62, 'X4_LOF': 0.51}
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def risk_parts(f, a, c):
    """lib.risk with the locked weights, also returning h."""
    R, H, h = np.full(len(a), np.nan), np.full(len(a), np.nan), 0.0
    for i in range(len(a)):
        if f['first'][i]:
            h = 0.0
            continue
        if np.isnan(a[i]):
            continue
        H[i] = h
        R[i] = W_RISK[0] * a[i] + W_RISK[1] * c[i] + W_RISK[2] * h
        h = LAM * h + (1 - LAM) * R[i]
    return R, H


BASE, TYPES = lib.load_base(config.BASE)
pairs, ct = lib.seed_swaps(config.STATS, SEED)
labels = lib.labels_from_swaps(pairs, ct)
recs = lib.relabel(BASE, pairs, ct)
f = lib.compute_features(recs, W=14, M=5)
ks, c = lib.flags_c(f['F'])
first = f['first']
Afl = f['A'].astype(np.float64)
log('features ready')

# ---- scores of every model
aB = lib.iforest_a(f, lib.design_matrix(f, 'A'))
aC = lib.iforest_a(f, lib.design_matrix(f, 'AF'))
Xr = np.hstack([lib.relative_A(f, W=14), f['F']])
aR = lib.iforest_a(f, Xr.astype(np.float32))
aRc = lib.calibrate_per_user(f, aR)
M = {}  # name -> dict(a=..., a_raw=..., R=..., H=..., score=...)
M['B'] = dict(a=aB, score=aB)
R_, H_ = risk_parts(f, aB, c); M['B_RS'] = dict(a=aB, R=R_, H=H_, score=R_)
M['C'] = dict(a=aC, score=aC)
R_, H_ = risk_parts(f, aC, c); M['D'] = dict(a=aC, R=R_, H=H_, score=R_)
M['X1'] = dict(a=aR, score=aR)
M['X2'] = dict(a_raw=aR, a=aRc, score=aRc)
R_, H_ = risk_parts(f, aR, c); M['X3'] = dict(a=aR, R=R_, H=H_, score=R_)
R_, H_ = risk_parts(f, aRc, c); M['X4'] = dict(a_raw=aR, a=aRc, R=R_, H=H_, score=R_)
for det in ('ECOD', 'COPOD', 'HBOS', 'LOF'):
    a_raw = lib.rolling_a(f, Xr, lib.make_detector(det))
    a_cal = lib.calibrate_per_user(f, a_raw)
    R_, H_ = risk_parts(f, a_cal, c)
    M['X4_' + det] = dict(a_raw=a_raw, a=a_cal, R=R_, H=H_, score=R_)
    log('detector', det)
DEC = {}
for name, d in M.items():
    avail = ~np.isnan(d['a']) & ~first
    st, dc, m = lib.evaluate(f, np.nan_to_num(d['score'], nan=-1) >= TH[name], avail, labels)
    DEC[name] = (st, dc, m['TP'])
stA, dcA, mA, aL, _ = lib.hybrid_detect(f, c, w=(1, 0, 0), theta=TH['A'], labels=labels)
DEC['A'] = (stA, dcA, mA['TP'])
stH, dcH, mH, aH, RH = lib.hybrid_detect(f, c, w=(0.5, 0.3, 0.2), lam=LAM, theta=TH['Hybrid'], labels=labels)
DEC['Hybrid'] = (stH, dcH, mH['TP'])
log('TP per model', {k: v[2] for k, v in DEC.items()})


def vec(x):
    return {TYPES[j]: int(x[j]) for j in np.flatnonzero(x)}


def nearest_known(s, t, r=1, queue=30):
    Q, wait = [], r
    for i in range(s, t + 1):
        x = Afl[i]
        a, mi = None, None
        if Q:
            Mq = np.vstack(Q)
            dist = np.abs(Mq - x).sum(1) / np.maximum(Mq, x).sum(1)
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


users = json.load(open(config.STATS))['user_info']
starts = np.flatnonzero(first)
bound = {f['uid'][s]: s for s in starts}
idx = {(u, d): i for i, (u, d) in enumerate(zip(f['uid'], f['d']))}
out = dict(seed=SEED, thresholds=TH, positives=[])
for victim, t_sw in ct.items():
    ident = pairs[victim]              # the account that now shows the other user's behaviour
    d0 = int(t_sw) // 86400
    t = idx.get((ident, d0))
    if t is None:
        continue
    s = bound[ident]
    ua, ub = users[victim], users[ident]
    ra, rb = ua['cnt'] / ua['days'], ub['cnt'] / ub['days']
    sim = round(0.3 * min(ra, rb) / max(ra, rb) + 0.7 * len(set(ua['actions']) & set(ub['actions'])) / len(set(ua['actions']) | set(ub['actions'])), 3)
    prev = [r for r in recs[ident] if r.d < d0][-14:]
    today = recs[ident][t - s]
    lt = [math.log1p(r.total) for r in prev]
    med = float(np.median(lt)); mad = float(np.median(np.abs(np.asarray(lt) - med)))
    seen = set()
    for r in prev:
        seen.update(r.pref)
    nearest, nknown = nearest_known(s, t)
    models = {}
    for name, d in M.items():
        st, dc, _ = DEC[name]
        models[name] = {k: r4(v[t]) for k, v in d.items() if k != 'score'}
        models[name].update(status=st[t], decision=dc[t])
    models['A'] = dict(distance=r4(aL[t]), status=stA[t], decision=dcA[t])
    hH = (RH[t] - 0.5 * aH[t] - 0.3 * c[t]) / 0.2 if not np.isnan(RH[t]) else None
    models['Hybrid'] = dict(distance=r4(aH[t]), h=r4(hH), R=r4(RH[t]), status=stH[t], decision=dcH[t])
    out['positives'].append(dict(
        account=ident, behaviour_from=victim, day=d0, similarity=sim,
        totals=dict(today=int(today.total), median_prev14=float(np.median([r.total for r in prev])) if prev else None),
        today=vec(Afl[t]), nearest_day=nearest, known_days=nknown,
        F1_inputs=dict(log_today=r4(math.log1p(today.total)), median_log=r4(med), mad_log=r4(mad)),
        F6_inputs=dict(path_events=int(sum(today.pref.values())), in_new_folders=int(sum(cn for p, cn in today.pref.items() if p not in seen))),
        F=[r4(x) for x in f['F'][t]], flags={k: bool(ks[k][t]) for k in ks}, c=r4(c[t]),
        models=models))
log('positives', len(out['positives']))
det = {m: sum(p['models'][m]['status'] == 'detection' and p['models'][m]['decision'] == 'anomalous' for p in out['positives']) for m in TH}
log('detected (must equal official TP)', det)
os.makedirs(config.RESULTS, exist_ok=True)
json.dump(out, open(os.path.join(config.RESULTS, 'trace_uidswap.json'), 'w'), separators=(',', ':'))
log('done')
