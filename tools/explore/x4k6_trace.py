"""Reproduction of "X4 + extra features + k6" (x4_extra_features.py, variant X4+feat+k) with a full trace.
Recomputes everything from scratch: validation on seeds 0+2 picks theta with rule_budget (highest TPR with
<= 1.75 false alarms/day), then seed 1 is scored. For the 20 swap days of seed 1 every intermediate value is
written out: a_raw (global IF percentile), a (rank among the user's previous 30 days), k1..k6, c = sum(w)/10,
h, R = .5a + .4c + .1h, the evaluation status (detection/training) and the decision.
Expected (x4_extra_features.json): theta 0.68, seed 1 TP 12/20, FPR 0.044.
  Output: results/x4k6_trace.json
  Run   : python -W ignore tools/x4k6_trace.py      (~12 min)
"""
import bisect
import itertools
import json
import os
import sys
import time
from collections import ChainMap, Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

W, NPEER, M, DAYS = 14, 30, 5, 1910
LAM, BUDGET = 0.3, 1.75
VAL, TEST1, ROBUST = [0, 2], 1, [3, 4, 5, 6, 7, 8, 9]
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
WEIGHTS = [(0.5, 0.4, 0.1), (0.6, 0.3, 0.1), (0.4, 0.5, 0.1), (0.3, 0.6, 0.1)]
CONFIGS = [dict(w=w, k5tight=k, strong=s) for w, k, s in itertools.product(WEIGHTS, [False, True], [False, True])]
OUTF = os.path.join(config.RESULTS, 'x4k6_trace.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def cname(c):
    return 'w=' + '/'.join(str(x) for x in c['w']) + ('+k5tight' if c['k5tight'] else '') + ('+strong' if c['strong'] else '')


def n_changes(c):
    return (c['w'] != WEIGHTS[0]) + c['k5tight'] + c['strong']


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}

# ---------------------------------------------------------------- transitions per original (uid, day)
cache = os.path.join(config.WORK, 'transitions.parquet')
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='3GB'")
TRANS = {}
for u, d, a, b, n in con.execute(f"SELECT uid, d, a, b, n FROM read_parquet('{cache}')").fetchall():
    TRANS.setdefault((u, d), {})[TIX[a] * NT + TIX[b]] = n
con.close()
log(f'transitions: {len(TRANS)} user-days')

# ---------------------------------------------------------------- peer profiles per date (original data)
PEERS = [u for u, days in BASE.items() if len(days) >= 28]
PIDX = {u: k for k, u in enumerate(PEERS)}
PD = {u: sorted(BASE[u]) for u in PEERS}
PC = {u: np.vstack([np.zeros(NT), np.cumsum(np.array([BASE[u][d].types for d in PD[u]], float), 0)]) for u in PEERS}
_prof_cache = {}


def peer_matrix(day):
    if day not in _prof_cache:
        P = np.zeros((len(PEERS), NT))
        for u in PEERS:
            k = bisect.bisect_left(PD[u], day)
            if k >= M:
                P[PIDX[u]] = PC[u][k] - PC[u][max(0, k - NPEER)]
        s = P.sum(1, keepdims=True)
        _prof_cache[day] = np.where(s > 0, P / np.where(s > 0, s, 1), np.nan)
    return _prof_cache[day]


def js_rows(p, Q):
    m = 0.5 * (p + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum(1)
        b = np.where(Q > 0, Q * np.log2(Q / m), 0).sum(1)
    return 0.5 * a + 0.5 * b


def js(p, q):
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return lib.js_div(p.astype(float), q.astype(float))


def extra_features(f, recs, pairs, ct, trans):
    """7 extra features aligned with compute_features' rows (as x4_extra_features.py)."""
    out = np.full((len(f['d']), 7), np.nan)
    row = 0
    for uid in sorted(recs):
        lst = recs[uid]
        partner = pairs.get(uid)
        own_days = BASE.get(uid, {})

        def orig(d, uid=uid, partner=partner, own_days=own_days):
            if d in own_days and (uid not in ct or d * 86400 < ct[uid]):
                return uid
            return partner
        for i, r in enumerate(lst):
            if i > 0:
                hist = lst[max(0, i - W):i]
                Fi = f['F'][row]
                hh = lambda x: np.asarray(x.hist, float).reshape(24, 12).sum(1)
                hour_js = js(hh(r), sum(hh(p) for p in hist))
                tt = trans.get((orig(r.d), r.d), {})
                th = Counter()
                for p in hist:
                    th.update(trans.get((orig(p.d), p.d), {}))
                keys = set(tt) | set(th)
                trans_js = js(np.array([tt.get(k, 0) for k in keys], float), np.array([th.get(k, 0) for k in keys], float)) if keys else 0.0
                trans_new = (sum(n for k, n in tt.items() if k not in th) / sum(tt.values())) if tt else 0.0
                ft = set(r.pref)
                fh = set().union(*[set(p.pref) for p in hist])
                folder_jac = (1 - len(ft & fh) / len(ft | fh)) if ft else 0.0
                today = np.asarray(r.types, float)
                own = np.array([p.types for p in lst[max(0, i - NPEER):i]], float).sum(0)
                own_js = js(today, own)
                P = peer_matrix(r.d)
                mask = ~np.isnan(P[:, 0])
                if uid in PIDX:
                    mask[PIDX[uid]] = False
                best = float(np.nanmin(js_rows(today / today.sum(), P[mask]))) if mask.any() and today.sum() > 0 else 1.0
                out[row] = (abs(Fi[0]), abs(Fi[4]), hour_js, trans_js, trans_new, folder_jac, own_js - best)
            row += 1
    return out


def flags(F, E, k5tight):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    k1 = g(np.abs(F[:, 0])) >= 3
    k2 = (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2)
    k3_ = g(F[:, 3]) >= 0.3
    k4 = g(F[:, 5]) >= 0.5
    k5 = ((g(F[:, 6]) >= 0.2) | (g(F[:, 7]) >= 5)) if k5tight else ((g(F[:, 6]) >= 0.1) | (g(F[:, 7]) >= 3))
    k6 = g(E[:, 3]) >= 0.5
    return k2 | k4 | k6, (2 * k1 + 2 * k2 + k3_ + 2 * k4 + k5 + 2 * k6) / 10.0


def prepare(recs, pairs, ct, trans, configs):
    """Scores of each configuration for one data set (relabelled or injected)."""
    f = lib.compute_features(recs, W=W)
    E = extra_features(f, recs, pairs, ct, trans)
    X = np.nan_to_num(np.hstack([lib.relative_A(f, W=W), f['F'], E]).astype(np.float32))
    a = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    out = {'f': f, 'avail': ~np.isnan(a) & ~f['first'], 'R': {}}
    for c in configs:
        strong, cc = flags(f['F'], E, c['k5tight'])
        out['R'][cname(c)] = (np.nan_to_num(lib.risk(f, a, cc, c['w'], LAM), nan=-1), strong if c['strong'] else None)
    _prof_cache.clear()
    return out


def run(d, lab, c, th):
    R, gate = d['R'][cname(c)]
    al = R >= th
    if gate is not None:
        al &= gate
    return lib.evaluate(d['f'], al, d['avail'], lab, r=1)[2]


def test1(s, configs):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    d = prepare(lib.relabel(BASE, pairs, ct), pairs, ct, TRANS, configs)
    return d, lib.labels_from_swaps(pairs, ct)


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


def save(out):
    json.dump(out, open(OUTF, 'w'), indent=1, default=str)



FL = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6']
FW = np.array([2, 2, 1, 2, 1, 2])


def flag_matrix(F, E):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    return np.column_stack([g(np.abs(F[:, 0])) >= 3, (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2), g(F[:, 3]) >= 0.3,
                            g(F[:, 5]) >= 0.5, (g(F[:, 6]) >= 0.1) | (g(F[:, 7]) >= 3), g(E[:, 3]) >= 0.5])


def build(s):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    recs = lib.relabel(BASE, pairs, ct)
    f = lib.compute_features(recs, W=W)
    E = extra_features(f, recs, pairs, ct, TRANS)
    X = np.nan_to_num(np.hstack([lib.relative_A(f, W=W), f['F'], E]).astype(np.float32))
    a_raw = lib.iforest_a(f, X, schedule='rolling')
    a = lib.calibrate_per_user(f, a_raw)
    K = flag_matrix(f['F'], E)
    c = (K * FW).sum(1) / 10.0
    R = lib.risk(f, a, c, (0.5, 0.4, 0.1), LAM)
    h = np.where(np.isnan(R), np.nan, (R - 0.5 * a - 0.4 * c) / 0.1)
    _prof_cache.clear()
    return dict(f=f, E=E, a_raw=a_raw, a=a, K=K, c=c, h=h, R=np.nan_to_num(R, nan=-1), avail=~np.isnan(a) & ~f['first'],
                lab=lib.labels_from_swaps(pairs, ct), pairs=pairs)


D = {}
for s in [0, 2, 1]:
    t = time.time()
    D[s] = build(s)
    log(f'seed {s}: {time.time() - t:.0f}s')


def ev(s, th):
    d = D[s]
    return lib.evaluate(d['f'], d['R'] >= th, d['avail'], d['lab'], r=1)


rows = []
for th in TH:
    m = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        r = ev(s, th)[2]
        for k in m:
            m[k] += r[k]
    m['TPR'], m['FPR'], m['fp_per_day'] = m['TP'] / (m['TP'] + m['FN']), m['FP'] / (m['FP'] + m['TN']), m['FP'] / (len(VAL) * DAYS)
    rows.append((th, m))
th, vm = max([x for x in rows if x[1]['fp_per_day'] <= BUDGET], key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
status, dec, m1 = ev(TEST1, th)
log(f'theta={th} validation TP={vm["TP"]}/40 FPR={vm["FPR"]:.4f} FP/day={vm["fp_per_day"]:.2f}')
log(f'seed 1: TP={m1["TP"]}/20 FP={m1["FP"]} FPR={m1["FPR"]:.4f} alerts/day={m1["alerts_per_day"]:.2f}')

d = D[TEST1]
f = d['f']
inv = {v: k for k, v in d['pairs'].items()}
trace = []
for u, day in sorted(d['lab'].items()):
    i = int(np.flatnonzero((f['uid'] == u) & (f['d'] == day))[0])
    trace.append({'akun': u, 'aktivitas_dari': inv.get(u), 'hari': int(day), 'status': status[i], 'keputusan': dec[i],
                  'terdeteksi': bool(status[i] == 'detection' and dec[i] == 'anomalous'),
                  'a_mentah': round(float(d['a_raw'][i]), 3), 'a': round(float(d['a'][i]), 3),
                  **{k: bool(v) for k, v in zip(FL, d['K'][i])}, 'c': round(float(d['c'][i]), 2),
                  'h': round(float(d['h'][i]), 3), 'R': round(float(d['R'][i]), 3),
                  'F1': round(float(f['F'][i, 0]), 2), 'F2': round(float(f['F'][i, 1]), 3), 'F6': round(float(f['F'][i, 5]), 2),
                  'trans_js': round(float(d['E'][i, 3]), 3), 'trans_new': round(float(d['E'][i, 4]), 3)})
OUT = {'meta': __doc__.strip(), 'theta': th, 'validation': {k: round(v, 4) if isinstance(v, float) else v for k, v in vm.items()},
       'seed1': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m1.items()}, 'trace_seed1': trace}
save(OUT)
for r in trace:
    log(f"{r['akun'][:28]:28s} {r['status'][:6]} {'ALARM' if r['terdeteksi'] else '-':5s} a={r['a']:.2f} c={r['c']:.1f} h={r['h']:.2f} R={r['R']:.3f} "
        + ''.join(k for k in FL if r[k]) + f" trans_js={r['trans_js']:.2f} F2={r['F2']:.2f}")
log('DONE')
