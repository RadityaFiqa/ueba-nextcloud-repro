"""Exploration (post-hoc): the base model of x4_refine.py ("X4 + extra features + k6", weights .5/.4/.1)
at fixed thresholds 0.60..0.70, uid swap (Test 1) on seeds 0-9, next to Landauer (theta 0.6).
No selection is made here; it only shows the detection / false-alarm trade-off per threshold.
  Output: results/x4_refine_th.json
  Run   : python -W ignore tools/x4_refine_th.py <landauer_runs_dir>      (~20 min)
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

A_RUNS = sys.argv[1]
W, NPEER, M, DAYS = 14, 30, 5, 1910
LAM, BUDGET = 0.3, 1.75
VAL, TEST1, ROBUST = [0, 2], 1, [3, 4, 5, 6, 7, 8, 9]
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
WEIGHTS = [(0.5, 0.4, 0.1), (0.6, 0.3, 0.1), (0.4, 0.5, 0.1), (0.3, 0.6, 0.1)]
CONFIGS = [dict(w=w, k5tight=k, strong=s) for w, k, s in itertools.product(WEIGHTS, [False, True], [False, True])]
OUTF = os.path.join(config.RESULTS, 'x4_refine_th.json')
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


THS = [round(x, 2) for x in np.arange(0.60, 0.705, 0.01)]
BASEC = CONFIGS[0]
L = {s: json.load(open(os.path.join(A_RUNS, f'seed{s}', 'metrics.json'))) for s in range(10)}
OUT = {'meta': __doc__.strip(), 'config': cname(BASEC), 'seeds': {}}
for s in range(10):
    t = time.time()
    d, lab = test1(s, [BASEC])
    row = {'Landauer': {'TP': L[s]['TP'], 'FP': L[s]['FP'], 'FPR': round(L[s]['FP'] / (L[s]['FP'] + L[s]['TN']), 4),
                        'fp_per_day': round(L[s]['FP'] / DAYS, 3)}}
    for th in THS:
        m = run(d, lab, BASEC, th)
        row[str(th)] = {'TP': m['TP'], 'FP': m['FP'], 'FPR': round(m['FPR'], 4), 'fp_per_day': round(m['FP'] / DAYS, 3)}
    OUT['seeds'][s] = row
    del d
    save(OUT)
    log(f"seed {s}: Landauer TP={row['Landauer']['TP']} | " + ' '.join(f"{th}:{row[str(th)]['TP']}/{row[str(th)]['FPR']:.3f}" for th in THS) + f' ({time.time() - t:.0f}s)')
log('DONE')
