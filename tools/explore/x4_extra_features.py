"""Exploration (post-hoc), uid swap (Test 1): model X4 with additional contextual features, chosen
on seeds 0+2 and tested on seeds 3-9 (never used for any choice) and seed 1, against Landauer.

New per user-day features (all vs the user's own previous 14 active days, causal):
  absF1      |F1|  two-sided volume deviation (a sharp drop counts too)
  absF5      |F5|  two-sided active-span deviation
  hour_js    JS divergence of today's hour-of-day profile vs the pooled history profile
  trans_js   JS divergence of today's event-type transitions (a -> b, log order) vs the history's
  trans_new  share of today's transitions never seen in the history
  folder_jac 1 - Jaccard(folders today, folders in the history); 0 without paths today
  peer_gain  JS(today, own 30-day profile) - min over long-term users JS(today, their 30-day profile)
Variants (everything else as X4: global IF 200 trees / 60 days / weekly, per-user calibration, R = 0.5a+0.4c+0.1h):
  X4        IF on relative Set A + F1..F8 (57), flags k1..k5
  X4+feat   IF on relative Set A + F1..F8 + the 7 new features (64), flags k1..k5
  X4+feat+k IF as X4+feat; flags: k1 two-sided (|F1| >= 3), k2..k5, new k6 "order changed" (trans_js >= 0.5, weight 2) -> c / 10
  X4+feat+k+strong   as above, alarm only if (k2 OR k4 OR k6) also fires
Thresholds: rule_budget = highest TPR with <= 1.75 false alarms/day on seeds 0+2; rule_tpr = lowest FPR with TPR >= 0.65.
  Output: results/x4_extra_features.json
  Run   : python -W ignore tools/x4_extra_features.py <landauer_runs_dir>      (~60 min)
"""
import bisect
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

A_RUNS = sys.argv[1]
W, NPEER, M, DAYS = 14, 30, 5, 1910
W_W, LAM, BUDGET = (0.5, 0.4, 0.1), 0.3, 1.75
VAL, TEST = [0, 2], [3, 4, 5, 6, 7, 8, 9]
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}

# ---------------------------------------------------------------- transitions per original (uid, day)
cache = os.path.join(config.WORK, 'transitions.parquet')
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='3GB'")
if not os.path.exists(cache):
    con.execute(f"""COPY (
        WITH tr AS (SELECT uid, CAST(date_diff('day', DATE '1970-01-01', day) AS INTEGER) AS d,
                           lag(type) OVER (PARTITION BY uid, day ORDER BY rn) AS a, type AS b
                    FROM read_parquet('{config.PARQUET}'))
        SELECT uid, d, a, b, count(*) AS n FROM tr WHERE a IS NOT NULL GROUP BY 1, 2, 3, 4
    ) TO '{cache}' (FORMAT PARQUET)""")
TRANS = {}
for u, d, a, b, n in con.execute(f"SELECT uid, d, a, b, n FROM read_parquet('{cache}')").fetchall():
    TRANS.setdefault((u, d), {})[TIX[a] * NT + TIX[b]] = n
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
    """JS divergence (base 2) between distribution p and each row of Q (rows NaN -> NaN)."""
    m = 0.5 * (p + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum(1)
        b = np.where(Q > 0, Q * np.log2(Q / m), 0).sum(1)
    return 0.5 * a + 0.5 * b


def js(p, q):
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return lib.js_div(p.astype(float), q.astype(float))


def extra_features(f, recs, pairs, ct):
    """7 new features aligned with compute_features' rows (same uid/day order)."""
    out = np.full((len(f['d']), 7), np.nan)
    row = 0
    for uid in sorted(recs):
        lst = recs[uid]
        partner = pairs.get(uid)
        own_days = BASE.get(uid, {})

        def orig(d, uid=uid, partner=partner, own_days=own_days):
            # relabel keeps the user's own day unless it falls after the user's own switch time
            if d in own_days and (uid not in ct or d * 86400 < ct[uid]):
                return uid
            return partner
        for i, r in enumerate(lst):
            if i > 0:
                hist = lst[max(0, i - W):i]
                Fi = f['F'][row]
                hh = lambda x: np.asarray(x.hist, float).reshape(24, 12).sum(1)
                hour_js = js(hh(r), sum(hh(p) for p in hist))
                tt = TRANS.get((orig(r.d), r.d), {})
                th = Counter()
                for p in hist:
                    th.update(TRANS.get((orig(p.d), p.d), {}))
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


def flags_new(F, E):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    k1 = g(np.abs(F[:, 0])) >= 3
    k2 = (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2)
    k3 = g(F[:, 3]) >= 0.3
    k4 = g(F[:, 5]) >= 0.5
    k5 = (g(F[:, 6]) >= 0.1) | (g(F[:, 7]) >= 3)
    k6 = g(E[:, 3]) >= 0.5
    return k2 | k4 | k6, (2 * k1 + 2 * k2 + k3 + 2 * k4 + k5 + 2 * k6) / 10.0


DATA = {}
for s in VAL + [1] + TEST:
    t = time.time()
    pairs, ct = lib.seed_swaps(config.STATS, s)
    recs = lib.relabel(BASE, pairs, ct)
    f = lib.compute_features(recs, W=W)
    E = extra_features(f, recs, pairs, ct)
    rel = lib.relative_A(f, W=W)
    avail_first = ~f['first']
    _, c_old = lib.flags_c(f['F'])
    strong_new, c_new = flags_new(f['F'], E)
    sc = {}
    for name, X in (('base', np.hstack([rel, f['F']])), ('feat', np.hstack([rel, f['F'], E]))):
        a = lib.calibrate_per_user(f, lib.iforest_a(f, np.nan_to_num(X.astype(np.float32)), schedule='rolling'))
        sc[name] = a
    av = ~np.isnan(sc['base']) & avail_first
    R = lambda a, c: np.nan_to_num(lib.risk(f, a, c, W_W, LAM), nan=-1)
    DATA[s] = {'f': f, 'lab': lib.labels_from_swaps(pairs, ct), 'avail': av,
               'X4': (R(sc['base'], c_old), None), 'X4+feat': (R(sc['feat'], c_old), None),
               'X4+feat+k': (R(sc['feat'], c_new), None), 'X4+feat+k+strong': (R(sc['feat'], c_new), strong_new)}
    del recs, E, rel
    _prof_cache.clear()
    log(f'seed {s}: {time.time() - t:.0f}s')

VARS = ['X4', 'X4+feat', 'X4+feat+k', 'X4+feat+k+strong']


def run(s, v, th):
    d = DATA[s]
    R, gate = d[v]
    al = R >= th
    if gate is not None:
        al &= gate
    return lib.evaluate(d['f'], al, d['avail'], d['lab'], r=1)[2]


def pooled(v, th):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        m = run(s, v, th)
        for k in t:
            t[k] += m[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    t['fp_per_day'] = t['FP'] / (len(VAL) * DAYS)
    return t


OUT = {'meta': __doc__.strip(), 'variants': {}}
for v in VARS:
    c = [(th, pooled(v, th)) for th in TH]
    inb = [x for x in c if x[1]['fp_per_day'] <= BUDGET]
    r_budget = max(inb, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    ok = [x for x in c if x[1]['TPR'] >= 0.65]
    r_tpr = min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(c, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    res = {}
    for rule, (th, val) in (('rule_budget', r_budget), ('rule_tpr', r_tpr)):
        per = {s: run(s, v, th) for s in [1] + TEST}
        res[rule] = {'theta': th, 'val_TPR': round(val['TPR'], 3), 'val_fp_per_day': round(val['fp_per_day'], 2),
                     'seed1': {'TP': per[1]['TP'], 'FPR': round(per[1]['FPR'], 4), 'apd': round(per[1]['alerts_per_day'], 2)},
                     'test_3_9': {'TP': sum(per[s]['TP'] for s in TEST), 'of': 20 * len(TEST),
                                  'per_seed': [per[s]['TP'] for s in TEST],
                                  'FPR': round(float(np.mean([per[s]['FPR'] for s in TEST])), 4),
                                  'apd': round(float(np.mean([per[s]['alerts_per_day'] for s in TEST])), 2)}}
        log(f"{v:18s} {rule:11s} theta={th}: val TPR={val['TPR']:.3f} FP/day={val['fp_per_day']:.2f} | seed1 TP={per[1]['TP']}/20 "
            f"| seeds3-9 TP={res[rule]['test_3_9']['TP']}/140 FPR={res[rule]['test_3_9']['FPR']:.4f} apd={res[rule]['test_3_9']['apd']:.2f}")
    OUT['variants'][v] = res
L = {s: json.load(open(os.path.join(A_RUNS, f'seed{s}', 'metrics.json'))) for s in [1] + TEST}
OUT['landauer'] = {'seed1': {'TP': L[1]['TP'], 'FPR': round(L[1]['FPR'], 4)},
                   'test_3_9': {'TP': sum(L[s]['TP'] for s in TEST), 'of': 140, 'per_seed': [L[s]['TP'] for s in TEST],
                                'FPR': round(float(np.mean([L[s]['FPR'] for s in TEST])), 4),
                                'apd': round(float(np.mean([(L[s]['TP'] + L[s]['FP']) / DAYS for s in TEST])), 2)}}
log('Landauer', OUT['landauer'])
json.dump(OUT, open(os.path.join(config.RESULTS, 'x4_extra_features.json'), 'w'), indent=1, default=str)
log('DONE')
