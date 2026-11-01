"""Exploration (post-hoc), uid swap (Test 1): Landauer's detector extended with context, as the
authors suggest in their Sect. VI (compare further attributes such as accessed paths).

Same decision rule as Landauer (detect.py mode 1): per user a queue of up to 30 stored days, the
distance of today to the most similar stored day, alarm if distance > theta, one re-training day
after an alarm, matched day moved to the end of the queue. Only the distance changes:
  L       d_types                                  (Landauer as published; sanity: seed 1 -> 13/20)
  L+T     mean(d_types, d_trans)                    + event-type transitions a -> b (log order)
  L+F     mean(d_types, d_folder)                   + 2-level folder prefixes of path events
  L+T+F   mean(d_types, d_trans, d_folder)
with d_block = sum|x - y| / sum max(x, y) on that block's counts (0 when both days are empty).
Thresholds chosen on seeds 0+2: rule_budget (highest TPR with <= 1.75 false alarms/day, Landauer's
own budget) and rule_tpr (lowest FPR with TPR >= 0.65); also reported at theta = 0.6.
Reported on seed 1 and on seeds 3-9 (never used for any choice).
  Output: results/landauer_extended.json
  Run   : python -W ignore tools/landauer_extended.py      (~40 min)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

VAL, TEST, DAYS, BUDGET, QUEUE = [0, 2], [3, 4, 5, 6, 7, 8, 9], 1910, 1.75, 30
TH = [round(x, 2) for x in np.arange(0.40, 0.92, 0.02)]
VARS = {'L': ('types',), 'L+T': ('types', 'trans'), 'L+F': ('types', 'folder'), 'L+T+F': ('types', 'trans', 'folder')}
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}
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


TOPK = 200


def dense(dicts):
    """list of {key: count} -> (n x (K+1)) float32 matrix over the user's TOPK most used keys;
    all other keys share one 'other' column (keeps memory bounded for very active users)."""
    tot = {}
    for dct in dicts:
        for k, v in dct.items():
            tot[k] = tot.get(k, 0) + v
    keys = sorted(tot, key=lambda k: -tot[k])[:TOPK]
    pos = {k: j for j, k in enumerate(keys)}
    Mx = np.zeros((len(dicts), len(keys) + 1), np.float32)
    for i, dct in enumerate(dicts):
        for k, v in dct.items():
            Mx[i, pos.get(k, len(keys))] += v
    return Mx


def prepare(seed):
    """per user: blocks {types, trans, folder} as dense matrices over that user's days, + labels."""
    pairs, ct = lib.seed_swaps(config.STATS, seed)
    recs = lib.relabel(BASE, pairs, ct)
    users = []
    for uid in sorted(recs):
        lst = recs[uid]
        partner = pairs.get(uid)
        own = BASE.get(uid, {})

        def orig(d):
            return uid if (d in own and (uid not in ct or d * 86400 < ct[uid])) else partner
        users.append({'uid': uid, 'd': [r.d for r in lst],
                      'types': np.array([r.types for r in lst], np.float64),
                      'trans': dense([TRANS.get((orig(r.d), r.d), {}) for r in lst]),
                      'folder': dense([dict(r.pref) for r in lst])})
    return users, lib.labels_from_swaps(pairs, ct)


def block_dist(Mq, x):
    den = np.maximum(Mq, x).sum(1)
    return np.where(den > 0, np.abs(Mq - x).sum(1) / np.where(den > 0, den, 1), 0.0)


def detect(users, labels, blocks, theta, r=1):
    """Landauer's detect.py mode 1 with a combined distance; returns metrics dict."""
    uid_l, d_l, st_l, dc_l = [], [], [], []
    for U in users:
        n = len(U['d'])
        Q = []
        wait = r
        for t in range(n):
            if t == n - 1:
                st, dc = '', ''                     # detect.py never closes the last day
            else:
                mi, a = None, None
                if Q:
                    idx = np.array(Q)
                    dist = np.mean([block_dist(U[b][idx], U[b][t]) for b in blocks], axis=0)
                    mi = int(np.argmin(dist))
                    a = float(dist[mi])
                st = 'detection' if wait <= 0 else 'training'
                alert = a is not None and a > theta
                if alert:
                    dc = 'anomalous'
                    wait = r
                else:
                    dc = 'normal'
                    wait -= 1
                    if mi is not None:
                        Q.append(Q.pop(mi))
                if wait >= 0:
                    Q.append(t)
                if len(Q) > QUEUE:
                    Q.pop(0)
            uid_l.append(U['uid']); d_l.append(U['d'][t]); st_l.append(st); dc_l.append(dc)
    return lib.metrics(np.array(uid_l, dtype=object), np.array(d_l), np.array(st_l, dtype=object),
                       np.array(dc_l, dtype=object), labels)


_d1 = prepare(1)                   # sanity first: the plain variant must reproduce the published run
_m = detect(*_d1, VARS['L'], 0.6)
log(f"sanity L seed1 theta=0.6: TP={_m['TP']} FP={_m['FP']} (published 13 / 3337)")
del _d1, _m

DATA = {}
for s in VAL:                      # only the two validation seeds are held at once
    t = time.time()
    DATA[s] = prepare(s)
    log(f'seed {s} prepared: {time.time() - t:.0f}s')


def pooled(v, th):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        m = detect(*DATA[s], VARS[v], th)
        for k in t:
            t[k] += m[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    t['fp_per_day'] = t['FP'] / (len(VAL) * DAYS)
    return t


OUT = {'meta': __doc__.strip(), 'variants': {}}
CHOSEN = {}
for v in VARS:                     # phase 1: thresholds on the validation seeds
    curve = [(th, pooled(v, th)) for th in TH]
    inb = [x for x in curve if x[1]['fp_per_day'] <= BUDGET]
    r_budget = max(inb, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    ok = [x for x in curve if x[1]['TPR'] >= 0.65]
    r_tpr = min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(curve, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    CHOSEN[v] = {'rule_budget': r_budget[0], 'rule_tpr': r_tpr[0], 'fixed_0.6': 0.6}
    OUT['variants'][v] = {'chosen': CHOSEN[v],
                          'val_curve': {th: {k: round(m[k], 4) if isinstance(m[k], float) else m[k]
                                             for k in ('TP', 'TPR', 'FPR', 'fp_per_day')} for th, m in curve}}
    log(f'{v}: chosen {CHOSEN[v]} (val TPR at budget {r_budget[1]["TPR"]:.3f}, FP/day {r_budget[1]["fp_per_day"]:.2f})')
DATA.clear()

PER = {v: {r: {} for r in CHOSEN[v]} for v in VARS}
for s in [1] + TEST:               # phase 2: one test seed at a time
    t = time.time()
    data = prepare(s)
    if s == 1:
        m = detect(*data, VARS['L'], 0.6)
        OUT['sanity_L_seed1_0.6'] = {'TP': m['TP'], 'FP': m['FP']}
        log(f"sanity L seed1 theta=0.6: TP={m['TP']} FP={m['FP']} (published 13 / 3337)")
    for v in VARS:
        for r, th in CHOSEN[v].items():
            PER[v][r][s] = detect(*data, VARS[v], th)
    del data
    log(f'seed {s} evaluated: {time.time() - t:.0f}s')

for v in VARS:
    for r, th in CHOSEN[v].items():
        per = PER[v][r]
        OUT['variants'][v][r] = {
            'theta': th,
            'seed1': {'TP': per[1]['TP'], 'FPR': round(per[1]['FPR'], 4), 'apd': round(per[1]['alerts_per_day'], 2)},
            'test_3_9': {'TP': sum(per[s]['TP'] for s in TEST), 'of': 20 * len(TEST), 'per_seed': [per[s]['TP'] for s in TEST],
                         'FPR': round(float(np.mean([per[s]['FPR'] for s in TEST])), 4),
                         'apd': round(float(np.mean([per[s]['alerts_per_day'] for s in TEST])), 2)}}
        x = OUT['variants'][v][r]
        log(f"{v:6s} {r:11s} theta={th}: seed1 TP={x['seed1']['TP']}/20 FPR={x['seed1']['FPR']:.4f} | seeds3-9 "
            f"TP={x['test_3_9']['TP']}/140 {x['test_3_9']['per_seed']} FPR={x['test_3_9']['FPR']:.4f} apd={x['test_3_9']['apd']:.2f}")
json.dump(OUT, open(os.path.join(config.RESULTS, 'landauer_extended.json'), 'w'), indent=1, default=str)
log('DONE')
