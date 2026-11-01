"""Analysis (post-hoc): which contextual features separate uid-swap days from normal days, and
which of them catch the swaps that Landauer misses?

Positives : the 200 swap days of seeds 0-9 (first active day after each switch).
Negatives : 4000 random scored days of seed 1, users not involved in any swap of seed 1.
Every feature compares the day with the user's own previous 14 active days (peer: 30), causally.
  existing : F1..F8, |F1| and |F5| (two-sided volume and span), Landauer nearest-day distance (30 days)
  new      : hour_js     JS divergence of today's hour-of-day profile vs the pooled history profile
             weekday_new 1 - share of history days on today's weekday
             trans_js    JS divergence of today's event-type transitions (a -> b) vs the history's
             trans_new   share of today's transitions never seen in the history
             folder_jac  1 - Jaccard(folders today, folders in the history)        (no path today -> 0)
             peer_gain   JS(today, own 30-day profile) - min over other users JS(today, their 30-day profile)
                         (> 0: today looks more like somebody else than like the account owner)
Per feature: ROC-AUC (positives vs negatives) and recall at a 5% false-positive rate (threshold =
95th percentile of the negatives), over all 200 positives and over those Landauer missed.
  Output: results/feature_screen_swap.json
  Run   : python -W ignore tools/feature_screen_swap.py <landauer_runs_dir>      (~25 min)
"""
import bisect
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

A_RUNS = sys.argv[1]
W, NPEER, NNEG, M = 14, 30, 4000, 5
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
NT = len(TYPES)


def js(p, q):
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return lib.js_div(p.astype(float), q.astype(float))


def nn_dist(x, Q):
    den = np.maximum(Q, x).sum(1)
    return float(np.min(np.where(den > 0, np.abs(Q - x).sum(1) / np.where(den > 0, den, 1), 0.0)))


# ---------------------------------------------------------------- peer profiles (original data)
PEERS = [u for u, days in BASE.items() if len(days) >= 28]
PDAYS = {u: sorted(BASE[u]) for u in PEERS}
PCUM = {}
for u in PEERS:
    T = np.array([BASE[u][d].types for d in PDAYS[u]], dtype=np.float64)
    PCUM[u] = np.vstack([np.zeros(NT), np.cumsum(T, 0)])


def peer_profile(u, day):
    """sum of u's type counts over its last NPEER active days strictly before `day` (None if < M days)."""
    k = bisect.bisect_left(PDAYS[u], day)
    if k < M:
        return None
    return PCUM[u][k] - PCUM[u][max(0, k - NPEER)]


# ---------------------------------------------------------------- samples: (seed, account, index in its recs, source uid)
SAMPLES = []          # dicts with keys: kind, seed, acc, i, src, detected
RECS = {}
FEAT = {}
for s in range(10):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    recs = lib.relabel(BASE, pairs, ct)
    f = lib.compute_features(recs, W=W)
    idx = {(u, d): i for i, (u, d) in enumerate(zip(f['uid'], f['d']))}
    det = set(json.load(open(os.path.join(A_RUNS, f'seed{s}', 'metrics.json')))['detected_users'])
    for u, t in ct.items():
        acc, day = pairs[u], int(t) // 86400
        lst = recs[acc]
        i = next(j for j, r in enumerate(lst) if r.d >= day)
        SAMPLES.append(dict(kind='pos', seed=s, acc=acc, i=i, src=u, detected=acc in det,
                            F=f['F'][idx[(acc, lst[i].d)]].tolist()))
    if s == 1:
        swapped = set(pairs)
        cand = [(u, i) for u, lst in recs.items() if u not in swapped for i in range(M, len(lst) - 1)]
        random.seed(0)
        for u, i in random.sample(cand, NNEG):
            SAMPLES.append(dict(kind='neg', seed=1, acc=u, i=i, src=u, detected=False,
                                F=f['F'][idx[(u, recs[u][i].d)]].tolist()))
    RECS[s] = {smp['acc']: recs[smp['acc']] for smp in SAMPLES if smp['seed'] == s}
    del recs, f, idx
    log(f'seed {s}: samples so far {len(SAMPLES)}')

# ---------------------------------------------------------------- transitions for the needed (original uid, day)
need = set()
for smp in SAMPLES:
    lst = RECS[smp['seed']][smp['acc']]
    need.add((smp['src'], lst[smp['i']].d))           # today's events belong to the source user
    for r in lst[max(0, smp['i'] - W):smp['i']]:
        need.add((smp['acc'], r.d))                    # history days are before any switch -> account's own
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='3GB'")
need_csv = os.path.join(config.RESULTS, 'feature_screen_need.csv')
with open(need_csv, 'w') as fh:
    fh.write('uid,d\n' + ''.join(f'{u},{d}\n' for u, d in sorted(need)))
con.execute(f"CREATE TEMP TABLE need AS SELECT uid, CAST(d AS INTEGER) AS d FROM read_csv('{need_csv}', header=true, columns={{'uid':'VARCHAR','d':'INTEGER'}})")
rows = con.execute(f"""
    WITH ev AS (
        SELECT e.uid, CAST(date_diff('day', DATE '1970-01-01', e.day) AS INTEGER) AS d, e.type, e.rn
        FROM read_parquet('{config.PARQUET}') e
        JOIN need n ON e.uid = n.uid AND CAST(date_diff('day', DATE '1970-01-01', e.day) AS INTEGER) = n.d),
    tr AS (SELECT uid, d, lag(type) OVER (PARTITION BY uid, d ORDER BY rn) AS a, type AS b FROM ev)
    SELECT uid, d, a, b, count(*) FROM tr WHERE a IS NOT NULL GROUP BY 1, 2, 3, 4""").fetchall()
TRANS = {}
for u, d, a, b, n in rows:
    TRANS.setdefault((u, d), Counter())[(a, b)] += n
del rows
log(f'transitions for {len(need)} user-days')


def wday(day):
    return (day + 3) % 7   # 1970-01-01 was a Thursday -> 0 = Monday


# ---------------------------------------------------------------- features per sample
NAMES = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'absF1', 'absF5', 'landauer_nn',
         'hour_js', 'weekday_new', 'trans_js', 'trans_new', 'folder_jac', 'peer_gain']
X = []
for smp in SAMPLES:
    lst = RECS[smp['seed']][smp['acc']]
    r, i = lst[smp['i']], smp['i']
    hist = lst[max(0, i - W):i]
    F = [np.nan if v is None else v for v in smp['F']]
    v = dict(zip(NAMES[:8], F))
    v['absF1'], v['absF5'] = abs(F[0]), abs(F[4])
    v['landauer_nn'] = nn_dist(np.asarray(r.types, float), np.array([p.types for p in lst[max(0, i - 30):i]], float))
    hh = lambda x: np.asarray(x.hist, float).reshape(24, 12).sum(1)
    v['hour_js'] = js(hh(r), sum(hh(p) for p in hist))
    v['weekday_new'] = 1 - sum(wday(p.d) == wday(r.d) for p in hist) / len(hist)
    tt = TRANS.get((smp['src'], r.d), Counter())
    th = Counter()
    for p in hist:
        th.update(TRANS.get((smp['acc'], p.d), Counter()))
    keys = sorted(set(tt) | set(th), key=str)
    v['trans_js'] = js(np.array([tt[k] for k in keys], float), np.array([th[k] for k in keys], float)) if keys else 0.0
    v['trans_new'] = (sum(n for k, n in tt.items() if k not in th) / sum(tt.values())) if tt else 0.0
    ft = set(r.pref)
    fh = set().union(*[set(p.pref) for p in hist]) if hist else set()
    v['folder_jac'] = (1 - len(ft & fh) / len(ft | fh)) if ft else 0.0
    today = np.asarray(r.types, float)
    own = np.array([p.types for p in lst[max(0, i - NPEER):i]], float).sum(0)
    own_js = js(today, own)
    best = 1.0
    for u in PEERS:
        if u == smp['acc']:
            continue
        prof = peer_profile(u, r.d)
        if prof is not None and prof.sum() > 0:
            best = min(best, js(today, prof))
    v['peer_gain'] = own_js - best
    X.append([v[n] for n in NAMES])
X = np.array(X, float)
log('features done')

kind = np.array([s['kind'] for s in SAMPLES])
missed = np.array([s['kind'] == 'pos' and not s['detected'] for s in SAMPLES])
pos, neg = kind == 'pos', kind == 'neg'
OUT = {'meta': __doc__.strip(), 'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()), 'n_missed_by_landauer': int(missed.sum()),
       'features': {}}
for j, n in enumerate(NAMES):
    x = np.nan_to_num(X[:, j], nan=0.0)
    thr = float(np.quantile(x[neg], 0.95))
    auc = lambda m: float(roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(neg.sum())], np.r_[x[m], x[neg]]))
    OUT['features'][n] = {'auc_all': round(auc(pos), 3), 'auc_missed': round(auc(missed), 3),
                          'recall_at_5pct_all': round(float((x[pos] > thr).mean()), 3),
                          'recall_at_5pct_missed': round(float((x[missed] > thr).mean()), 3),
                          'median_pos': round(float(np.median(x[pos])), 3), 'median_neg': round(float(np.median(x[neg])), 3)}
OUT['samples'] = [{'kind': s['kind'], 'seed': s['seed'], 'acc': s['acc'], 'detected': s['detected'],
                   **{n: (None if np.isnan(X[k, j]) else round(float(X[k, j]), 4)) for j, n in enumerate(NAMES)}}
                  for k, s in enumerate(SAMPLES) if s['kind'] == 'pos']
json.dump(OUT, open(os.path.join(config.RESULTS, 'feature_screen_swap.json'), 'w'), indent=1)
for n, r in sorted(OUT['features'].items(), key=lambda x: -x[1]['auc_missed']):
    log(f"{n:12s} AUC all={r['auc_all']:.3f} missed={r['auc_missed']:.3f} | recall@5% all={r['recall_at_5pct_all']:.2f} "
        f"missed={r['recall_at_5pct_missed']:.2f} | median pos={r['median_pos']} neg={r['median_neg']}")
log('DONE')
