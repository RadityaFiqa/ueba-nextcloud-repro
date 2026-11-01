"""STEP 12a - FINAL experiment, data preparation (run parts A and B in parallel, one per CPU).

For every data set (uid swap seeds 0-9; session hijack seeds 0, 2 and 1 at three intensities) it stores what the
evaluation (step12_final_eval.py) needs, as work/final/<name>.npz:
  * day index: uid, d, first, last
  * F1..F8 and trans_js (flags are computed in the evaluation)
  * a_raw (global IF percentile on the 57 user-relative inputs) and a (ranked against the user's previous 30 days)
  * Landauer's detector (nearest-day distance, lib.hybrid_detect w=(1,0,0)) at several thresholds:
    alarm and training-status masks per threshold
  * seed 1 only: a for the ablations (IF refitted without one feature group, without the 49 relative counts,
    or with raw log counts instead of relative counts)
All inputs are relative to the assessed user's own history (14 active days); no labels, no IP.
  Run: python -W ignore src/step12_final_prep.py A   and   python -W ignore src/step12_final_prep.py B
"""
import os
import sys
import time
from collections import ChainMap, Counter

import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
import lib_order_features as core  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

PART = sys.argv[1]
W = 14
L_TH = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
OUT_DIR = os.path.join(config.WORK, 'final')
os.makedirs(OUT_DIR, exist_ok=True)
JOBS = {'A': [('swap', s, None) for s in (0, 1, 3, 5, 7)] + [('hijack', 0, lv) for lv in k3.LEVELS] + [('hijack', 1, 'rendah')],
        'B': [('swap', s, None) for s in (2, 4, 6, 8, 9)] + [('hijack', 2, lv) for lv in k3.LEVELS] + [('hijack', 1, 'sedang'), ('hijack', 1, 'tinggi')]}
# IF refits for the ablations (seed 1 only): columns of [rel (49) | F1..F8] kept, or raw counts
F_GROUPS = {'tanpa_jumlah': [0], 'tanpa_pola': [1, 2], 'tanpa_waktu': [3, 4], 'tanpa_folder': [5], 'tanpa_risiko': [6], 'tanpa_lonjakan': [7]}
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', PART, *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}
TRANS = os.path.join(config.WORK, 'transitions.parquet')
if not os.path.exists(TRANS):  # per (uid, day): counts of consecutive event-type pairs a -> b in log order
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='3GB'")
    con.execute(f"""COPY (
        WITH tr AS (SELECT uid, CAST(date_diff('day', DATE '1970-01-01', day) AS INTEGER) AS d,
                           lag(type) OVER (PARTITION BY uid, day ORDER BY rn) AS a, type AS b
                    FROM read_parquet('{config.PARQUET}'))
        SELECT uid, d, a, b, count(*) AS n FROM tr WHERE a IS NOT NULL GROUP BY 1, 2, 3, 4
    ) TO '{TRANS}' (FORMAT PARQUET)""")
    con.close()
if PART == 'T':  # only build the transitions table (run once before parts A and B)
    sys.exit(0)
core.init(BASE, TYPES, TRANS)
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
CLEAN = lib.relabel(BASE, {}, {})


def inject(rec, events):  # as step09
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


def build(kind, s, lv):
    if kind == 'swap':
        pairs, ct = lib.seed_swaps(config.STATS, s)
        return lib.relabel(BASE, pairs, ct), pairs, ct, None, lib.labels_from_swaps(pairs, ct)
    plans = k3.plan_attacks(s, BASE, W30, ELIG)
    recs, over = dict(CLEAN), {}
    for p in plans:
        ev = sorted(k3.attack_events(p, lv, s, BASE, ALL_PREFIX), key=lambda e: e[0])
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
        tr = Counter(core.S['trans'].get((p['victim'], p['day']), {}))
        for (_, a, _), (_, b, _) in zip(ev, ev[1:]):
            tr[TIX[a] * NT + TIX[b]] += 1
        over[(p['victim'], p['day'])] = dict(tr)
    return recs, {}, {}, ChainMap(over, core.S['trans']), {p['victim']: p['day'] for p in plans}


def if_a(f, X):
    a_raw = lib.iforest_a(f, np.nan_to_num(X.astype(np.float32)), schedule='rolling')
    return a_raw, lib.calibrate_per_user(f, a_raw)


for kind, s, lv in JOBS[PART]:
    t = time.time()
    name = f'{kind}_s{s}' + (f'_{lv}' if lv else '')
    recs, pairs, ct, trans, lab = build(kind, s, lv)
    f = lib.compute_features(recs, W=W)
    E = core.extra_features(f, recs, pairs, ct, peer=False, trans=trans)
    rel = lib.relative_A(f, W=W)
    a_raw, a = if_a(f, np.hstack([rel, f['F']]))
    out = dict(uid=np.asarray(f['uid'], dtype=str), d=f['d'], first=f['first'], last=f['last'], F=f['F'], trans_js=E[:, 3],
               a_raw=a_raw, a=a, lab_u=np.array(list(lab), dtype=str), lab_d=np.array(list(lab.values())))
    _, c0 = lib.flags_c(f['F'])
    for th in L_TH:
        st, dc = lib.hybrid_detect(f, c0, w=(1, 0, 0), theta=th, labels=lab)[:2]
        out[f'L_alarm_{th}'] = dc == 'anomalous'
        out[f'L_train_{th}'] = st == 'training'
    if s == 1:
        for g, cols in F_GROUPS.items():
            keep = [c for c in range(8) if c not in cols]
            out[f'a_{g}'] = if_a(f, np.hstack([rel, f['F'][:, keep]]))[1]
        out['a_tanpa_jumlah_relatif'] = if_a(f, f['F'])[1]
        out['a_jumlah_mentah'] = if_a(f, np.hstack([np.log1p(f['A'].astype(np.float64)), f['F']]))[1]
    np.savez_compressed(os.path.join(OUT_DIR, name + '.npz'), **out)
    del recs, f, E, rel, out
    log(f'{name}: {time.time() - t:.0f}s')
log('PREP_DONE')
