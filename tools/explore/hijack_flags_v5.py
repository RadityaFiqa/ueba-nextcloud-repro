"""Diagnostic: why are session hijacks missed by version 5 (X4 + k1 two-sided + k6, theta 0.68)?
For the 60 injected sessions of Test 2 (seed 1) and for all normal user-days of the same data, reports how
often each flag fires, the values of F1 (daily volume z), F8 (10-minute burst z), F6 (new folders), a, c, h, R,
and per missed session the reason (R below theta, or the day fell in adaptation).
  Output: results/hijack_flags_v5.json
  Run   : python -W ignore tools/hijack_flags_v5.py      (~8 min)
"""
import json
import os
import sys
from collections import ChainMap, Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
import x4_refine_core as core  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

TH, W, TEST = 0.68, 14, 1
FL = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6']
BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}
core.init(BASE, TYPES, os.path.join(config.WORK, 'transitions.parquet'))


def flag_matrix(F, E):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    return np.column_stack([g(np.abs(F[:, 0])) >= 3, (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2), g(F[:, 3]) >= 0.3,
                            g(F[:, 5]) >= 0.5, (g(F[:, 6]) >= 0.1) | (g(F[:, 7]) >= 3), g(E[:, 3]) >= 0.5])


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


ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(TEST, BASE, W30, ELIG)
CLEAN = lib.relabel(BASE, {}, {})
OUT = {'meta': __doc__.strip(), 'levels': {}}
for level in k3.LEVELS:
    recs, over = dict(CLEAN), {}
    for p in PLANS:
        ev = sorted(k3.attack_events(p, level, TEST, BASE, ALL_PREFIX), key=lambda e: e[0])
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
        tr = Counter(core.S['trans'].get((p['victim'], p['day']), {}))
        for (_, a, _), (_, b, _) in zip(ev, ev[1:]):
            tr[TIX[a] * NT + TIX[b]] += 1
        over[(p['victim'], p['day'])] = dict(tr)
    f = lib.compute_features(recs, W=W)
    E = core.extra_features(f, recs, {}, {}, peer=False, trans=ChainMap(over, core.S['trans']))[:, :6]
    X = np.nan_to_num(np.hstack([lib.relative_A(f, W=W), f['F']]).astype(np.float32))
    a = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    K = flag_matrix(f['F'], E)
    c = (K * np.array([2, 2, 1, 2, 1, 2])).sum(1) / 10.0
    R = lib.risk(f, a, c, (0.5, 0.4, 0.1), 0.3)
    h = (R - 0.5 * a - 0.4 * c) / 0.1
    avail = ~np.isnan(a) & ~f['first']
    labels = {p['victim']: p['day'] for p in PLANS}
    status, dec, m = lib.evaluate(f, np.nan_to_num(R, nan=-1) >= TH, avail, labels, r=1)
    att = np.array([int(np.flatnonzero((f['uid'] == u) & (f['d'] == d))[0]) for u, d in labels.items()])
    normal = (status == 'detection') & ~np.isin(np.arange(len(R)), att)
    rows = []
    for idx in att:
        det = status[idx] == 'detection' and dec[idx] == 'anomalous'
        why = 'terdeteksi' if det else ('hari adaptasi' if status[idx] == 'training' else 'R di bawah batas')
        rows.append({'akun': f['uid'][idx], 'alasan': why, 'a': round(float(a[idx]), 3), 'c': round(float(c[idx]), 2),
                     'h': round(float(h[idx]), 3), 'R': round(float(R[idx]), 3), **{k: bool(v) for k, v in zip(FL, K[idx])},
                     'F1': round(float(f['F'][idx, 0]), 2), 'F6': round(float(f['F'][idx, 5]), 2), 'F8': round(float(f['F'][idx, 7]), 2)})
    OUT['levels'][level] = {
        'TP': m['TP'],
        'flag_rate_attack': {k: round(float(K[att, j].mean()), 3) for j, k in enumerate(FL)},
        'flag_rate_normal': {k: round(float(K[normal, j].mean()), 3) for j, k in enumerate(FL)},
        'median_attack': {k: round(float(np.nanmedian(v[att])), 3) for k, v in (('a', a), ('c', c), ('R', R), ('F1', f['F'][:, 0]), ('F8', f['F'][:, 7]))},
        'median_normal': {k: round(float(np.nanmedian(v[normal])), 3) for k, v in (('a', a), ('c', c), ('R', R))},
        'reasons': dict(Counter(r['alasan'] for r in rows)), 'sessions': rows}
    L = OUT['levels'][level]
    print(level, 'TP', L['TP'], L['reasons'], '\n  attack flags', L['flag_rate_attack'], '\n  normal flags', L['flag_rate_normal'],
          '\n  median attack', L['median_attack'], ' normal', L['median_normal'], flush=True)
    del recs, over, f, E, X
json.dump(OUT, open(os.path.join(config.RESULTS, 'hijack_flags_v5.json'), 'w'), indent=1, default=str)
print('DONE')
