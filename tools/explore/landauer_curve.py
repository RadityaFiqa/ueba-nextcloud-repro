"""Landauer's detector (lib.hybrid_detect with w = (1, 0, 0), i.e. nearest-day distance only) at several
thresholds on seed 1: uid swap and the 60 session hijacks (same generator as step09). Threshold 0.6 must
reproduce 13/20 with 3,337 false alarms (swap) and 9/60 (hijack).
  Output: results/landauer_curve.json
  Run   : python -W ignore tools/landauer_curve.py      (~6 min)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

TH = [0.45, 0.5, 0.55, 0.56, 0.6, 0.65, 0.7, 0.75]
BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}


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


def landauer(recs, labels):
    f = lib.compute_features(recs, W=14)
    _, c = lib.flags_c(f['F'])
    out = {}
    for th in TH:
        m = lib.hybrid_detect(f, c, w=(1, 0, 0), theta=th, labels=labels)[2]
        out[th] = {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'TP_adj', 'FP', 'FPR', 'alerts_per_day')}
    return out


pairs, ct = lib.seed_swaps(config.STATS, 1)
OUT = {'meta': __doc__.strip(), 'swap': landauer(lib.relabel(BASE, pairs, ct), lib.labels_from_swaps(pairs, ct)), 'hijack': {}}
print('swap', {th: (v['TP'], v['FP']) for th, v in OUT['swap'].items()}, flush=True)
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(1, BASE, W30, ELIG)
CLEAN = lib.relabel(BASE, {}, {})
for level in k3.LEVELS:
    recs = dict(CLEAN)
    for p in PLANS:
        ev = k3.attack_events(p, level, 1, BASE, ALL_PREFIX)
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
    OUT['hijack'][level] = landauer(recs, {p['victim']: p['day'] for p in PLANS})
    print('hijack', level, {th: v['TP'] for th, v in OUT['hijack'][level].items()}, flush=True)
for th in TH:
    tot = sum(OUT['hijack'][lv][th]['TP'] for lv in k3.LEVELS)
    adj = sum(OUT['hijack'][lv][th]['TP_adj'] for lv in k3.LEVELS)
    apd = np.mean([OUT['hijack'][lv][th]['alerts_per_day'] for lv in k3.LEVELS])
    s = OUT['swap'][th]
    print(f"theta {th}: swap {s['TP']}/20 (adj {s['TP_adj']}) FP {s['FP']} FPR {s['FPR']:.4f} {s['FP'] / 1910:.2f}/hari | hijack {tot}/60 (adj {adj}) "
          + '/'.join(str(OUT['hijack'][lv][th]['TP']) for lv in k3.LEVELS), flush=True)
json.dump(OUT, open(os.path.join(config.RESULTS, 'landauer_curve.json'), 'w'), indent=1, default=str)
print('DONE')
