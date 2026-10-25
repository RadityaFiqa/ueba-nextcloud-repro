"""K3-a: daily evaluation of A, D, X4 and hybrid on K3 session hijacks (seed 1, three
intensities), with the thresholds and weights already locked (no re-tuning).
Data = clean CLUE-LDS (no uid swap); the attack events are added to the victim's
attack-day summary (counts, 5-min histogram, first/last minute, sensitive count,
path count, folder prefixes)."""
import config  # data / work / results paths (see config.py)
import json
import os
import time

import numpy as np
from scipy.stats import binomtest

import k3_common as k3
import lib
from base_daily import SENSITIVE

SEED = 1
OUT = os.path.join(config.RESULTS, 'k3_day.json')
LOG = open(os.path.join(config.RESULTS, 'k3_day.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


T0 = time.time()
BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(SEED, BASE, W30, ELIG)
CLEAN = lib.relabel(BASE, {}, {})
log(f'{len(ELIG)} eligible users, {len(PLANS)} victims, setup {time.time() - T0:.0f}s')


def inject(rec, events):
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


MODELS = ['A', 'Hybrid', 'D', 'X4']
res = {'meta': __doc__.strip(), 'plans': PLANS, 'levels': {}}
POS = {m: [] for m in MODELS}           # per positive (level, victim): detected as TP?
for level in k3.LEVELS:
    t = time.time()
    recs = dict(CLEAN)
    n_events = 0
    for p in PLANS:
        ev = k3.attack_events(p, level, SEED, BASE, ALL_PREFIX)
        n_events += len(ev)
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
    labels = {p['victim']: p['day'] for p in PLANS}
    f = lib.compute_features(recs, W=14)
    _, c = lib.flags_c(f['F'])
    out = {}
    st, dc, out['A'], _, _ = lib.hybrid_detect(f, c, w=(1, 0, 0), theta=0.6, labels=labels)
    dec = {'A': (st, dc)}
    st, dc, out['Hybrid'], _, _ = lib.hybrid_detect(f, c, w=(0.5, 0.3, 0.2), lam=0.3, theta=0.4, labels=labels)
    dec['Hybrid'] = (st, dc)
    a = lib.iforest_a(f, lib.design_matrix(f, 'AF'))
    R = lib.risk(f, a, c, (0.5, 0.4, 0.1), 0.3)
    st, dc, out['D'] = lib.evaluate(f, np.nan_to_num(R, nan=-1) >= 0.55, ~np.isnan(a) & ~f['first'], labels)
    dec['D'] = (st, dc)
    X = np.hstack([lib.relative_A(f, W=14), f['F']]).astype(np.float32)
    a = lib.calibrate_per_user(f, lib.iforest_a(f, X))
    R = lib.risk(f, a, c, (0.5, 0.4, 0.1), 0.3)
    st, dc, out['X4'] = lib.evaluate(f, np.nan_to_num(R, nan=-1) >= 0.60, ~np.isnan(a) & ~f['first'], labels)
    dec['X4'] = (st, dc)
    idx = {(u, d): i for i, (u, d) in enumerate(zip(f['uid'], f['d']))}
    for m in MODELS:
        s_, d_ = dec[m]
        for p in PLANS:
            i = idx[(p['victim'], p['day'])]
            POS[m].append(bool(s_[i] == 'detection' and d_[i] == 'anomalous'))
    res['levels'][level] = {'attack_events': n_events, 'metrics': out}
    log(f"[{level}] {n_events} injected events, {time.time() - t:.0f}s | " +
        ' | '.join(f"{m}: TP={out[m]['TP']}/20 FPR={out[m]['FPR']:.4f}" for m in MODELS))

res['pooled_60'] = {}
for m in MODELS:
    tp = sum(POS[m])
    b = sum(x and not y for x, y in zip(POS[m], POS['A']))
    cc = sum(y and not x for x, y in zip(POS[m], POS['A']))
    p = binomtest(b, b + cc, 0.5).pvalue if (m != 'A' and b + cc) else 1.0
    res['pooled_60'][m] = {'TP': tp, 'TPR': tp / len(POS[m]), 'only_model': b, 'only_A': cc, 'mcnemar_p': p}
    log(f'pooled {m}: TP {tp}/60 (TPR {tp / 60:.3f}); only {m}={b}, only A={cc}, p={p:.3g}')
json.dump(res, open(OUT, 'w'), indent=1, default=str)
log(f'K3_DAY_DONE in {time.time() - T0:.0f}s')
