"""Additional analysis: how much do the contextual features F1-F8 contribute to model D on Test 2
(K3 session hijack, per day, seed 1, 60 sessions)?

Variants of model D (everything else locked: W = 14, IF 200 trees / 60-day window / weekly retrain,
R = 0.5a + 0.4c + 0.1h, lambda = 0.3, threshold 0.55):
  D           full model
  D-c         context score c switched off (R from a and h only)
  D-F(IF)     F1-F8 removed from the Isolation Forest input (49 type counts only), c kept
  D-F(all)    F1-F8 removed everywhere (IF on 49 counts, c off)
  D-Fi        feature Fi removed from the IF input AND from the flags that use it
Each variant is reported (1) at the fixed threshold 0.55 and (2) at an equal alarm budget:
the threshold where its false alarms do not exceed those of the full model D.
Also reported: how often each flag k1-k5 fires on the 60 attack days vs on normal days,
and the average contribution of 0.5a, 0.4c and 0.1h to R on the attack days.
  Output: results/feature_ablation.json
  Run   : python -W ignore tools/feature_ablation.py      (~45 min)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

SEED, TH, W_RISK, LAM = 1, 0.55, (0.5, 0.4, 0.1), 0.3
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(SEED, BASE, W30, ELIG)
CLEAN = lib.relabel(BASE, {}, {})


def inject(rec, events):  # same as step09
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


def run(f, R, avail, labels, th):
    _, _, m = lib.evaluate(f, np.nan_to_num(R, nan=-1) >= th, avail, labels)
    return m


def equal_budget(f, R, avail, labels, fp_max):
    """Smallest threshold whose false alarms <= fp_max (bisection)."""
    lo, hi = 0.0, 1.5
    for _ in range(14):
        mid = (lo + hi) / 2
        if run(f, R, avail, labels, mid)['FP'] <= fp_max:
            hi = mid
        else:
            lo = mid
    m = run(f, R, avail, labels, hi)
    return round(hi, 4), m


OUT = {'meta': __doc__.strip(), 'levels': {}, 'pooled': {}}
VARS = ['D', 'D-c', 'D-F(IF)', 'D-F(all)'] + [f'D-F{i + 1}' for i in range(8)]
POOL = {v: {'TP_055': 0, 'FP_055': 0, 'TP_eq': 0, 'FP_eq': 0} for v in VARS}
FLAG_ATT = {k: 0 for k in lib.FLAGS}
FLAG_NORM = {k: 0 for k in lib.FLAGS}
N_NORM = 0
FEAT_ATT, FEAT_NORM = [], []
COMP = []  # (0.5a, 0.4c, 0.1h) normalised components on attack days
for level in k3.LEVELS:
    t = time.time()
    recs = dict(CLEAN)
    for p in PLANS:
        ev = k3.attack_events(p, level, SEED, BASE, ALL_PREFIX)
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
    labels = {p['victim']: p['day'] for p in PLANS}
    f = lib.compute_features(recs, W=14)
    ks, c = lib.flags_c(f['F'])
    idx = {(u, d): i for i, (u, d) in enumerate(zip(f['uid'], f['d']))}
    att = np.zeros(len(f['d']), bool)
    for p in PLANS:
        att[idx[(p['victim'], p['day'])]] = True
    norm = ~att & ~f['first']
    for k in lib.FLAGS:
        FLAG_ATT[k] += int(ks[k][att].sum())
        FLAG_NORM[k] += int(ks[k][norm].sum())
    N_NORM += int(norm.sum())
    FEAT_ATT.append(f['F'][att])
    FEAT_NORM.append(np.nan_to_num(f['F'][norm]))

    a_full = lib.iforest_a(f, lib.design_matrix(f, 'AF'))
    a_A = lib.iforest_a(f, lib.design_matrix(f, 'A'))
    avail = ~np.isnan(a_full) & ~f['first']
    zero = np.zeros(len(c))
    R = {'D': lib.risk(f, a_full, c, W_RISK, LAM),
         'D-c': lib.risk(f, a_full, zero, (0.5, 0.0, 0.1), LAM),
         'D-F(IF)': lib.risk(f, a_A, c, W_RISK, LAM),
         'D-F(all)': lib.risk(f, a_A, zero, (0.5, 0.0, 0.1), LAM)}
    # components of the full model on attack days (h rebuilt from R as in lib.risk)
    hval = (R['D'] - 0.5 * a_full - 0.4 * c) / 0.1
    for i in np.flatnonzero(att):
        COMP.append((0.5 * a_full[i], 0.4 * c[i], 0.1 * hval[i], R['D'][i]))
    for fi in range(8):
        F2 = f['F'].copy()
        F2[:, fi] = np.nan                       # flag conditions on Fi never fire
        _, ci = lib.flags_c(F2)
        ai = lib.iforest_a(f, lib.design_matrix(f, ('AF', [fi])))
        R[f'D-F{fi + 1}'] = lib.risk(f, ai, ci, W_RISK, LAM)
    res = {}
    base_m = run(f, R['D'], avail, labels, TH)
    for v in VARS:
        av = ~np.isnan(R[v]) & ~f['first']
        m055 = base_m if v == 'D' else run(f, R[v], av, labels, TH)
        th_eq, meq = (TH, base_m) if v == 'D' else equal_budget(f, R[v], av, labels, base_m['FP'])
        res[v] = {'TP_055': m055['TP'], 'FP_055': m055['FP'], 'th_eq': th_eq, 'TP_eq': meq['TP'], 'FP_eq': meq['FP']}
        for k in POOL[v]:
            POOL[v][k] += res[v][k]
    OUT['levels'][level] = res
    log(f'[{level}] {time.time() - t:.0f}s ' + ' | '.join(f"{v}: {res[v]['TP_055']}/{res[v]['TP_eq']}" for v in VARS))
    json.dump(OUT, open(os.path.join(config.RESULTS, 'feature_ablation.json'), 'w'), indent=1, default=str)

days = 1910
OUT['pooled'] = {v: {**POOL[v], 'alerts_per_day_eq': round(POOL[v]['FP_eq'] / 3 / days, 2)} for v in VARS}
OUT['flag_rate'] = {k: {'attack': round(FLAG_ATT[k] / 60, 3), 'normal': round(FLAG_NORM[k] / N_NORM, 4)} for k in lib.FLAGS}
FA, FN = np.vstack(FEAT_ATT), np.vstack(FEAT_NORM)
OUT['feature_median'] = {lib.FNAMES[i]: {'attack': round(float(np.median(FA[:, i])), 3),
                                         'normal': round(float(np.median(FN[:, i])), 3)} for i in range(8)}
C = np.array(COMP)
OUT['components_attack_days'] = {'0.5a': round(float(C[:, 0].mean()), 3), '0.4c': round(float(C[:, 1].mean()), 3),
                                 '0.1h': round(float(C[:, 2].mean()), 3), 'R': round(float(C[:, 3].mean()), 3),
                                 'c_decisive': int(((C[:, 3] >= TH) & (C[:, 0] + C[:, 2] < TH)).sum())}
json.dump(OUT, open(os.path.join(config.RESULTS, 'feature_ablation.json'), 'w'), indent=1, default=str)
log(json.dumps({k: OUT[k] for k in ('pooled', 'flag_rate', 'feature_median', 'components_attack_days')}, indent=1))
log('DONE')
