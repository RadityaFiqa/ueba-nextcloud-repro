"""Exploration (post-hoc): minimal versions of "X4 + order of activities + k6", without the redundant extras
(trans_new, hour_js, folder_jac, |F1|, |F5|) and without peer_gain. All inputs relative to the user's own history.
  A  IF on 57 inputs (relative counts + F1..F8, as X4); flags k1 two-sided, k2..k5, k6 (trans_js >= 0.5, w 2); c / 10
  B  IF on 58 inputs (A + trans_js); same flags
  C  IF on 63 inputs (x4k6_nopeer.py: + |F1|, |F5|, hour_js, trans_js, trans_new, folder_jac); same flags
R = .5a + .4c + .1h, lambda 0.3, a ranked against the user's own previous 30 days, W = 14.
Threshold per version chosen on seeds 0+2 only with rule_budget (highest TPR with <= 1.75 false alarms/day).
Reported on seed 1 + seeds 3-9 (uid swap) and Test 2 seed 1 (session hijack).
  Output: results/x4k6_minimal.json
  Run   : python -W ignore tools/x4k6_minimal.py <landauer_runs_dir>      (~35 min)
"""
import json
import os
import sys
import time
from collections import ChainMap, Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
import x4_refine_core as core  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

A_RUNS = sys.argv[1]
W, LAM, BUDGET, DAYS = 14, 0.3, 1.75, 1910
VAL, TEST, ROBUST = [0, 2], 1, [3, 4, 5, 6, 7, 8, 9]
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
VERSIONS = {'A': [], 'B': [3], 'C': [0, 1, 2, 3, 4, 5]}  # columns of E (|F1|, |F5|, hour_js, trans_js, trans_new, folder_jac) fed to the IF
OUTF = os.path.join(config.RESULTS, 'x4k6_minimal.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}
core.init(BASE, TYPES, os.path.join(config.WORK, 'transitions.parquet'))


def prepare(recs, pairs, ct, trans=None):
    f = lib.compute_features(recs, W=W)
    E = core.extra_features(f, recs, pairs, ct, peer=False, trans=trans)[:, :6]
    rel = lib.relative_A(f, W=W)
    c = core.c_k6(f['F'], E)
    out = {'f': f, 'R': {}, 'avail': None}
    for v, cols in VERSIONS.items():
        X = np.nan_to_num(np.hstack([rel, f['F'], E[:, cols]]).astype(np.float32))
        a = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
        out['avail'] = ~np.isnan(a) & ~f['first']
        out['R'][v] = np.nan_to_num(lib.risk(f, a, c, (0.5, 0.4, 0.1), LAM), nan=-1)
    return out


def test1(s):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    return prepare(lib.relabel(BASE, pairs, ct), pairs, ct), lib.labels_from_swaps(pairs, ct)


def run(d, lab, v, th):
    return lib.evaluate(d['f'], d['R'][v] >= th, d['avail'], lab, r=1)[2]


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'TP_adj', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


def save(o):
    json.dump(o, open(OUTF, 'w'), indent=1, default=str)


V = {}
for s in VAL:
    t = time.time()
    V[s] = test1(s)
    log(f'validation seed {s}: {time.time() - t:.0f}s')
THETA, OUT = {}, {'meta': __doc__.strip(), 'validation': {}, 'test1': {}}
for v in VERSIONS:
    cands = []
    for th in TH:
        m = dict(TP=0, FN=0, FP=0, TN=0)
        for s in VAL:
            r = run(*V[s], v, th)
            for k in m:
                m[k] += r[k]
        m['TPR'], m['FPR'], m['fp_per_day'] = m['TP'] / (m['TP'] + m['FN']), m['FP'] / (m['FP'] + m['TN']), m['FP'] / (len(VAL) * DAYS)
        cands.append((th, m))
    th, m = max([x for x in cands if x[1]['fp_per_day'] <= BUDGET], key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    THETA[v] = th
    OUT['validation'][v] = {'theta': th, **{k: round(x, 4) if isinstance(x, float) else x for k, x in m.items()}}
    log(f'thresholds {v}: theta={th} val TP={m["TP"]}/40 FPR={m["FPR"]:.4f} FP/day={m["fp_per_day"]:.2f}')
save(OUT)
del V

for s in [TEST] + ROBUST:
    t = time.time()
    d, lab = test1(s)
    L = json.load(open(os.path.join(A_RUNS, f'seed{s}', 'metrics.json')))
    OUT['test1'][s] = {'Landauer': {'TP': L['TP'], 'FP': L['FP'], 'FPR': round(L['FP'] / (L['FP'] + L['TN']), 4)},
                       **{v: short(run(d, lab, v, THETA[v])) for v in VERSIONS}}
    del d
    save(OUT)
    r = OUT['test1'][s]
    log(f"seed {s}: Landauer {r['Landauer']['TP']} {r['Landauer']['FPR']:.4f} | " + ' | '.join(f"{v} {r[v]['TP']} {r[v]['FPR']:.4f}" for v in VERSIONS) + f' ({time.time() - t:.0f}s)')
OUT['totals_3_9'] = {k: sum(OUT['test1'][s][k]['TP'] for s in ROBUST) for k in ['Landauer'] + list(VERSIONS)}
try:
    from scipy.stats import wilcoxon
    for v in VERSIONS:
        OUT[f'wilcoxon_{v}_3_9'] = round(float(wilcoxon([OUT['test1'][s][v]['TP'] for s in ROBUST],
                                                        [OUT['test1'][s]['Landauer']['TP'] for s in ROBUST], alternative='greater').pvalue), 4)
except Exception as e:
    OUT['wilcoxon_error'] = str(e)
save(OUT)
log('totals 3-9', OUT['totals_3_9'], {k: v for k, v in OUT.items() if k.startswith('wilcoxon')})

ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(TEST, BASE, W30, ELIG)
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


T2 = {v: {'TP': 0, 'per': {}, 'apd': []} for v in VERSIONS}
for level in k3.LEVELS:
    t = time.time()
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
    labels = {p['victim']: p['day'] for p in PLANS}
    d = prepare(recs, {}, {}, trans=ChainMap(over, core.S['trans']))
    for v in VERSIONS:
        m = run(d, labels, v, THETA[v])
        T2[v]['TP'] += m['TP']; T2[v]['per'][level] = m['TP']; T2[v]['apd'].append(m['alerts_per_day'])
    del d, recs, over
    log(f'test2 {level}: {time.time() - t:.0f}s')
OUT['test2_seed1'] = {k: {'TP_of_60': v['TP'], 'per': v['per'], 'alerts_per_day': round(float(np.mean(v['apd'])), 3)} for k, v in T2.items()}
save(OUT)
log('test2', OUT['test2_seed1'])
log('DONE')
