"""Exploration (post-hoc): "X4 + extra features + k6" WITHOUT peer_gain, so that every input is relative to
the assessed user's own history (peer_gain compared today with other users' 30-day profiles).
IF on relative counts (49) + F1..F8 + 6 extra features (|F1|, |F5|, hour_js, trans_js, trans_new, folder_jac);
a ranked against the user's own previous 30 days; flags k1 two-sided, k2..k5, k6 (trans_js >= 0.5, weight 2),
c = sum / 10; R = .5a + .4c + .1h, lambda 0.3; history W = 14 active days.
Threshold chosen on seeds 0+2 only:
  rule_budget  highest TPR with <= 1.75 false alarms/day (Landauer's budget)      <- main
  rule_tpr     lowest FPR with TPR >= 0.65                                         (for reference)
Reported on seed 1 and seeds 3-9 (uid swap, with Landauer per seed) and on Test 2 seed 1 (session hijack;
transitions of injected events added as one block in time order, as x4_refine.py).
  Output: results/x4k6_nopeer.json
  Run   : python -W ignore tools/x4k6_nopeer.py <landauer_runs_dir>      (~25 min)
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
OUTF = os.path.join(config.RESULTS, 'x4k6_nopeer.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}
core.init(BASE, TYPES, os.path.join(config.WORK, 'transitions.parquet'))
log('init done')


def prepare(recs, pairs, ct, trans=None):
    f = lib.compute_features(recs, W=W)
    E = core.extra_features(f, recs, pairs, ct, peer=False, trans=trans)[:, :6]
    X = np.nan_to_num(np.hstack([lib.relative_A(f, W=W), f['F'], E]).astype(np.float32))
    a = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    c = core.c_k6(f['F'], E)
    return {'f': f, 'avail': ~np.isnan(a) & ~f['first'], 'R': np.nan_to_num(lib.risk(f, a, c, (0.5, 0.4, 0.1), LAM), nan=-1)}


def test1(s):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    return prepare(lib.relabel(BASE, pairs, ct), pairs, ct), lib.labels_from_swaps(pairs, ct)


def run(d, lab, th):
    return lib.evaluate(d['f'], d['R'] >= th, d['avail'], lab, r=1)[2]


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'TP_adj', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


def save(o):
    json.dump(o, open(OUTF, 'w'), indent=1, default=str)


# ---------------------------------------------------------------- 1. thresholds on seeds 0+2
V = {}
for s in VAL:
    t = time.time()
    V[s] = test1(s)
    log(f'validation seed {s}: {time.time() - t:.0f}s')
cands = []
for th in TH:
    m = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        r = run(*V[s], th)
        for k in m:
            m[k] += r[k]
    m['TPR'], m['FPR'], m['fp_per_day'] = m['TP'] / (m['TP'] + m['FN']), m['FP'] / (m['FP'] + m['TN']), m['FP'] / (len(VAL) * DAYS)
    cands.append((th, m))
budget = max([x for x in cands if x[1]['fp_per_day'] <= BUDGET], key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
ok = [x for x in cands if x[1]['TPR'] >= 0.65]
tpr = min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(cands, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
RULES = {'rule_budget': budget[0], 'rule_tpr': tpr[0]}
OUT = {'meta': __doc__.strip(), 'thresholds': RULES,
       'validation': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in m.items()} for k, (th, m) in (('rule_budget', budget), ('rule_tpr', tpr))},
       'test1': {}}
log('thresholds', RULES, '| budget val', OUT['validation']['rule_budget'], '| tpr val', OUT['validation']['rule_tpr'])
save(OUT)
del V

# ---------------------------------------------------------------- 2. seed 1 and seeds 3-9
for s in [TEST] + ROBUST:
    t = time.time()
    d, lab = test1(s)
    L = json.load(open(os.path.join(A_RUNS, f'seed{s}', 'metrics.json')))
    OUT['test1'][s] = {'Landauer': {'TP': L['TP'], 'FP': L['FP'], 'FPR': round(L['FP'] / (L['FP'] + L['TN']), 4), 'alerts_per_day': round(L['FP'] / DAYS, 3)},
                       **{rule: short(run(d, lab, th)) for rule, th in RULES.items()}}
    del d
    save(OUT)
    r = OUT['test1'][s]
    log(f"seed {s}: Landauer {r['Landauer']['TP']} FPR {r['Landauer']['FPR']:.4f} | budget {r['rule_budget']['TP']} FPR {r['rule_budget']['FPR']:.4f} "
        f"| tpr {r['rule_tpr']['TP']} FPR {r['rule_tpr']['FPR']:.4f} ({time.time() - t:.0f}s)")
tot = {k: sum(OUT['test1'][s][k]['TP'] for s in ROBUST) for k in ['Landauer', 'rule_budget', 'rule_tpr']}
OUT['test1_seeds3_9_total'] = tot
try:
    from scipy.stats import wilcoxon
    for rule in RULES:
        x = [OUT['test1'][s][rule]['TP'] for s in ROBUST]
        y = [OUT['test1'][s]['Landauer']['TP'] for s in ROBUST]
        OUT[f'wilcoxon_{rule}_vs_Landauer_3_9'] = round(float(wilcoxon(x, y, alternative='greater').pvalue), 4)
except Exception as e:  # scipy missing or all ties
    OUT['wilcoxon_error'] = str(e)
save(OUT)
log('seeds 3-9 totals', tot, {k: v for k, v in OUT.items() if k.startswith('wilcoxon')})

# ---------------------------------------------------------------- 3. Test 2, seed 1
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


T2 = {rule: {'TP': 0, 'per': {}, 'apd': []} for rule in RULES}
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
    for rule, th in RULES.items():
        m = run(d, labels, th)
        T2[rule]['TP'] += m['TP']; T2[rule]['per'][level] = m['TP']; T2[rule]['apd'].append(m['alerts_per_day'])
    del d, recs, over
    log(f'test2 {level}: {time.time() - t:.0f}s')
OUT['test2_seed1'] = {k: {'TP_of_60': v['TP'], 'per': v['per'], 'alerts_per_day': round(float(np.mean(v['apd'])), 3)} for k, v in T2.items()}
save(OUT)
log('test2', OUT['test2_seed1'])
log('DONE')
