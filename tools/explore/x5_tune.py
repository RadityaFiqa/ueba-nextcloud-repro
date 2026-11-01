"""Exploration (post-hoc), pre-specified: adjust the burst / new-folder flags of version 5 (X4 + k1 two-sided + k6),
choosing on validation data of BOTH attacks (seeds 0+2: uid swaps and newly generated session hijacks), then
testing on seed 1 (both attacks) and seeds 3-9 (uid swap). IF inputs unchanged (57, all user-relative).
Flags: k1 |F1|>=3 (w2), k2 (w2), k3 (w1), k4 F6>=0.5 (w = wk4), k6 trans_js>=0.5 (w2), and either
  combined k5: F7>=0.1 or F8>=3 (w1)                                    split = False
  split    k5a: F7>=0.1 (w1)  +  k5b: F8>=burst_thr (w = wb)             split = True
c = sum of raised weights / sum of all weights;  R = wa*a + wc*c + wh*h, lambda 0.3.
Grid: split x burst_thr {2,3} x wb {1,2} x wk4 {2,3} x (wa,wc,wh) {(.5,.4,.1),(.4,.5,.1)}  (+ unsplit x wk4 x w), theta 0.40..0.99.
Selection rule (fixed in advance): false alarms <= 1.75/day on the validation uid-swap data (Landauer's budget),
then highest mean of (swap TPR, hijack TPR); ties -> lower FPR, then fewer changes from version 5.
  Output: results/x5_tune.json
  Run   : python -W ignore tools/x5_tune.py <landauer_runs_dir>      (~45 min)
"""
import itertools
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
RW = [(0.5, 0.4, 0.1), (0.4, 0.5, 0.1)]
CONFIGS = [dict(split=False, thr=3, wb=1, wk4=k, w=w) for k in (2, 3) for w in RW] + \
          [dict(split=True, thr=t, wb=b, wk4=k, w=w) for t, b, k, w in itertools.product((3, 2), (1, 2), (2, 3), RW)]
V5 = CONFIGS[0]
OUTF = os.path.join(config.RESULTS, 'x5_tune.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def cname(c):
    s = 'k5 dipecah, F8>=%d bobot %d' % (c['thr'], c['wb']) if c['split'] else 'k5 gabung'
    return f"{s}; k4 bobot {c['wk4']}; w={'/'.join(str(x) for x in c['w'])}"


def n_changes(c):
    return int(c['split']) + (c['split'] and c['thr'] != 3) + (c['split'] and c['wb'] != 1) + (c['wk4'] != 2) + (c['w'] != RW[0])


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}
core.init(BASE, TYPES, os.path.join(config.WORK, 'transitions.parquet'))


def c_of(F, E, cfg):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    ks = [g(np.abs(F[:, 0])) >= 3, (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2), g(F[:, 3]) >= 0.3, g(F[:, 5]) >= 0.5, g(E[:, 3]) >= 0.5]
    w = [2, 2, 1, cfg['wk4'], 2]
    if cfg['split']:
        ks += [g(F[:, 6]) >= 0.1, g(F[:, 7]) >= cfg['thr']]; w += [1, cfg['wb']]
    else:
        ks += [(g(F[:, 6]) >= 0.1) | (g(F[:, 7]) >= 3)]; w += [1]
    return (np.column_stack(ks) * np.array(w)).sum(1) / float(sum(w))


def prepare(recs, pairs, ct, configs, trans=None):
    f = lib.compute_features(recs, W=W)
    E = core.extra_features(f, recs, pairs, ct, peer=False, trans=trans)[:, :6]
    X = np.nan_to_num(np.hstack([lib.relative_A(f, W=W), f['F']]).astype(np.float32))
    a = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    R = {cname(c): np.nan_to_num(lib.risk(f, a, c_of(f['F'], E, c), c['w'], LAM), nan=-1) for c in configs}
    return {'f': f, 'avail': ~np.isnan(a) & ~f['first'], 'R': R}


def swap_data(s, configs):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    return prepare(lib.relabel(BASE, pairs, ct), pairs, ct, configs), lib.labels_from_swaps(pairs, ct)


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
CLEAN = lib.relabel(BASE, {}, {})


def hijack_data(s, level, configs):
    plans = k3.plan_attacks(s, BASE, W30, ELIG)
    recs, over = dict(CLEAN), {}
    for p in plans:
        ev = sorted(k3.attack_events(p, level, s, BASE, ALL_PREFIX), key=lambda e: e[0])
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
        tr = Counter(core.S['trans'].get((p['victim'], p['day']), {}))
        for (_, a, _), (_, b, _) in zip(ev, ev[1:]):
            tr[TIX[a] * NT + TIX[b]] += 1
        over[(p['victim'], p['day'])] = dict(tr)
    return prepare(recs, {}, {}, configs, trans=ChainMap(over, core.S['trans'])), {p['victim']: p['day'] for p in plans}


def run(d, lab, c, th):
    return lib.evaluate(d['f'], d['R'][cname(c)] >= th, d['avail'], lab, r=1)[2]


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FP', 'TPR', 'FPR', 'alerts_per_day')}


def save(o):
    json.dump(o, open(OUTF, 'w'), indent=1, default=str)


# ---------------------------------------------------------------- 1. validation: swaps + hijacks on seeds 0 and 2
SW, HJ = {}, {}
for s in VAL:
    t = time.time()
    SW[s] = swap_data(s, CONFIGS)
    for level in k3.LEVELS:
        HJ[(s, level)] = hijack_data(s, level, CONFIGS)
    log(f'validation seed {s}: {time.time() - t:.0f}s')
res = []
for c in CONFIGS:
    for th in TH:
        m = dict(TP=0, FN=0, FP=0, TN=0)
        for s in VAL:
            r = run(*SW[s], c, th)
            for k in m:
                m[k] += r[k]
        fpd = m['FP'] / (len(VAL) * DAYS)
        if fpd > BUDGET:
            continue
        hj = sum(run(*HJ[key], c, th)['TP'] for key in HJ)
        tpr_s, tpr_h = m['TP'] / (m['TP'] + m['FN']), hj / (20 * len(HJ))
        res.append((c, th, {'swap_TP': m['TP'], 'swap_TPR': tpr_s, 'hijack_TP': hj, 'hijack_TPR': tpr_h,
                            'score': (tpr_s + tpr_h) / 2, 'FPR': m['FP'] / (m['FP'] + m['TN']), 'fp_per_day': fpd}))
best_c, best_th, best_m = max(res, key=lambda x: (x[2]['score'], -x[2]['FPR'], -n_changes(x[0])))
v5_th, v5_m = max(((th, m) for c, th, m in res if c == V5), key=lambda x: (x[1]['score'], -x[1]['FPR']))
per_cfg = {}
for c, th, m in res:
    k = cname(c)
    if k not in per_cfg or (m['score'], -m['FPR']) > (per_cfg[k]['score'], -per_cfg[k]['FPR']):
        per_cfg[k] = {'theta': th, **{kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in m.items()}}
OUT = {'meta': __doc__.strip(), 'locked': {'config': cname(best_c), 'theta': best_th, **{k: round(v, 4) if isinstance(v, float) else v for k, v in best_m.items()}},
       'version5_same_rule': {'config': cname(V5), 'theta': v5_th, **{k: round(v, 4) if isinstance(v, float) else v for k, v in v5_m.items()}},
       'validation_per_config': per_cfg, 'test': {}}
log('LOCKED', OUT['locked'])
log('V5    ', OUT['version5_same_rule'])
save(OUT)
del SW, HJ

# ---------------------------------------------------------------- 2. test on seed 1 only (both attacks), every config
# Each config keeps the threshold it got on validation; the locked config is marked. Seeds 3-9 are left for later.
THETA = {cname(c): per_cfg[cname(c)]['theta'] for c in CONFIGS if cname(c) in per_cfg}
CFG = [c for c in CONFIGS if cname(c) in THETA]
L = json.load(open(os.path.join(A_RUNS, f'seed{TEST}', 'metrics.json')))
d, lab = swap_data(TEST, CFG)
OUT['test']['Landauer swap seed1'] = {'TP': L['TP'], 'FPR': round(L['FP'] / (L['FP'] + L['TN']), 4), 'alerts_per_day': round(L['FP'] / DAYS, 3)}
for c in CFG:
    OUT['test'].setdefault(cname(c), {})['swap'] = short(run(d, lab, c, THETA[cname(c)]))
del d
save(OUT)
log('seed 1 swap done')
for level in k3.LEVELS:
    t = time.time()
    d, lab = hijack_data(TEST, level, CFG)
    for c in CFG:
        OUT['test'][cname(c)][f'hijack {level}'] = run(d, lab, c, THETA[cname(c)])['TP']
    del d
    save(OUT)
    log(f'hijack {level}: {time.time() - t:.0f}s')
for c in CFG:
    r = OUT['test'][cname(c)]
    r['hijack_total'] = sum(r[f'hijack {lv}'] for lv in k3.LEVELS)
    mark = ' <== TERKUNCI' if c == best_c else (' <== versi 5' if c == V5 else '')
    log(f"{cname(c):45s} theta={THETA[cname(c)]:.2f} | swap {r['swap']['TP']}/20 FPR {r['swap']['FPR']:.4f} {r['swap']['alerts_per_day']:.2f}/hari "
        f"| hijack {r['hijack_total']}/60 ({r['hijack rendah']}/{r['hijack sedang']}/{r['hijack tinggi']}){mark}")
save(OUT)
log('DONE')
