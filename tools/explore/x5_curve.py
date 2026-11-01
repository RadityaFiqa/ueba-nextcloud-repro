"""Operating curve of the final model (x5_tune.py locked config: k5 split, burst F8>=2 w1, new folders w3,
w = .5/.4/.1): detections vs false-alarm budget. Config fixed; only the threshold moves.
For each budget B in {1.0, 1.75, 2.5, 3.5, 5.0} false alarms/day, theta is chosen on seeds 0+2 (uid swaps +
generated session hijacks) with the x5_tune rule: within budget, highest mean of (swap TPR, hijack TPR);
ties -> lower FPR. Reported on seed 1: uid swap (TP, TP_adj, FPR) and session hijack (TP, TP_adj per level).
  Output: results/x5_curve.json
  Run   : python -W ignore tools/x5_curve.py <landauer_runs_dir>      (~20 min)
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
FINAL = dict(split=True, thr=2, wb=1, wk4=3, w=(0.5, 0.4, 0.1))
BUDGETS = [1.0, 1.75, 2.5, 3.5, 5.0]
OUTF = os.path.join(config.RESULTS, 'x5_curve.json')
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



def full(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'TP_adj', 'FP', 'TPR', 'FPR', 'alerts_per_day')}


SW = {s: swap_data(s, [FINAL]) for s in VAL}
log('validation swaps ready')
HJ = {(s, lv): hijack_data(s, lv, [FINAL]) for s in VAL for lv in k3.LEVELS}
log('validation hijacks ready')
grid = []
for th in TH:
    m = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        r = run(*SW[s], FINAL, th)
        for k in m:
            m[k] += r[k]
    hj = sum(run(*HJ[key], FINAL, th)['TP'] for key in HJ)
    grid.append((th, {'swap_TP': m['TP'], 'hijack_TP': hj, 'score': (m['TP'] / 40 + hj / 120) / 2,
                      'FPR': m['FP'] / (m['FP'] + m['TN']), 'fp_per_day': m['FP'] / (len(VAL) * DAYS)}))
del SW, HJ
CHOSEN = {}
for B in BUDGETS:
    ok = [x for x in grid if x[1]['fp_per_day'] <= B]
    th, m = max(ok, key=lambda x: (x[1]['score'], -x[1]['FPR']))
    CHOSEN[B] = th
    log(f'budget {B}: theta={th} val swap {m["swap_TP"]}/40 hijack {m["hijack_TP"]}/120 FP/day {m["fp_per_day"]:.2f}')
OUT = {'meta': __doc__.strip(), 'config': cname(FINAL), 'validation_grid': [(th, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}) for th, m in grid],
       'chosen_theta': CHOSEN, 'seed1': {}}
save(OUT)
ths = sorted(set(CHOSEN.values()))
d, lab = swap_data(TEST, [FINAL])
sw = {th: full(run(d, lab, FINAL, th)) for th in ths}
del d
hj = {th: {} for th in ths}
for lv in k3.LEVELS:
    d, lab = hijack_data(TEST, lv, [FINAL])
    for th in ths:
        hj[th][lv] = full(run(d, lab, FINAL, th))
    del d
for B in BUDGETS:
    th = CHOSEN[B]
    OUT['seed1'][str(B)] = {'theta': th, 'swap': sw[th], 'hijack_TP': sum(hj[th][lv]['TP'] for lv in k3.LEVELS),
                            'hijack_TP_adj': sum(hj[th][lv]['TP_adj'] for lv in k3.LEVELS), 'hijack_per_level': {lv: hj[th][lv]['TP'] for lv in k3.LEVELS},
                            'hijack_alerts_per_day': round(float(np.mean([hj[th][lv]['alerts_per_day'] for lv in k3.LEVELS])), 3)}
    r = OUT['seed1'][str(B)]
    log(f"budget {B}: theta={th} | swap {r['swap']['TP']}/20 (adj {r['swap']['TP_adj']}) FPR {r['swap']['FPR']:.4f} {r['swap']['alerts_per_day']:.2f}/hari "
        f"| hijack {r['hijack_TP']}/60 (adj {r['hijack_TP_adj']}) {r['hijack_per_level']}")
save(OUT)
log('DONE')
