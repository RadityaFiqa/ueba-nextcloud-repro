"""Additional exploration (post-hoc): where do the false alarms of model X4 come from, and which
changes lower them without losing detections?

Part A - sources of X4's false alarms (Test 1, seed 1, R >= 0.60): concentration per user, identity type,
         flags raised, calibrated vs raw IF score, overlap with Landauer's false alarms.
Part B - variants of X4 (everything else locked), thresholds chosen on seeds 0+2 (Test 1), reported on
         seed 1 for Test 1 (uid swap) and Test 2 (K3 session hijack, 60 sessions, no re-tuning):
  X4          R >= t                                   (reference)
  X4-gate     a_cal >= t_a AND c >= t_c                (gate of R4, on X4's scores)
  X4-Rgate    R >= t AND c >= t_c
  X4-strong   R >= t AND (k2 OR k4)
  X4-floor    R with a = min(a_cal, a_raw)             (calibration cannot lift an ordinary day)
  X4-noK1     c without k1
  X4-noK3K5   c without k3 and k5
Selection rules on validation: rule_tpr (lowest FPR with TPR >= 0.65) and rule_budget (highest TPR with
false alarms <= 1.75 per day).
  Output: results/fpr_reduction.json
  Run   : python -W ignore tools/fpr_reduction.py      (~40 min)
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

VAL, TEST, DAYS = [0, 2], 1, 1910
W_FEAT, W_W, LAM, BUDGET = 14, (0.5, 0.4, 0.1), 0.3, 1.75
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
TC = [0.125, 0.25, 0.375, 0.5]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}


def x4_scores(f):
    """Locked X4 pipeline -> dict with flags, c variants, a_raw, a_cal, avail."""
    X = np.hstack([lib.relative_A(f, W=W_FEAT), f['F']]).astype(np.float32)
    a_raw = lib.iforest_a(f, X, schedule='rolling')
    a_cal = lib.calibrate_per_user(f, a_raw)
    ks, c = lib.flags_c(f['F'])
    _, c_nok1 = lib.flags_c(f['F'], drop_flags=('k1',))
    _, c_nok35 = lib.flags_c(f['F'], drop_flags=('k3', 'k5'))
    avail = ~np.isnan(a_cal) & ~f['first']
    R = lambda a, cc: np.nan_to_num(lib.risk(f, a, cc, W_W, LAM), nan=-1)
    return {'f': f, 'ks': ks, 'c': c, 'a_raw': a_raw, 'a_cal': a_cal, 'avail': avail,
            'R': R(a_cal, c), 'R_floor': R(np.fmin(a_cal, a_raw), c),
            'R_nok1': R(a_cal, c_nok1), 'R_nok35': R(a_cal, c_nok35)}


def alert(d, v, p):
    """p = (t,) or (t, t_c)."""
    t = p[0]
    if v == 'X4':
        return d['R'] >= t
    if v == 'X4-gate':
        return (np.nan_to_num(d['a_cal'], nan=-1) >= t) & (d['c'] >= p[1])
    if v == 'X4-Rgate':
        return (d['R'] >= t) & (d['c'] >= p[1])
    if v == 'X4-strong':
        return (d['R'] >= t) & (d['ks']['k2'] | d['ks']['k4'])
    if v == 'X4-floor':
        return d['R_floor'] >= t
    if v == 'X4-noK1':
        return d['R_nok1'] >= t
    if v == 'X4-noK3K5':
        return d['R_nok35'] >= t


VARS = {'X4': False, 'X4-gate': True, 'X4-Rgate': True, 'X4-strong': False,
        'X4-floor': False, 'X4-noK1': False, 'X4-noK3K5': False}


def run(d, lab, v, p, full=False):
    st, dc, m = lib.evaluate(d['f'], alert(d, v, p), d['avail'], lab, r=1)
    return (st, dc, m) if full else m


# ---------------------------------------------------------------- Test 1 data
T1 = {}
for s in VAL + [TEST]:
    t = time.time()
    pairs, ct = lib.seed_swaps(config.STATS, s)
    f = lib.compute_features(lib.relabel(BASE, pairs, ct), W=W_FEAT)
    T1[s] = (x4_scores(f), lib.labels_from_swaps(pairs, ct))
    log(f'test1 seed {s}: {time.time() - t:.0f}s')

OUT = {'meta': __doc__.strip()}

# ---------------------------------------------------------------- Part A: sources of false alarms
d, lab = T1[TEST]
f = d['f']
st, dc, m = run(d, lab, 'X4', (0.60,), full=True)
det = st == 'detection'
pos = np.array([lab.get(u) == x for u, x in zip(f['uid'], f['d'])])
fp = det & (dc == 'anomalous') & ~pos
tn = det & (dc == 'normal') & ~pos
con = duckdb.connect()
utype = dict(con.execute(f"SELECT uid, any_value(uidType) FROM read_parquet('{config.PARQUET}') GROUP BY 1").fetchall())
per_user = Counter(f['uid'][fp])
top = per_user.most_common()
nfp = int(fp.sum())
share = lambda n: round(sum(c for _, c in top[:n]) / nfp, 3)
users_det = Counter(f['uid'][det & ~pos])
# Landauer decisions (seed 1 dump) for overlap
A = {}
for line in open(config.A_DUMP):
    u, x, *rest = json.loads(line)
    A[(u, x)] = rest
a_fp = np.array([A.get((u, x), ['', ''])[0] == 'detection' and A.get((u, x), ['', ''])[-1] == 'anomalous'
                 for u, x in zip(f['uid'], f['d'])])
tot_ev = f['A'].sum(1)
partA = {
    'false_alarms': nfp, 'users_with_fp': len(per_user), 'users_scored': len(users_det),
    'share_top1': share(1), 'share_top5': share(5), 'share_top10': share(10), 'share_top20': share(20),
    'top10': [{'uid': u, 'type': utype.get(u), 'fp': c, 'days_scored': users_det[u],
               'fp_rate': round(c / users_det[u], 3)} for u, c in top[:10]],
    'fp_by_uidtype': dict(Counter(utype.get(u) for u in f['uid'][fp])),
    'flag_rate_fp': {k: round(float(d['ks'][k][fp].mean()), 3) for k in lib.FLAGS},
    'flag_rate_tn': {k: round(float(d['ks'][k][tn].mean()), 3) for k in lib.FLAGS},
    'c_zero_share_fp': round(float((d['c'][fp] == 0).mean()), 3),
    'a_cal_ge_095_share_fp': round(float((d['a_cal'][fp] >= 0.95).mean()), 3),
    'a_raw_median_fp': round(float(np.nanmedian(d['a_raw'][fp])), 3),
    'a_raw_median_tn': round(float(np.nanmedian(d['a_raw'][tn])), 3),
    'a_raw_lt_05_share_fp': round(float((d['a_raw'][fp] < 0.5).mean()), 3),
    'events_median_fp': float(np.median(tot_ev[fp])), 'events_median_tn': float(np.median(tot_ev[tn])),
    'also_landauer_fp_share': round(float(a_fp[fp].mean()), 3),
}
OUT['partA_sources_X4_test1_seed1'] = partA
log('PART A', json.dumps(partA, indent=1))
json.dump(OUT, open(os.path.join(config.RESULTS, 'fpr_reduction.json'), 'w'), indent=1, default=str)

# ---------------------------------------------------------------- Part B: choose thresholds on validation
def pooled(v, p):
    t = dict(TP=0, FN=0, FP=0, TN=0)
    for s in VAL:
        mm = run(*T1[s], v, p)
        for k in t:
            t[k] += mm[k]
    t['TPR'] = t['TP'] / (t['TP'] + t['FN'])
    t['FPR'] = t['FP'] / (t['FP'] + t['TN'])
    t['fp_per_day'] = t['FP'] / (len(VAL) * DAYS)
    return t


CHOSEN = {}
for v, two in VARS.items():
    grid = [(t, tc) for t in TH for tc in TC] if two else [(t,) for t in TH]
    c = [(p, pooled(v, p)) for p in grid]
    ok = [x for x in c if x[1]['TPR'] >= 0.65]
    r1 = min(ok, key=lambda x: (x[1]['FPR'], -x[1]['TPR'])) if ok else max(c, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    inb = [x for x in c if x[1]['fp_per_day'] <= BUDGET]
    r2 = max(inb, key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
    CHOSEN[v] = {'rule_tpr': r1, 'rule_budget': r2}
    log(f'{v}: rule_tpr {r1[0]} val TPR {r1[1]["TPR"]:.3f} FP/day {r1[1]["fp_per_day"]:.2f} | '
        f'rule_budget {r2[0]} val TPR {r2[1]["TPR"]:.3f} FP/day {r2[1]["fp_per_day"]:.2f}')


def short(mm):
    return {k: (round(mm[k], 4) if isinstance(mm[k], float) else mm[k])
            for k in ('TP', 'FN', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


RES = {v: {r: {'params': CHOSEN[v][r][0], 'validation': {k: round(x, 4) if isinstance(x, float) else x
                                                            for k, x in CHOSEN[v][r][1].items()},
               'test1_seed1': short(run(*T1[TEST], v, CHOSEN[v][r][0]))} for r in CHOSEN[v]} for v in VARS}
del T1

# ---------------------------------------------------------------- Test 2 (K3 session hijack), seed 1
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


T2 = {v: {r: {'TP': 0, 'FP': 0, 'alerts_per_day': []} for r in CHOSEN[v]} for v in VARS}
for level in k3.LEVELS:
    t = time.time()
    recs = dict(CLEAN)
    for p in PLANS:
        ev = k3.attack_events(p, level, TEST, BASE, ALL_PREFIX)
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
    labels = {p['victim']: p['day'] for p in PLANS}
    dd = x4_scores(lib.compute_features(recs, W=W_FEAT))
    for v in VARS:
        for r in CHOSEN[v]:
            mm = run(dd, labels, v, CHOSEN[v][r][0])
            T2[v][r]['TP'] += mm['TP']; T2[v][r]['FP'] += mm['FP']; T2[v][r]['alerts_per_day'].append(mm['alerts_per_day'])
    log(f'test2 {level}: {time.time() - t:.0f}s')
for v in VARS:
    for r in CHOSEN[v]:
        x = T2[v][r]
        RES[v][r]['test2_seed1'] = {'TP_of_60': x['TP'], 'alerts_per_day_mean': round(float(np.mean(x['alerts_per_day'])), 3)}
OUT['partB_variants'] = RES
json.dump(OUT, open(os.path.join(config.RESULTS, 'fpr_reduction.json'), 'w'), indent=1, default=str)
for v in VARS:
    for r in ('rule_tpr', 'rule_budget'):
        x = RES[v][r]
        log(f"{v:10s} {r:11s} {x['params']}: T1 TP={x['test1_seed1']['TP']}/20 FPR={x['test1_seed1']['FPR']:.4f} "
            f"alerts/day={x['test1_seed1']['alerts_per_day']:.2f} | T2 TP={x['test2_seed1']['TP_of_60']}/60 "
            f"alerts/day={x['test2_seed1']['alerts_per_day_mean']:.2f}")
log('DONE')
