"""Exploration (post-hoc), pre-specified: refinements of "X4 + extra features + k6" (x4_extra_features.py),
all user-relative and WITHOUT any Landauer component. Everything is chosen on seeds 0+2 only; seed 1
and seeds 3-9 are computed only after the choice is locked; Test 2 (session hijack) uses the locked choice.

Base (as X4+feat+k): global IF (200 trees / 60 days / weekly) on relative Set A + F1..F8 + 7 extra
features, per-user calibration of a, flags k1 two-sided, k2..k5, k6 order changed (trans_js >= 0.5).
Grid (16 configurations x thresholds 0.40..0.99):
  w        risk weights (a, c, h) in {(.5,.4,.1), (.6,.3,.1), (.4,.5,.1), (.3,.6,.1)}, lambda = 0.3
  k5tight  k5 on F7 >= 0.2 OR F8 >= 5 (instead of 0.1 / 3)
  strong   alarm only if k2 OR k4 OR k6 also fires
Adaptation stays r = 1 (as Landauer), so every non-adaptation day is scored.
Selection rule (fixed in advance): highest validation TPR with <= 1.75 false alarms/day
(Landauer's budget); ties -> lower FPR, then simpler config (fewer changes from the base).
Test 2: transitions of injected events are added as one contiguous block in time order (approximation:
the injected session is not interleaved with the owner's events of that day).
  Output: results/x4_refine.json
  Run   : python -W ignore tools/x4_refine.py <landauer_runs_dir>      (~70 min)
"""
import bisect
import itertools
import json
import os
import sys
import time
from collections import ChainMap, Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_k3_attack as k3  # noqa: E402
import lib_models as lib  # noqa: E402
from step04_daily_tables import SENSITIVE  # noqa: E402

A_RUNS = sys.argv[1]
W, NPEER, M, DAYS = 14, 30, 5, 1910
LAM, BUDGET = 0.3, 1.75
VAL, TEST1, ROBUST = [0, 2], 1, [3, 4, 5, 6, 7, 8, 9]
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
WEIGHTS = [(0.5, 0.4, 0.1), (0.6, 0.3, 0.1), (0.4, 0.5, 0.1), (0.3, 0.6, 0.1)]
CONFIGS = [dict(w=w, k5tight=k, strong=s) for w, k, s in itertools.product(WEIGHTS, [False, True], [False, True])]
OUTF = os.path.join(config.RESULTS, 'x4_refine.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


def cname(c):
    return 'w=' + '/'.join(str(x) for x in c['w']) + ('+k5tight' if c['k5tight'] else '') + ('+strong' if c['strong'] else '')


def n_changes(c):
    return (c['w'] != WEIGHTS[0]) + c['k5tight'] + c['strong']


BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {t: i for i, t in enumerate(TYPES)}

# ---------------------------------------------------------------- transitions per original (uid, day)
cache = os.path.join(config.WORK, 'transitions.parquet')
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='3GB'")
TRANS = {}
for u, d, a, b, n in con.execute(f"SELECT uid, d, a, b, n FROM read_parquet('{cache}')").fetchall():
    TRANS.setdefault((u, d), {})[TIX[a] * NT + TIX[b]] = n
con.close()
log(f'transitions: {len(TRANS)} user-days')

# ---------------------------------------------------------------- peer profiles per date (original data)
PEERS = [u for u, days in BASE.items() if len(days) >= 28]
PIDX = {u: k for k, u in enumerate(PEERS)}
PD = {u: sorted(BASE[u]) for u in PEERS}
PC = {u: np.vstack([np.zeros(NT), np.cumsum(np.array([BASE[u][d].types for d in PD[u]], float), 0)]) for u in PEERS}
_prof_cache = {}


def peer_matrix(day):
    if day not in _prof_cache:
        P = np.zeros((len(PEERS), NT))
        for u in PEERS:
            k = bisect.bisect_left(PD[u], day)
            if k >= M:
                P[PIDX[u]] = PC[u][k] - PC[u][max(0, k - NPEER)]
        s = P.sum(1, keepdims=True)
        _prof_cache[day] = np.where(s > 0, P / np.where(s > 0, s, 1), np.nan)
    return _prof_cache[day]


def js_rows(p, Q):
    m = 0.5 * (p + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum(1)
        b = np.where(Q > 0, Q * np.log2(Q / m), 0).sum(1)
    return 0.5 * a + 0.5 * b


def js(p, q):
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return lib.js_div(p.astype(float), q.astype(float))


def extra_features(f, recs, pairs, ct, trans):
    """7 extra features aligned with compute_features' rows (as x4_extra_features.py)."""
    out = np.full((len(f['d']), 7), np.nan)
    row = 0
    for uid in sorted(recs):
        lst = recs[uid]
        partner = pairs.get(uid)
        own_days = BASE.get(uid, {})

        def orig(d, uid=uid, partner=partner, own_days=own_days):
            if d in own_days and (uid not in ct or d * 86400 < ct[uid]):
                return uid
            return partner
        for i, r in enumerate(lst):
            if i > 0:
                hist = lst[max(0, i - W):i]
                Fi = f['F'][row]
                hh = lambda x: np.asarray(x.hist, float).reshape(24, 12).sum(1)
                hour_js = js(hh(r), sum(hh(p) for p in hist))
                tt = trans.get((orig(r.d), r.d), {})
                th = Counter()
                for p in hist:
                    th.update(trans.get((orig(p.d), p.d), {}))
                keys = set(tt) | set(th)
                trans_js = js(np.array([tt.get(k, 0) for k in keys], float), np.array([th.get(k, 0) for k in keys], float)) if keys else 0.0
                trans_new = (sum(n for k, n in tt.items() if k not in th) / sum(tt.values())) if tt else 0.0
                ft = set(r.pref)
                fh = set().union(*[set(p.pref) for p in hist])
                folder_jac = (1 - len(ft & fh) / len(ft | fh)) if ft else 0.0
                today = np.asarray(r.types, float)
                own = np.array([p.types for p in lst[max(0, i - NPEER):i]], float).sum(0)
                own_js = js(today, own)
                P = peer_matrix(r.d)
                mask = ~np.isnan(P[:, 0])
                if uid in PIDX:
                    mask[PIDX[uid]] = False
                best = float(np.nanmin(js_rows(today / today.sum(), P[mask]))) if mask.any() and today.sum() > 0 else 1.0
                out[row] = (abs(Fi[0]), abs(Fi[4]), hour_js, trans_js, trans_new, folder_jac, own_js - best)
            row += 1
    return out


def flags(F, E, k5tight):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    k1 = g(np.abs(F[:, 0])) >= 3
    k2 = (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2)
    k3_ = g(F[:, 3]) >= 0.3
    k4 = g(F[:, 5]) >= 0.5
    k5 = ((g(F[:, 6]) >= 0.2) | (g(F[:, 7]) >= 5)) if k5tight else ((g(F[:, 6]) >= 0.1) | (g(F[:, 7]) >= 3))
    k6 = g(E[:, 3]) >= 0.5
    return k2 | k4 | k6, (2 * k1 + 2 * k2 + k3_ + 2 * k4 + k5 + 2 * k6) / 10.0


def prepare(recs, pairs, ct, trans, configs):
    """Scores of each configuration for one data set (relabelled or injected)."""
    f = lib.compute_features(recs, W=W)
    E = extra_features(f, recs, pairs, ct, trans)
    X = np.nan_to_num(np.hstack([lib.relative_A(f, W=W), f['F'], E]).astype(np.float32))
    a = lib.calibrate_per_user(f, lib.iforest_a(f, X, schedule='rolling'))
    out = {'f': f, 'avail': ~np.isnan(a) & ~f['first'], 'R': {}}
    for c in configs:
        strong, cc = flags(f['F'], E, c['k5tight'])
        out['R'][cname(c)] = (np.nan_to_num(lib.risk(f, a, cc, c['w'], LAM), nan=-1), strong if c['strong'] else None)
    _prof_cache.clear()
    return out


def run(d, lab, c, th):
    R, gate = d['R'][cname(c)]
    al = R >= th
    if gate is not None:
        al &= gate
    return lib.evaluate(d['f'], al, d['avail'], lab, r=1)[2]


def test1(s, configs):
    pairs, ct = lib.seed_swaps(config.STATS, s)
    d = prepare(lib.relabel(BASE, pairs, ct), pairs, ct, TRANS, configs)
    return d, lib.labels_from_swaps(pairs, ct)


def short(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'FP', 'TN', 'TPR', 'FPR', 'alerts_per_day')}


def save(out):
    json.dump(out, open(OUTF, 'w'), indent=1, default=str)


# ---------------------------------------------------------------- 1. choose on seeds 0+2 only
V = {}
for s in VAL:
    t = time.time()
    V[s] = test1(s, CONFIGS)
    log(f'validation seed {s}: {time.time() - t:.0f}s')
cands = []
for c in CONFIGS:
    for th in TH:
        m = dict(TP=0, FN=0, FP=0, TN=0)
        for s in VAL:
            r = run(*V[s], c, th)
            for k in m:
                m[k] += r[k]
        m['TPR'], m['FPR'] = m['TP'] / (m['TP'] + m['FN']), m['FP'] / (m['FP'] + m['TN'])
        m['fp_per_day'] = m['FP'] / (len(VAL) * DAYS)
        if m['fp_per_day'] <= BUDGET:
            cands.append((c, th, m))
best_c, best_th, best_m = max(cands, key=lambda x: (x[2]['TPR'], -x[2]['FPR'], -n_changes(x[0])))
base_c = CONFIGS[0]
base_th, base_m = max(((th, m) for c, th, m in cands if c == base_c), key=lambda x: (x[1]['TPR'], -x[1]['FPR']))
per_config = {}
for c, th, m in cands:
    k = cname(c)
    if k not in per_config or (m['TPR'], -m['FPR']) > (per_config[k]['TPR'], -per_config[k]['FPR']):
        per_config[k] = {'theta': th, 'TP': m['TP'], 'TPR': round(m['TPR'], 3), 'FPR': round(m['FPR'], 4), 'fp_per_day': round(m['fp_per_day'], 2)}
log(f'LOCKED: {cname(best_c)} theta={best_th} val TPR={best_m["TPR"]:.3f} FPR={best_m["FPR"]:.4f} FP/day={best_m["fp_per_day"]:.2f}')
log(f'base:   {cname(base_c)} theta={base_th} val TPR={base_m["TPR"]:.3f} FPR={base_m["FPR"]:.4f}')
OUT = {'meta': __doc__.strip(), 'locked': {'config': cname(best_c), 'theta': best_th,
       'validation': {k: round(v, 4) if isinstance(v, float) else v for k, v in best_m.items()}},
       'base': {'config': cname(base_c), 'theta': base_th}, 'validation_best_per_config': per_config}
save(OUT)
del V
KEEP = [best_c] if best_c == base_c else [best_c, base_c]
REP = [(cname(best_c), best_c, best_th)] + ([] if best_c == base_c else [('base ' + cname(base_c), base_c, base_th)])

# ---------------------------------------------------------------- 2. seed 1 (main), then seeds 3-9
L = {s: json.load(open(os.path.join(A_RUNS, f'seed{s}', 'metrics.json'))) for s in [TEST1] + ROBUST}
OUT['test1'] = {}
for s in [TEST1] + ROBUST:
    t = time.time()
    d, lab = test1(s, KEEP)
    lfpr = L[s]['FP'] / (L[s]['FP'] + L[s]['TN'])
    OUT['test1'][s] = {'Landauer': {'TP': L[s]['TP'], 'FP': L[s]['FP'], 'FPR': round(lfpr, 4), 'alerts_per_day': round(L[s]['FP'] / DAYS, 3)}}
    for key, c, th in REP:
        OUT['test1'][s][key] = short(run(d, lab, c, th))
    del d
    save(OUT)
    log(f'seed {s}: ' + ' | '.join(f"{k} TP={v['TP']} FPR={v['FPR']:.4f}" for k, v in OUT['test1'][s].items()) + f' ({time.time() - t:.0f}s)')

# ---------------------------------------------------------------- 3. Test 2 (session hijack), seed 1, locked config
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = k3.plan_attacks(TEST1, BASE, W30, ELIG)
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


T2 = {key: {'TP': 0, 'per': {}, 'apd': []} for key, _, _ in REP}
for level in k3.LEVELS:
    t = time.time()
    recs, over = dict(CLEAN), {}
    for p in PLANS:
        ev = sorted(k3.attack_events(p, level, TEST1, BASE, ALL_PREFIX), key=lambda e: e[0])
        lst = list(recs[p['victim']])
        i = next(j for j, r in enumerate(lst) if r.d == p['day'])
        lst[i] = inject(lst[i], ev)
        recs[p['victim']] = lst
        tr = Counter(TRANS.get((p['victim'], p['day']), {}))
        for (_, a, _), (_, b, _) in zip(ev, ev[1:]):
            tr[TIX[a] * NT + TIX[b]] += 1
        over[(p['victim'], p['day'])] = dict(tr)
    labels = {p['victim']: p['day'] for p in PLANS}
    d = prepare(recs, {}, {}, ChainMap(over, TRANS), KEEP)
    for key, c, th in REP:
        m = run(d, labels, c, th)
        T2[key]['TP'] += m['TP']; T2[key]['per'][level] = m['TP']; T2[key]['apd'].append(m['alerts_per_day'])
    del d, recs, over
    log(f'test2 {level}: {time.time() - t:.0f}s')
OUT['test2_seed1'] = {k: {'TP_of_60': v['TP'], 'per': v['per'], 'alerts_per_day': round(float(np.mean(v['apd'])), 3)} for k, v in T2.items()}
save(OUT)
log('test2', OUT['test2_seed1'])
log('DONE')
