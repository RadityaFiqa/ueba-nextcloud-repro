"""Run every scenario of Skenario_Pengujian.xlsx except K3 (approved 2026-10-22).

Protocol: single reported run = seed 1 (all 5,389 users). All parameters and
thresholds are chosen on validation seeds 0 and 2 only (seed 1 is the test
run, so it is excluded from tuning), then locked and applied to seed 1.

Threshold rule (P4): lowest pooled FPR among thresholds with pooled TPR >= 0.65;
if none reaches 0.65, the highest TPR (ties -> lowest FPR).
Config choice in P1-P3 uses the same rule on each config's selected threshold.
"""
import sys
import config  # data / work / results paths (see config.py)
import json
import os
import random
import subprocess
import time

import numpy as np
from scipy.stats import binomtest

import lib

STATS = config.STATS
OUT = config.RESULTS
VAL, TEST = [0, 2], 1
THETA_A = [round(x, 3) for x in np.arange(0.50, 0.995, 0.01)] + [0.995, 0.999]
THETA_R = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
THETA_C = [k / 8 for k in range(1, 9)]
os.makedirs(OUT, exist_ok=True)
LOG = open(f'{OUT}/run_all.log', 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n'); LOG.flush()


T0 = time.time()
BASE, TYPES = lib.load_base(config.BASE)
SEEDS, FEAT, ASC = {}, {}, {}


def seed_data(seed):
    if seed not in SEEDS:
        pairs, ct = lib.seed_swaps(STATS, seed)
        SEEDS[seed] = (lib.relabel(BASE, pairs, ct), lib.labels_from_swaps(pairs, ct), pairs, ct)
    return SEEDS[seed]


def feat(seed, W):
    if (seed, W) not in FEAT:
        FEAT[(seed, W)] = lib.compute_features(seed_data(seed)[0], W=W, M=5)
    return FEAT[(seed, W)]


def ascore(seed, W, colset, schedule, fresh=False):
    key = (seed, W, json.dumps(colset), schedule)
    if fresh or key not in ASC:
        t = time.time()
        f = feat(seed, W)
        a = lib.iforest_a(f, lib.design_matrix(f, colset), schedule=schedule)
        log(f'  IF seed={seed} W={W} cols={colset} sched={schedule}: {time.time() - t:.0f}s')
        if fresh:
            return a
        ASC[key] = a
    return ASC[key]


DEFAULT = dict(kind='risk', cols='AF', W=30, sched='rolling', w=(0.6, 0.3, 0.1), lam=0.5, r=1,
               thr_scale=None, drop_flags=())


def V(**kw):
    v = dict(DEFAULT)
    v.update(kw)
    return v


def scores(seed, v):
    """-> (score array or (a, c) for gate, avail)"""
    f = feat(seed, v['W'])
    a = ascore(seed, v['W'], v['cols'], v['sched'])
    avail = ~np.isnan(a) & ~f['first']
    if v['kind'] == 'if':
        return a, avail
    _, c = lib.flags_c(f['F'], v['thr_scale'], v['drop_flags'])
    if v['kind'] == 'gate':
        return (a, c), avail
    return lib.risk(f, a, c, v['w'], v['lam']), avail


def run(seed, v, theta):
    f = feat(seed, v['W'])
    labels = seed_data(seed)[1]
    sc, avail = scores(seed, v)
    if v['kind'] == 'gate':
        a, c = sc
        alert = (np.nan_to_num(a, nan=-1) >= theta[0]) & (c >= theta[1])
    else:
        alert = np.nan_to_num(sc, nan=-1) >= theta
    status, dec, m = lib.evaluate(f, alert, avail, labels, r=v['r'])
    return m, status, dec


def grid(v):
    if v['kind'] == 'if':
        return THETA_A
    if v['kind'] == 'gate':
        return [(ta, tc) for ta in THETA_A for tc in THETA_C]
    return THETA_R


def pooled(v, theta, seeds):
    tot = dict(TP=0, FN=0, FP=0, TN=0, TP_adj=0, FN_adj=0)
    for s in seeds:
        m, _, _ = run(s, v, theta)
        for k in tot:
            tot[k] += m[k]
    tot['TPR'] = tot['TP'] / max(1, tot['TP'] + tot['FN'])
    tot['TPR_adj'] = tot['TP_adj'] / max(1, tot['TP_adj'] + tot['FN_adj'])
    tot['FPR'] = tot['FP'] / max(1, tot['FP'] + tot['TN'])
    return tot


def choose(cands):
    """cands: list of (key, pooled metrics). Rule P4."""
    ok = [c for c in cands if c[1]['TPR'] >= 0.65]
    if ok:
        return min(ok, key=lambda c: (c[1]['FPR'], -c[1]['TPR']))
    return max(cands, key=lambda c: (c[1]['TPR'], -c[1]['FPR']))


def select_theta(v, name):
    cands = [(th, pooled(v, th, VAL)) for th in grid(v)]
    th, m = choose(cands)
    log(f'  [{name}] val theta={th} pooled TPR={m["TPR"]:.3f} FPR={m["FPR"]:.4f}')
    return th, m, cands


R = {'meta': {'protocol': 'single reported run seed 1; tuning on seeds 0 and 2',
              'theta_rule': 'min pooled FPR s.t. pooled TPR>=0.65, else max TPR'}}


def save():
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1, default=str)


# ---------------------------------------------------------------- P1-P4 tuning (validation seeds)
log('=== P1 history length W')
P1 = {}
for W in (14, 30, 60):
    th, m, _ = select_theta(V(W=W), f'P1 W={W}')
    P1[W] = (th, m)
W_LOCK = choose([(W, m) for W, (th, m) in P1.items()])[0]
R['P1'] = {'candidates': {W: {'theta': th, **m} for W, (th, m) in P1.items()}, 'locked_W': W_LOCK}
log('locked W =', W_LOCK); save()

log('=== P2 weights')
P2 = {}
for w in [(0.6, 0.3, 0.1), (0.5, 0.4, 0.1), (0.7, 0.2, 0.1), (0.5, 0.3, 0.2)]:
    th, m, _ = select_theta(V(W=W_LOCK, w=w), f'P2 w={w}')
    P2[w] = (th, m)
W_W = choose([(w, m) for w, (th, m) in P2.items()])[0]
R['P2'] = {'candidates': {str(w): {'theta': th, **m} for w, (th, m) in P2.items()}, 'locked_weights': W_W}
log('locked weights =', W_W); save()

log('=== P3 decay lambda')
P3 = {}
for lam in (0.3, 0.5, 0.7):
    th, m, _ = select_theta(V(W=W_LOCK, w=W_W, lam=lam), f'P3 lam={lam}')
    P3[lam] = (th, m)
LAM = choose([(lam, m) for lam, (th, m) in P3.items()])[0]
R['P3'] = {'candidates': {lam: {'theta': th, **m} for lam, (th, m) in P3.items()}, 'locked_lambda': LAM}
log('locked lambda =', LAM); save()

D_CFG = dict(W=W_LOCK, w=W_W, lam=LAM)
log('=== P4 theta_R curve')
th_D, m_D, cands = select_theta(V(**D_CFG), 'P4 D')
R['P4'] = {'locked_theta_R': th_D,
           'validation_curve': {str(th): m for th, m in cands},
           'test_curve_seed1': {str(th): run(TEST, V(**D_CFG), th)[0] for th in THETA_R}}
save()

log('=== P5 flag thresholds +-20% (validation, locked theta)')
P5 = {'base': pooled(V(**D_CFG), th_D, VAL)}
for name, (wt, conds) in lib.FLAGS.items():
    for fi, t in conds:
        for fac in (0.8, 1.2):
            P5[f'{name}:{lib.FNAMES[fi]}x{fac}'] = pooled(V(**D_CFG, thr_scale={(name, fi): fac}), th_D, VAL)
R['P5'] = P5; save()
log(f'P5 FPR range: {min(m["FPR"] for m in P5.values()):.4f}-{max(m["FPR"] for m in P5.values()):.4f}, '
    f'TPR range: {min(m["TPR"] for m in P5.values()):.3f}-{max(m["TPR"] for m in P5.values()):.3f}')

# ---------------------------------------------------------------- variants: tune on val, test on seed 1
VARIANTS = {
    'E2_B': V(kind='if', cols='A', **{'W': W_LOCK}),
    'E3_B+RS': V(cols='A', **D_CFG),
    'E4_C': V(kind='if', cols='AF', **{'W': W_LOCK}),
    'E5_D': V(**D_CFG),
    'R1_D-h': V(**{**D_CFG, 'w': (W_W[0], W_W[1], 0.0)}),
    'R2_D-c': V(**{**D_CFG, 'w': (W_W[0], 0.0, W_W[2])}),
    'R3_D-a': V(**{**D_CFG, 'w': (0.0, W_W[1], W_W[2])}),
    'R4_D-gate': V(kind='gate', **{'W': W_LOCK}),
    'F1_C-Vol': V(kind='if', cols=['AF', [0]], **{'W': W_LOCK}),
    'F2_C-Mix': V(kind='if', cols=['AF', [1, 2]], **{'W': W_LOCK}),
    'F3_C-Time': V(kind='if', cols=['AF', [3, 4]], **{'W': W_LOCK}),
    'F4_C-Res': V(kind='if', cols=['AF', [5]], **{'W': W_LOCK}),
    'F5_C-Sens': V(kind='if', cols=['AF', [6]], **{'W': W_LOCK}),
    'F6_C-Burst': V(kind='if', cols=['AF', [7]], **{'W': W_LOCK}),
    **{f'F7_D-{k}': V(**D_CFG, drop_flags=(k,)) for k in lib.FLAGS},
    'D1_D-static': V(**D_CFG, sched='static'),
    'D2_D-daily': V(**D_CFG, sched='daily'),
    'D3_D-noUserMode': V(**D_CFG, r=0),
}
R['variants'] = {}
DEC = {}
for name, v in VARIANTS.items():
    log(f'=== {name}')
    t = time.time()
    if name == 'E5_D':
        th, mval = th_D, m_D
    else:
        th, mval, _ = select_theta(v, name)
    m, status, dec = run(TEST, v, th)
    sc, _ = scores(TEST, v)
    if v['kind'] != 'gate':
        m['PR_AUC'] = lib.pr_auc(feat(TEST, v['W']), sc, seed_data(TEST)[1])
    R['variants'][name] = {'theta': th, 'validation': mval, 'test_seed1': m,
                           'config': {k: v[k] for k in v if k != 'thr_scale'}, 'seconds': round(time.time() - t)}
    DEC[name] = (status, dec)
    log(f'  TEST seed1: TP={m["TP"]} FN={m["FN"]} FP={m["FP"]} TN={m["TN"]} TPR={m["TPR"]:.3f} '
        f'TPR_adj={m["TPR_adj"]:.3f} FPR={m["FPR"]:.4f} R={m["R"]:.4f} alerts/day={m["alerts_per_day"]:.2f}')
    save()

# ---------------------------------------------------------------- E1 baseline A (seed 1) + sweep
log('=== E1 baseline A')
A = {}
for line in open(config.A_DUMP):
    u, d, *det = json.loads(line)
    A[(u, d)] = det
fT = feat(TEST, W_LOCK)
labT = seed_data(TEST)[1]
statA = np.array([A.get((u, d), ['', ''])[0] for u, d in zip(fT['uid'], fT['d'])], dtype=object)
decA = np.array([A.get((u, d), ['', ''])[-1] for u, d in zip(fT['uid'], fT['d'])], dtype=object)
mA = lib.metrics(fT['uid'], fT['d'], statA, decA, labT)
sweep = {}
for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    out = os.path.join(config.WORK, 'A', f'sweep_t{th}')
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'duck_seed.py'), '--parquet', config.PARQUET,
                    '--stats', STATS, '--outdir', out, '-s', '1', '-t', str(th)], check=True, capture_output=True)
    mm = json.load(open(f'{out}/metrics.json'))
    sweep[th] = {k: mm[k] for k in ('TP', 'FN', 'FP', 'TN', 'TPR', 'TPR_adj', 'FPR', 'R', 'Prec')}
    sweep[th]['alerts_per_day'] = (mm['TP'] + mm['FP']) / 1910
R['E1_A'] = {'theta': 0.6, 'test_seed1': mA, 'theta_sweep_seed1': sweep}
DEC['A'] = (statA, decA)
log(f'  A: TP={mA["TP"]} FP={mA["FP"]} TPR={mA["TPR"]:.3f} FPR={mA["FPR"]:.4f} alerts/day={mA["alerts_per_day"]:.2f}')
save()

# ---------------------------------------------------------------- E6 statistics (seed 1)
log('=== E6 McNemar + bootstrap')


def mcnemar(x, y):
    sx, dx = DEC[x]; sy, dy = DEC[y]
    b = c = 0
    for i in range(len(fT['d'])):
        if sx[i] == 'detection' and sy[i] == 'detection' and labT.get(fT['uid'][i]) != fT['d'][i]:
            ax, ay = dx[i] == 'anomalous', dy[i] == 'anomalous'
            b += ax and not ay
            c += ay and not ax
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {'pairs': f'{x} vs {y}', f'false_alarms_only_{x}': b, f'false_alarms_only_{y}': c, 'p_value_exact': p}


def boot_dfpr(x, y, B=2000, seed=0):
    per = {}
    for name, sgn in ((x, 1), (y, -1)):
        s, dd = DEC[name]
        for i in range(len(fT['d'])):
            if s[i] == 'detection' and labT.get(fT['uid'][i]) != fT['d'][i]:
                e = per.setdefault(fT['uid'][i], [0, 0, 0, 0])
                j = 0 if sgn == 1 else 2
                e[j] += dd[i] == 'anomalous'; e[j + 1] += 1
    users = list(per)
    M = np.array([per[u] for u in users], float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        S = M[rng.integers(0, len(users), len(users))].sum(0)
        diffs.append(S[0] / S[1] - S[2] / S[3])
    S = M.sum(0)
    return {'dFPR': S[0] / S[1] - S[2] / S[3], 'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


R['E6'] = {}
for x, y in (('E5_D', 'A'), ('E5_D', 'E4_C'), ('E4_C', 'E2_B')):
    R['E6'][f'{x} vs {y}'] = {**mcnemar(x, y), **boot_dfpr(x, y)}
    log(' ', R['E6'][f'{x} vs {y}'])
save()

# ---------------------------------------------------------------- K1/K2 + H1 per-victim table
log('=== H1 / K1 / K2')
_, labT, pairsT, ctT = seed_data(TEST)
users_stats = lib.load_users(STATS)


def sim(a, b):
    ua, ub = users_stats[a], users_stats[b]
    ra, rb = ua['cnt'] / ua['days'], ub['cnt'] / ub['days']
    return round(0.3 * min(ra, rb) / max(ra, rb) + 0.7 * len(ua['actions'] & ub['actions']) / len(ua['actions'] | ub['actions']), 3), max(ra, rb) / min(ra, rb)


idx = {(u, d): i for i, (u, d) in enumerate(zip(fT['uid'], fT['d']))}
rows = []
for victim, t in ctT.items():
    ident = pairsT[victim]
    i = idx.get((ident, int(t) // 86400))
    s, vr = sim(victim, ident)
    def det(name):
        st, dc = DEC[name]
        return None if i is None else {'status': st[i], 'decision': dc[i]}
    rows.append({'identity': ident, 'victim_behaviour': victim, 'switch_day': int(t) // 86400, 'similarity': s,
                 'volume_ratio': round(vr, 2), 'A': det('A'), 'D': det('E5_D')})
R['H1'] = rows
grp = {}
for r_ in rows:
    g = 'K1_easy(sim<0.3)' if r_['similarity'] < 0.3 else 'K2_hard(sim>=0.3)'
    e = grp.setdefault(g, {'n': 0, 'A_tp': 0, 'D_tp': 0, 'A_tp_adj': 0, 'D_tp_adj': 0})
    e['n'] += 1
    for mdl in ('A', 'D'):
        x = r_[mdl]
        if x and x['decision'] == 'anomalous':
            e[f'{mdl}_tp_adj'] += 1
            e[f'{mdl}_tp'] += x['status'] == 'detection'
for g, e in grp.items():
    e['A_TPR'], e['D_TPR'] = e['A_tp'] / e['n'], e['D_tp'] / e['n']
R['K1_K2'] = grp
log(' ', grp); save()

# ---------------------------------------------------------------- H2 false-alarm causes of D, H3 operations
log('=== H2 / H3')
sD, dD = DEC['E5_D']
ks, _ = lib.flags_c(fT['F'])
fps = [i for i in range(len(fT['d'])) if sD[i] == 'detection' and dD[i] == 'anomalous' and labT.get(fT['uid'][i]) != fT['d'][i]]


def cause(i):
    fired = [k for k in ks if ks[k][i]]
    return '+'.join(fired) if fired else 'IF_score_only'


rnd = random.Random(0)
sample = rnd.sample(fps, min(50, len(fps)))
from collections import Counter
R['H2'] = {'n_false_alarms_D': len(fps), 'sample50_causes': Counter(cause(i) for i in sample).most_common(),
           'all_fp_causes': Counter(cause(i) for i in fps).most_common(),
           'flag_rate_on_fp': {k: float(np.mean([ks[k][i] for i in fps])) if fps else 0 for k in ks}}
mD = R['variants']['E5_D']['test_seed1']
R['H3'] = {m_: {'alerts': R_['alerts'], 'alerts_per_day': R_['alerts_per_day'], 'TP': R_['TP'], 'Prec': R_['Prec']}
           for m_, R_ in (('A', mA), ('D', mD))}
log(' ', R['H2']['sample50_causes']); log(' ', R['H3']); save()

# ---------------------------------------------------------------- V5 causality, V6 columns, V7 reproducibility
log('=== V5 / V6 / V7')
recs1 = seed_data(TEST)[0]
cand = [u for u, rs in recs1.items() if len(rs) >= 10]
rnd = random.Random(1)
changed_before, changed_after = 0, 0
for u in rnd.sample(cand, 100):
    rs = recs1[u]
    t = rnd.randint(1, len(rs) - 2)
    pert = list(rs[:t + 1])
    for r_ in rs[t + 1:]:
        n = lib.Rec()
        n.d, n.first, n.last = r_.d, max(0, r_.first - 300), min(1439, r_.last + 300)
        n.types, n.hist = r_.types * 3, np.roll(r_.hist, 72) * 3
        n.total, n.sens, n.pathcnt = r_.total * 3, r_.sens * 3 + 5, r_.pathcnt * 3 + 1
        n.pref = {**{k: v * 3 for k, v in r_.pref.items()}, '/zz_new/prefix': 7}
        n.peak10 = float(n.hist.reshape(-1, 2).sum(1).max())
        pert.append(n)
    f0 = lib.compute_features({u: rs}, W=W_LOCK)
    f1 = lib.compute_features({u: pert}, W=W_LOCK)
    a0, a1 = f0['F'], f1['F']
    same = (a0 == a1) | (np.isnan(a0) & np.isnan(a1))
    changed_before += int((~same[:t + 1]).sum())
    changed_after += int((~same[t + 1:]).sum())
R['V5'] = {'users': 100, 'feature_values_changed_on_or_before_t': changed_before,
           'feature_values_changed_after_t (sanity, should be >0)': changed_after, 'pass': changed_before == 0}
log(' ', R['V5'])
cols = list(TYPES) + lib.FNAMES
banned = {'role', 'loginName', 'isLocalIP', 'location', 'ip', 'uidType', 'label'}
R['V6'] = {'n_columns_C_D': len(cols), 'columns': cols, 'banned_present': sorted(banned & set(cols)),
           'pass': len(cols) == 57 and not (banned & set(cols))}
log('  V6 columns:', len(cols), 'banned present:', R['V6']['banned_present'])
f3 = feat(3, W_LOCK)
lab3 = seed_data(3)[1]
outs = []
for rep in range(2):
    a = ascore(3, W_LOCK, 'AF', 'rolling', fresh=True)
    _, c = lib.flags_c(f3['F'])
    Rk = lib.risk(f3, a, c, W_W, LAM)
    avail = ~np.isnan(a) & ~f3['first']
    _, _, m = lib.evaluate(f3, np.nan_to_num(Rk, nan=-1) >= th_D, avail, lab3)
    outs.append(m)
R['V7'] = {'run1': outs[0], 'run2': outs[1], 'identical': outs[0] == outs[1]}
log('  V7 identical:', R['V7']['identical'])
save()
log(f'ALL_DONE in {time.time() - T0:.0f}s')
