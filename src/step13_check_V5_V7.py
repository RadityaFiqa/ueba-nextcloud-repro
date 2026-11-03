"""STEP 13 - CHECKS V5-V7 on the inputs of IF Personal and Model Utama (run after step 12).
  V5 causality   : changing a user's data after day t changes none of F1..F8, the 49 relative counts,
                   trans_js or the per-user ranking a on days <= t (100 random users, seed 1 data without swap)
  V6 no leakage  : the Isolation Forest sees 57 columns (49 relative counts + F1..F8), the flags see F1..F8 and
                   trans_js; no role, login name, IP, location or label
  V7 repeatable  : seed 3 (uid swap) computed twice gives identical F, trans_js, a_raw and a, and they equal
                   work/final/swap_s3.npz written by step 12
  Output: results/checks_V5_V7.json
  Run   : python -W ignore src/step13_check_V5_V7.py
"""
import json
import os
import random
from collections import ChainMap

import numpy as np

import config
import lib_models as lib
import lib_order_features as core

W = 14
OUT = os.path.join(config.RESULTS, 'checks_V5_V7.json')
os.makedirs(config.RESULTS, exist_ok=True)
BASE, TYPES = lib.load_base(config.BASE)
core.init(TYPES, os.path.join(config.WORK, 'transitions.parquet'))
R = {}


def same(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (x == y) | (np.isnan(x) & np.isnan(y))


def inputs(recs, pairs, ct, trans=None):
    f = lib.compute_features(recs, W=W)
    return f, lib.relative_A(f, W=W), core.trans_js(BASE, recs, pairs, ct, trans)


# ---------------------------------------------------------------- V5 causality
clean = lib.relabel(BASE, {}, {})
rnd = random.Random(1)
before, after = {}, {}
for u in rnd.sample([u for u, rs in clean.items() if len(rs) >= 10], 100):
    rs = clean[u]
    t = rnd.randint(1, len(rs) - 2)
    pert, tr = list(rs[:t + 1]), {}
    for r_ in rs[t + 1:]:  # 3x volume, shifted hours, more sensitive actions, a new folder, new transitions
        n = lib.Rec()
        n.d, n.first, n.last = r_.d, max(0, r_.first - 300), min(1439, r_.last + 300)
        n.types, n.hist = r_.types * 3, np.roll(r_.hist, 72) * 3
        n.total, n.sens, n.pathcnt = r_.total * 3, r_.sens * 3 + 5, r_.pathcnt * 3 + 1
        n.pref = {**{k: v * 3 for k, v in r_.pref.items()}, '/zz_new/prefix': 7}
        n.peak10 = float(n.hist.reshape(-1, 2).sum(1).max())
        pert.append(n)
        tr[(u, r_.d)] = {**{k: v * 3 for k, v in core.S['trans'].get((u, r_.d), {}).items()}, 0: 9}
    f0, rel0, tj0 = inputs({u: rs}, {}, {})
    f1, rel1, tj1 = inputs({u: pert}, {}, {}, ChainMap(tr, core.S['trans']))
    a0 = np.linspace(0, 1, len(rs))  # per-user ranking: change a_raw after t only
    a1 = a0.copy()
    a1[t + 1:] = 1 - a1[t + 1:]
    c0, c1 = lib.calibrate_per_user(f0, a0), lib.calibrate_per_user(f0, a1)
    for name, x, y in (('F1..F8', f0['F'], f1['F']), ('relative counts', rel0, rel1), ('trans_js', tj0, tj1), ('a', c0, c1)):
        s = same(x, y)
        before[name] = before.get(name, 0) + int((~s[:t + 1]).sum())
        after[name] = after.get(name, 0) + int((~s[t + 1:]).sum())
R['V5'] = {'users': 100, 'values_changed_on_or_before_t': before, 'values_changed_after_t (should be > 0)': after,
           'pass': all(v == 0 for v in before.values()) and all(v > 0 for v in after.values())}
print('V5', R['V5'], flush=True)
del clean

# ---------------------------------------------------------------- V6 columns
cols_if = ['rel_' + t for t in TYPES] + lib.FNAMES
cols_flags = lib.FNAMES + ['trans_js']
banned = {'role', 'loginName', 'isLocalIP', 'location', 'ip', 'uidType', 'label'}
R['V6'] = {'n_columns_isolation_forest': len(cols_if), 'columns_isolation_forest': cols_if, 'columns_flags': cols_flags,
           'banned_present': sorted(banned & {c.removeprefix('rel_') for c in cols_if + cols_flags}),
           'pass': len(cols_if) == 57 and not (banned & {c.removeprefix('rel_') for c in cols_if + cols_flags})}
print('V6 columns:', len(cols_if), 'banned present:', R['V6']['banned_present'], flush=True)

# ---------------------------------------------------------------- V7 repeatability (seed 3)
pairs, ct = lib.seed_swaps(config.STATS, 3)
recs = lib.relabel(BASE, pairs, ct)
runs = []
for _ in range(2):
    f, rel, tj = inputs(recs, pairs, ct)
    a_raw = lib.iforest_a(f, np.nan_to_num(np.hstack([rel, f['F']]).astype(np.float32)))
    runs.append({'F': f['F'], 'trans_js': tj, 'a_raw': a_raw, 'a': lib.calibrate_per_user(f, a_raw)})
stored = np.load(os.path.join(config.WORK, 'final', 'swap_s3.npz'))
R['V7'] = {k: {'rows': len(runs[0][k]), 'run1_eq_run2': bool(same(runs[0][k], runs[1][k]).all()),
               'eq_step12': bool(same(runs[0][k], stored[k]).all())} for k in runs[0]}
R['V7']['pass'] = all(v['run1_eq_run2'] and v['eq_step12'] for v in R['V7'].values())
print('V7', R['V7'], flush=True)
json.dump(R, open(OUT, 'w'), indent=1)
print('CHECKS', 'PASS' if all(R[k]['pass'] for k in ('V5', 'V6', 'V7')) else 'FAIL')
