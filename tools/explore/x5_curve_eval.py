"""Step 2 of the operating curve: reads work/x5_curve/*.npz (x5_curve_prep.py), picks theta per false-alarm budget
on seeds 0+2 (within budget, highest mean of swap TPR and hijack TPR; ties -> lower FPR), reports seed 1.
Final model: k5 split, burst F8>=2 w1, new folders w3, w = .5/.4/.1 (x5_tune.py locked config).
  Output: results/x5_curve.json
  Run   : python -W ignore tools/x5_curve_eval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

DIR = os.path.join(config.WORK, 'x5_curve')
DAYS, LEVELS = 1910, ['rendah', 'sedang', 'tinggi']
BUDGETS = [1.0, 1.75, 2.5, 3.5, 5.0]
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]


def load(name):
    z = np.load(os.path.join(DIR, name + '.npz'))
    f = {'uid': z['uid'], 'd': z['d'], 'first': z['first'], 'last': z['last']}
    return f, z['avail'], z['R'], dict(zip(z['lab_u'].tolist(), z['lab_d'].tolist()))


def run(ds, th):
    f, avail, R, lab = ds
    return lib.evaluate(f, R >= th, avail, lab, r=1)[2]


def full(m):
    return {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in ('TP', 'TP_adj', 'FP', 'TPR', 'FPR', 'alerts_per_day')}


SW = {s: load(f'swap_s{s}') for s in (0, 2)}
HJ = {(s, lv): load(f'hijack_s{s}_{lv}') for s in (0, 2) for lv in LEVELS}
grid = []
for th in TH:
    m = dict(TP=0, FN=0, FP=0, TN=0)
    for s in SW:
        r = run(SW[s], th)
        for k in m:
            m[k] += r[k]
    hj = sum(run(HJ[k], th)['TP'] for k in HJ)
    grid.append((th, {'swap_TP': m['TP'], 'hijack_TP': hj, 'score': (m['TP'] / 40 + hj / 120) / 2,
                      'FPR': m['FP'] / (m['FP'] + m['TN']), 'fp_per_day': m['FP'] / (2 * DAYS)}))
CHOSEN = {}
for B in BUDGETS:
    th, m = max([x for x in grid if x[1]['fp_per_day'] <= B], key=lambda x: (x[1]['score'], -x[1]['FPR']))
    CHOSEN[B] = th
    print(f'budget {B}: theta={th} val swap {m["swap_TP"]}/40 hijack {m["hijack_TP"]}/120 FP/day {m["fp_per_day"]:.2f}', flush=True)
T_SW = load('swap_s1')
T_HJ = {lv: load(f'hijack_s1_{lv}') for lv in LEVELS}
OUT = {'meta': __doc__.strip(), 'chosen_theta': CHOSEN,
       'validation_grid': [(th, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}) for th, m in grid], 'seed1': {}}
for B in BUDGETS:
    th = CHOSEN[B]
    sw = full(run(T_SW, th))
    hj = {lv: full(run(T_HJ[lv], th)) for lv in LEVELS}
    OUT['seed1'][str(B)] = {'theta': th, 'swap': sw, 'hijack_TP': sum(hj[lv]['TP'] for lv in LEVELS), 'hijack_TP_adj': sum(hj[lv]['TP_adj'] for lv in LEVELS),
                            'hijack_per_level': {lv: hj[lv]['TP'] for lv in LEVELS}}
    r = OUT['seed1'][str(B)]
    print(f"budget {B}: theta={th} | swap {sw['TP']}/20 (adj {sw['TP_adj']}) FPR {sw['FPR']:.4f} {sw['alerts_per_day']:.2f}/hari "
          f"| hijack {r['hijack_TP']}/60 (adj {r['hijack_TP_adj']}) {r['hijack_per_level']}", flush=True)
json.dump(OUT, open(os.path.join(config.RESULTS, 'x5_curve.json'), 'w'), indent=1, default=str)
print('DONE')
