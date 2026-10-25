"""Sum per-chunk counts and print metrics with the formulas of detect.py."""
import glob
import json
import os
import sys

run_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(run_dir, 'result_*.json')))
keys = ['tp', 'tp_adj', 'fp', 'tn', 'fn', 'fn_adj', 'train', 'detect', 'users', 'user_days']
c = {k: 0 for k in keys}
per_user = {}
for fn in files:
    r = json.load(open(fn))
    for k in keys:
        c[k] += r[k]
    for uid, pu in r['per_user'].items():
        assert uid not in per_user, f'{uid} in two chunks'
        per_user[uid] = pu

labels = [l.split(',')[0] for l in open(os.path.join(run_dir, 'labels.txt')) if l.strip()]
detected = [u for u in labels if per_user.get(u) == {'tp_adj': 1, 'fn': 0}]
missed = [u for u in labels if per_user.get(u) == {'tp_adj': 0, 'fn': 1}]
errors = [u for u in labels if u not in detected and u not in missed]

div = lambda a, b: a / b if b > 0 else float('nan')
m = {
    'chunks': len(files),
    'users': c['users'], 'user_days': c['user_days'],
    'total': c['tp'] + c['tn'] + c['fp'] + c['fn'],
    'train': c['train'], 'detect': c['detect'],
    'TP_adj': c['tp_adj'], 'TP': c['tp'], 'FP': c['fp'], 'TN': c['tn'],
    'FN_adj': c['fn_adj'], 'FN': c['fn'],
    'TPR_adj': div(c['tp_adj'], c['tp_adj'] + c['fn_adj']),
    'TPR': div(c['tp'], c['tp'] + c['fn']),
    'FPR': div(c['fp'], c['fp'] + c['tn']),
    'TNR': div(c['tn'], c['tn'] + c['fp']),
    'Prec': div(c['tp_adj'], c['tp_adj'] + c['fp']),
    'F1': div(c['tp_adj'], c['tp_adj'] + 0.5 * (c['fp'] + c['fn_adj'])),
    'ACC': div(c['tp_adj'] + c['tn'], c['tp_adj'] + c['tn'] + c['fp'] + c['fn_adj']),
    'R': div(c['train'], c['train'] + c['detect']),
    'detected_users': detected, 'missed_users': missed, 'eval_errors': errors,
}
for k, v in m.items():
    print(f'  {k} = {v}')
json.dump(m, open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=1)
