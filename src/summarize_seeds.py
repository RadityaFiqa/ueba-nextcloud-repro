"""Per-seed table and mean ± std (sample std, ddof=1) of baseline metrics.

Usage: python summarize.py RUNS_DIR [SEED ...]   (default seeds 3-9 = test seeds)
Writes RUNS_DIR/summary.csv and prints a markdown table.
"""
import csv
import json
import os
import statistics
import sys

runs_dir = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]] or list(range(3, 10))
cols = ['TP', 'FN', 'FP', 'TN', 'TPR', 'TPR_adj', 'FPR', 'R', 'Prec', 'F1']

rows = []
for s in seeds:
    m = json.load(open(os.path.join(runs_dir, f'seed{s}', 'metrics.json')))
    m['FP_per_day'] = None
    rows.append({'seed': s, **{c: m[c] for c in cols}, 'eval_errors': len(m['eval_errors'])})

with open(os.path.join(runs_dir, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed'] + cols + ['eval_errors'])
    w.writeheader()
    w.writerows(rows)

fmt = lambda c, v: f'{v:.4f}' if isinstance(v, float) else str(v)
print('| seed | ' + ' | '.join(cols) + ' |')
print('|' + '---|' * (len(cols) + 1))
for r in rows:
    print(f"| {r['seed']} | " + ' | '.join(fmt(c, r[c]) for c in cols) + ' |')
mean = {c: statistics.mean(r[c] for r in rows) for c in cols}
std = {c: statistics.stdev(r[c] for r in rows) for c in cols}
print('| **mean ± std** | ' + ' | '.join(
    f'{mean[c]:.4f} ± {std[c]:.4f}' if c in ('TPR', 'TPR_adj', 'FPR', 'R', 'Prec', 'F1') else f'{mean[c]:.1f} ± {std[c]:.1f}'
    for c in cols) + ' |')
print('eval_errors total:', sum(r['eval_errors'] for r in rows))
