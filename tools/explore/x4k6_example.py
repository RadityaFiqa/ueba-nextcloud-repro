"""Worked example of "X4 + extra features + k6" for one swapped account on seed 1: the raw inputs
(today's counts vs the user's own 14-day median, today's vs history's transitions), F1..F8, the 7
extra features, k1..k6, c, and the previous day's R and h. a / a_raw are taken from x4k6_trace.json
(they need the global IF, which x4k6_trace.py already computed).
  Run: python -W ignore tools/x4k6_example.py <uid>
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402
import x4_refine_core as core  # noqa: E402

UID = sys.argv[1]
BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
core.init(BASE, TYPES, os.path.join(config.WORK, 'transitions.parquet'))
pairs, ct = lib.seed_swaps(config.STATS, 1)
recs = lib.relabel(BASE, pairs, ct)
one = {UID: recs[UID]}
f = lib.compute_features(one, W=14)
E = core.extra_features(f, one, pairs, ct)
rel = lib.relative_A(f, W=14)
day = lib.labels_from_swaps(pairs, ct)[UID]
i = int(np.flatnonzero(f['d'] == day)[0])
lst = recs[UID]
hist = lst[max(0, i - 14):i]
A = f['A'].astype(float)
med = np.median(A[max(0, i - 14):i], axis=0)
tr = json.load(open(os.path.join(config.RESULTS, 'x4k6_trace.json')))
t = next(r for r in tr['trace_seed1'] if r['akun'] == UID)

inv = {v: k for k, v in pairs.items()}
print('akun', UID, 'hari', day, 'aktivitas dari', inv.get(UID), 'jumlah hari riwayat', len(hist))
print('\nJUMLAH AKTIVITAS: hari ini vs median 14 hari, rel = log1p(x) - log1p(median)')
order = np.argsort(-(A[i] + med))
for k in order:
    if A[i, k] == 0 and med[k] == 0:
        continue
    print(f'  {TYPES[k]:28s} hari ini {int(A[i, k]):5d}  median {med[k]:7.1f}  rel {rel[i, k]:+.2f}')
print('  total hari ini', int(A[i].sum()), ' median total 14 hari', float(np.median(A[max(0, i - 14):i].sum(1))))

orig = lambda d: UID if (d in BASE.get(UID, {}) and (UID not in ct or d * 86400 < ct[UID])) else pairs.get(UID)
tt = core.S['trans'].get((orig(lst[i].d), lst[i].d), {})
th = Counter()
for p in hist:
    th.update(core.S['trans'].get((orig(p.d), p.d), {}))
name = lambda k: f'{TYPES[k // NT]} -> {TYPES[k % NT]}'
st, sh = sum(tt.values()), sum(th.values())
print('\nURUTAN (transisi a -> b): porsi hari ini vs porsi 14 hari')
for k in sorted(set(tt) | set(th), key=lambda k: -(tt.get(k, 0) / max(st, 1) + th.get(k, 0) / max(sh, 1)))[:10]:
    print(f'  {name(k):55s} hari ini {tt.get(k, 0) / max(st, 1):.2f}  riwayat {th.get(k, 0) / max(sh, 1):.2f}')
print('  transisi hari ini', st, ' transisi riwayat', sh)

F = f['F'][i]
labels = ['F1 jumlah (z)', 'F2 pola JS', 'F3 jenis baru', 'F4 luar jam', 'F5 lama aktif (z)', 'F6 folder baru', 'F7 berisiko', 'F8 lonjakan 10 mnt (z)']
print('\nFITUR KONTEKS')
for lab, v in zip(labels, F):
    print(f'  {lab:24s} {v:+.3f}')
print('\nFITUR TAMBAHAN')
for lab, v in zip(['|F1|', '|F5|', 'hour_js', 'trans_js', 'trans_new', 'folder_jac', 'peer_gain'], E[i]):
    print(f'  {lab:24s} {v:+.3f}')
print('\nDARI JEJAK (x4k6_trace.json):', {k: t[k] for k in ('a_mentah', 'a', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'c', 'h', 'R', 'status', 'keputusan')})
print('jam aktivitas hari ini (per jam):', np.asarray(lst[i].hist).reshape(24, 12).sum(1).astype(int).tolist())
print('jam aktivitas riwayat (per jam):', sum(np.asarray(p.hist).reshape(24, 12).sum(1) for p in hist).astype(int).tolist())
