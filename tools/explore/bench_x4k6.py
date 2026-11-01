"""Compute cost of X4 and "X4 + extra features + k6" on seed 1 (full data), stage by stage.
Run after step04 (daily tables) exists; the transitions table is rebuilt into a temporary cache so its
cost is measured. Prints wall time per stage and the peak resident memory of this process.
  Output: results/bench_x4k6.json
  Run   : python -W ignore tools/bench_x4k6.py <tmp_dir>
"""
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import lib_models as lib  # noqa: E402

TMP = sys.argv[1]
T = {}


def mark(name, t0):
    T[name] = round(time.time() - t0, 1)
    print(f'{name:34s} {T[name]:8.1f} s   peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:7.0f} MB', flush=True)


t = time.time()
BASE, TYPES = lib.load_base(config.BASE)
NT = len(TYPES)
TIX = {x: i for i, x in enumerate(TYPES)}
mark('load daily tables', t)

t = time.time()
cache = os.path.join(TMP, 'transitions.parquet')
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='3GB'")
con.execute(f"""COPY (
    WITH tr AS (SELECT uid, CAST(date_diff('day', DATE '1970-01-01', day) AS INTEGER) AS d,
                       lag(type) OVER (PARTITION BY uid, day ORDER BY rn) AS a, type AS b
                FROM read_parquet('{config.PARQUET}'))
    SELECT uid, d, a, b, count(*) AS n FROM tr WHERE a IS NOT NULL GROUP BY 1, 2, 3, 4
) TO '{cache}' (FORMAT PARQUET)""")
mark('build transitions (DuckDB, 50.5M events)', t)

t = time.time()
import x4_refine_core as core  # noqa: E402  (extra_features & peer profiles, no side effects)
core.init(BASE, TYPES, cache)
mark('load transitions + peer profiles', t)

t = time.time()
pairs, ct = lib.seed_swaps(config.STATS, 1)
recs = lib.relabel(BASE, pairs, ct)
mark('inject swaps (seed 1)', t)

t = time.time()
f = lib.compute_features(recs, W=14)
mark('features F1-F8 (14 days)', t)

t = time.time()
rel = lib.relative_A(f, W=14)
mark('relative counts (49)', t)

t = time.time()
E = core.extra_features(f, recs, pairs, ct)
mark('extra features (7, incl. order)', t)

for name, X in (('X4', np.hstack([rel, f['F']])), ('X4+k6', np.hstack([rel, f['F'], E]))):
    X = np.nan_to_num(X.astype(np.float32))
    t = time.time()
    a_raw = lib.iforest_a(f, X, schedule='rolling')
    mark(f'{name}: IF train+score ({X.shape[1]} inputs)', t)
    t = time.time()
    a = lib.calibrate_per_user(f, a_raw)
    mark(f'{name}: per-user calibration', t)
    t = time.time()
    if name == 'X4':
        _, c = lib.flags_c(f['F'])
    else:
        c = core.c_k6(f['F'], E)
    R = np.nan_to_num(lib.risk(f, a, c, (0.5, 0.4, 0.1), 0.3), nan=-1)
    mark(f'{name}: flags + risk score', t)
    t = time.time()
    m = lib.evaluate(f, R >= (0.60 if name == 'X4' else 0.68), ~np.isnan(a) & ~f['first'], lib.labels_from_swaps(pairs, ct), r=1)[2]
    mark(f'{name}: decisions + evaluation', t)
    T[f'{name}: TP'] = m['TP']
    T[f'{name}: FPR'] = round(m['FPR'], 4)

T['user_days'] = int(len(f['d']))
T['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
json.dump(T, open(os.path.join(config.RESULTS, 'bench_x4k6.json'), 'w'), indent=1)
print(json.dumps(T, indent=1))
