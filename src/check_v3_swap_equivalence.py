"""OPTIONAL CHECK V3: the uid swap on the DuckDB tables equals the per-event swap
of Landauer's generator. Needs the output of legacy/run_seed.sh.
  Run   : python src/check_v3_swap_equivalence.py "RUN_DIR/chunk_*.json" work/events.parquet work/user_stats.json

------------------------------------------------------------------------
V3: uid swap on the summary table == uid swap per event (seed 1).

Reference: per-event swapped stream written by legacy/inject_split.py
(port of generate_test_file.py; chunk_*.json = clue_anomaly.json split by uid).
Candidate: the DuckDB swap used by step05_model_A_landauer.py (CASE on uid, ts >= switch).
Compares daily count vectors (uid, day, type -> count); reports user-days that differ.
"""
import json
import os
import sys

import duckdb

from lib_uid_swap import load_users, select_pairs  # noqa: E402

chunks_glob, pq, stats = sys.argv[1], sys.argv[2], sys.argv[3]
users = load_users(stats)
pairs, ct = select_pairs(users, 1, log=lambda s: None)

con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='2GB'")
con.execute("CREATE TEMP TABLE swaps(uid VARCHAR, partner VARCHAR, ct DOUBLE)")
con.executemany("INSERT INTO swaps VALUES (?, ?, ?)", [(u, pairs[u], t) for u, t in ct.items()])
con.execute(f"""
CREATE TEMP TABLE ref AS
SELECT uid, CAST(CAST(replace(time, 'Z', '') AS TIMESTAMP) AS DATE) AS day, type, count(*) AS cnt
FROM read_json('{chunks_glob}', format='newline_delimited',
               columns={{uid:'VARCHAR', type:'VARCHAR', time:'VARCHAR'}})
GROUP BY ALL""")
con.execute(f"""
CREATE TEMP TABLE cand AS
SELECT CASE WHEN s.uid IS NOT NULL AND epoch(e.ts) >= s.ct THEN s.partner ELSE e.uid END AS uid,
       e.day, e.type, count(*) AS cnt
FROM read_parquet('{pq}') e LEFT JOIN swaps s ON e.uid = s.uid
GROUP BY ALL""")
diff_rows = con.execute("""
SELECT count(*) FROM ref FULL OUTER JOIN cand USING (uid, day, type)
WHERE ref.cnt IS DISTINCT FROM cand.cnt""").fetchone()[0]
diff_days = con.execute("""
SELECT count(DISTINCT (uid, day)) FROM ref FULL OUTER JOIN cand USING (uid, day, type)
WHERE ref.cnt IS DISTINCT FROM cand.cnt""").fetchone()[0]
n = con.execute("SELECT (SELECT count(DISTINCT (uid, day)) FROM ref), (SELECT sum(cnt) FROM ref), (SELECT sum(cnt) FROM cand)").fetchone()
swapped_days = con.execute("""
SELECT count(DISTINCT (uid, day)) FROM cand WHERE uid IN (SELECT partner FROM swaps)""").fetchone()[0]
res = {'user_days_ref': n[0], 'events_ref': int(n[1]), 'events_cand': int(n[2]),
       'user_days_of_swapped_identities': swapped_days,
       'differing_count_cells': diff_rows, 'differing_user_days': diff_days, 'pass': diff_days == 0}
json.dump(res, open('v3.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
