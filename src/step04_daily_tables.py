"""STEP 4 - DATA: per user-day tables used by every model except A
(B, B+RS, C, D, X1-X4, hybrid) and by the K3 tests.
Also defines SENSITIVE, the list of sensitive actions (feature F7, flag k5, K3 context).
  Input : work/events.parquet
  Output: work/base/{types,daystat,hist,prefix}.parquet
  Run   : python src/step04_daily_tables.py work/events.parquet work/base

------------------------------------------------------------------------
Build the per-user-day base tables once from events.parquet (DuckDB).

Everything the models need is additive per (uid, day), so a seed's uid swap
(which always switches at UTC midnight) is an exact day-level relabel of
these tables (checked against the per-event swap in V3).

Outputs (Parquet, in OUT_DIR):
  types.parquet    uid, d, type, cnt                      -> Set A (49 columns)
  daystat.parquet  uid, d, total, first_mi, last_mi, sens, pathcnt
  hist.parquet     uid, d, bin, cnt   (bin = minute-of-day // 5, 288 bins)
  prefix.parquet   uid, d, prefix, cnt (2-level folder prefix of path events)
d = days since 1970-01-01 (UTC); times are UTC.
"""
import os
import sys
import time

import duckdb

SENSITIVE = [
    'file_deleted', 'deleted_from_trashbin', 'version_deleted',
    'shared_user', 'shared_link', 'shared_group', 'shared_email',
    'public_share_update_permission', 'public_share_password_changed',
    'public_share_expiration_date_changed', 'permission_changed',
    'password_changed', 'email_changed', 'user_created', 'user_deleted', 'user_disabled',
    'user_added_to_group', 'user_became_groupadmin', 'admin_settings_changed',
    'login_failed',
]

if __name__ == '__main__':
    pq, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"SET threads=2; SET memory_limit='2GB'; SET temp_directory='{out}/tmp'")
    con.execute(f"""
    CREATE VIEW ev AS
    SELECT uid, date_diff('day', DATE '1970-01-01', day) AS d, type,
           hour(ts) * 60 + minute(ts) AS mi,
           coalesce(json_extract_string(params, '$.path'), json_extract_string(params, '$.newpath'),
                    json_extract_string(params, '$.fileTarget'), json_extract_string(params, '$.filePath'),
                    json_extract_string(params, '$.itemTarget')) AS path
    FROM read_parquet('{pq}')""")
    sens = ', '.join(f"'{s}'" for s in SENSITIVE)
    jobs = {
        'types': "SELECT uid, d, type, count(*)::INT AS cnt FROM ev GROUP BY ALL",
        'daystat': f"""SELECT uid, d, count(*)::INT AS total, min(mi)::INT AS first_mi, max(mi)::INT AS last_mi,
                              sum(CASE WHEN type IN ({sens}) THEN 1 ELSE 0 END)::INT AS sens,
                              sum(CASE WHEN path IS NOT NULL THEN 1 ELSE 0 END)::INT AS pathcnt
                       FROM ev GROUP BY uid, d""",
        'hist': "SELECT uid, d, (mi // 5)::SMALLINT AS bin, count(*)::INT AS cnt FROM ev GROUP BY ALL",
        'prefix': """SELECT uid, d, regexp_extract(path, '^(/[^/]*(/[^/]*)?)', 1) AS prefix, count(*)::INT AS cnt
                     FROM ev WHERE path IS NOT NULL GROUP BY ALL""",
    }
    for name, q in jobs.items():
        t = time.time()
        con.execute(f"COPY ({q}) TO '{out}/{name}.parquet' (FORMAT parquet, COMPRESSION zstd)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out}/{name}.parquet')").fetchone()[0]
        print(f'{name}: {n:,} rows in {time.time() - t:.0f}s', flush=True)
    tot = con.execute(f"SELECT sum(total) FROM read_parquet('{out}/daystat.parquet')").fetchone()[0]
    print(f'events covered: {tot:,} (expect 50,522,931); total {time.time() - t0:.0f}s')
