"""STEP 1 - DATA: convert the CLUE-LDS dataset to Parquet.
  Input : data/clue.zip (Zenodo 7119953)
  Output: work/events.parquet (50,522,931 events, column rn = original line order)
  Run   : python src/step01_data_to_parquet.py --zip data/clue.zip --out work/events.parquet --work work

------------------------------------------------------------------------
Convert CLUE-LDS (clue.zip) to Parquet with DuckDB, keeping file order.

Steps
  1. unzip clue.json to a temporary file (DuckDB cannot read .zip);
  2. load it into a disk-backed DuckDB table (insertion order preserved,
     so rowid == line number - 1);
  3. export events.parquet with `rn` (= rowid) plus parsed columns;
  4. sanity checks: row count, and whether `id` increases with `rn`.

Columns: rn, id, uid, uidType, type, time (original string), ts (TIMESTAMP,
UTC), day (DATE), role, isLocalIP, location (JSON), params (JSON).
"""
import argparse
import os
import subprocess
import time

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument('--zip', required=True)
ap.add_argument('--out', default='events.parquet')
ap.add_argument('--work', default='.')
ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--memory', default='4GB')
args = ap.parse_args()

t0 = time.time()
raw = os.path.join(args.work, 'clue.json')
if not os.path.exists(raw):
    print('unzipping ...', flush=True)
    subprocess.run(['unzip', '-o', '-q', args.zip, 'clue.json', '-d', args.work], check=True)
print(f'unzipped in {time.time() - t0:.0f}s', flush=True)

db_path = os.path.join(args.work, 'convert.duckdb')
if os.path.exists(db_path):
    os.remove(db_path)
con = duckdb.connect(db_path)
con.execute(f"SET threads={args.threads}; SET memory_limit='{args.memory}'; SET preserve_insertion_order=true;")
con.execute(f"SET temp_directory='{os.path.join(args.work, 'duck_tmp')}'")

t1 = time.time()
con.execute(f"""
CREATE TABLE ev AS
SELECT * FROM read_json('{raw}',
  format='newline_delimited',
  columns={{id:'BIGINT', uid:'VARCHAR', uidType:'VARCHAR', type:'VARCHAR', time:'VARCHAR',
           role:'VARCHAR', isLocalIP:'BOOLEAN', location:'JSON', params:'JSON'}},
  maximum_object_size=104857600)
""")
print(f'loaded in {time.time() - t1:.0f}s', flush=True)

t2 = time.time()
con.execute(f"""
COPY (
  SELECT rowid AS rn, id, uid, uidType, type, time,
         CAST(replace(time, 'Z', '') AS TIMESTAMP) AS ts,
         CAST(CAST(replace(time, 'Z', '') AS TIMESTAMP) AS DATE) AS day,
         role, isLocalIP, location, params
  FROM ev ORDER BY rowid
) TO '{args.out}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
""")
print(f'exported in {time.time() - t2:.0f}s', flush=True)

chk = con.execute(f"""
SELECT count(*) AS n,
       max(rn) AS max_rn,
       sum(CASE WHEN id <= prev_id THEN 1 ELSE 0 END) AS id_not_increasing,
       sum(CASE WHEN ts < prev_ts_user THEN 1 ELSE 0 END) AS ts_backwards_within_user
FROM (
  SELECT rn, id, ts, lag(id) OVER (ORDER BY rn) AS prev_id,
         lag(ts) OVER (PARTITION BY uid ORDER BY rn) AS prev_ts_user
  FROM read_parquet('{args.out}')
)
""").fetchone()
print(f'rows={chk[0]:,} max_rn={chk[1]:,} id_not_increasing={chk[2]} ts_backwards_within_user={chk[3]}')
con.close()
os.remove(db_path)
print(f'total {time.time() - t0:.0f}s; parquet {os.path.getsize(args.out) / 1e9:.2f} GB')
