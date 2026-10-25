"""STEP 3 - DATA CHECK V1 (field coverage) and V2 (no event lost in aggregation).
  Input : work/events.parquet
  Output: results/v1_v2.json
  Run   : python src/step03_check_v1_v2_data.py work/events.parquet results/v1_v2.json

------------------------------------------------------------------------
V1 (field coverage) and V2 (aggregation keeps every event) on events.parquet."""
import json
import sys

import duckdb

pq = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else 'v1_v2.json'
con = duckdb.connect()
con.execute("SET threads=1; SET memory_limit='1500MB'")

# V1: share of events / user-days carrying each field
fields = {
    'type': "type IS NOT NULL",
    'time': "time IS NOT NULL AND ts IS NOT NULL",
    'path': "json_extract_string(params, '$.path') IS NOT NULL",
    'any_path_field': "coalesce(json_extract_string(params, '$.path'), json_extract_string(params, '$.newpath'), "
                      "json_extract_string(params, '$.fileTarget'), json_extract_string(params, '$.filePath'), "
                      "json_extract_string(params, '$.itemTarget')) IS NOT NULL",
    'isLocalIP': "isLocalIP IS NOT NULL",
    'location': "location IS NOT NULL",
    'role': "role IS NOT NULL",
}
sel_ev = ', '.join(f"avg(CASE WHEN {c} THEN 1 ELSE 0 END) AS {k}" for k, c in fields.items())
sel_ud = ', '.join(f"max(CASE WHEN {c} THEN 1 ELSE 0 END) AS {k}" for k, c in fields.items())
v1 = {}
for scope, where in [('all', ''), ('uidType=name', "WHERE uidType = 'name'")]:
    ev = con.execute(f"SELECT count(*), {sel_ev} FROM read_parquet('{pq}') {where}").fetchone()
    ud = con.execute(f"""SELECT count(*), {', '.join(f'avg({k})' for k in fields)} FROM (
                         SELECT uid, day, {sel_ud} FROM read_parquet('{pq}') {where} GROUP BY uid, day)""").fetchone()
    v1[scope] = {'events': ev[0], 'user_days': ud[0],
                 'pct_events': {k: round(100 * v, 4) for k, v in zip(fields, ev[1:])},
                 'pct_user_days': {k: round(100 * v, 4) for k, v in zip(fields, ud[1:])}}

# V2: sum of counts in the user-day-type summary == raw row count
raw = con.execute(f"SELECT count(*) FROM read_parquet('{pq}')").fetchone()[0]
agg = con.execute(f"SELECT sum(cnt), count(*) FROM (SELECT uid, day, type, count(*) cnt FROM read_parquet('{pq}') GROUP BY ALL)").fetchone()
v2 = {'raw_rows': raw, 'summary_event_sum': int(agg[0]), 'summary_rows': agg[1], 'difference': raw - int(agg[0]),
      'expected': 50522931, 'pass': raw == int(agg[0]) == 50522931}

res = {'V1': v1, 'V2': v2}
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
