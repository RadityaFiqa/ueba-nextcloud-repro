"""Per-user statistics needed by the uid-swap pair selection (pairs.py), from events.parquet.

Same content and order as legacy/stats.py (which streams clue.zip in ~20 min):
users in first-appearance order; per user the sorted event types, event count,
number of active days and the sorted UTC-midnight timestamps of those days.
"""
import json
import sys

import duckdb

import config

out = sys.argv[1] if len(sys.argv) > 1 else config.STATS
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='2GB'")
rows = con.execute(f"""
SELECT uid, count(*) AS cnt,
       list_sort(list_distinct(list(type))) AS actions,
       list_sort(list_distinct(list(epoch(CAST(day AS TIMESTAMP))))) AS day_list
FROM read_parquet('{config.PARQUET}')
GROUP BY uid
ORDER BY min(rn)""").fetchall()
info = {uid: {'actions': actions, 'cnt': cnt, 'days': len(day_list), 'day_list': [float(d) for d in day_list]}
        for uid, cnt, actions, day_list in rows}
json.dump({'user_list': [r[0] for r in rows], 'user_info': info}, open(out, 'w'))
print(f'{len(rows):,} users, {sum(v["cnt"] for v in info.values()):,} events -> {out}')
