"""STEP 5 - MODEL A: baseline of Landauer et al. (2022), uid-swap attack (Test 1).
  What  : picks the uid-swap pairs (lib_uid_swap.py), swaps the uids from the switch time,
          then runs Landauer's detect.py loop unchanged (section 3 below).
  Input : work/events.parquet, work/user_stats.json
  Output: OUTDIR/labels.txt, OUTDIR/metrics.json; with --dump the per user-day decisions
          that steps 6-8 reuse as "model A".
  Run   : python src/step05_model_A_landauer.py --parquet work/events.parquet
              --stats work/user_stats.json --outdir work/A/seed1 -s 1 --dump work/A/seed1_t0.6.jsonl
  Expect: seed 1 -> TP 13, FN 7, FP 3337, TN 69112, train 5289 (= Landauer's README), ~10 s

------------------------------------------------------------------------
Baseline A (Landauer et al., 2022) for one seed on events.parquet, via DuckDB.

Exact reformulation of generate_test_file.py + detect.py:
  * pairs/switch times: lib_uid_swap.py (port of generate_test_file.py);
  * injection: SQL CASE on (uid, ts >= switch time) instead of rewriting JSON;
  * detect.py scores a user whenever the day of their next event differs
    from the previous one, i.e. on *runs* of consecutive same-day events in
    file order. Runs are computed in SQL (lag(day) over rn per user), and
    each run's event-type counts are the count vector detect.py would build.
    This stays exact even if events are not time-sorted within a user.
  * the scoring loop below is detect.py's loop body, unchanged, applied to
    each run except a user's last one (detect.py never closes the last day).

Mode 2 (idf) is not supported (it depends on global first-seen order).
"""
import argparse
import datetime
import json
import os
import sys
import time

import duckdb
import pytz

from lib_uid_swap import DEFAULTS, load_users, select_pairs, write_labels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--parquet', required=True)
ap.add_argument('--stats', required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('-s', '--seed', type=int, default=1)
ap.add_argument('--users', default=None, help='optional file with uids to keep (subset)')
ap.add_argument('-t', '--thresh', type=float, default=0.6)
ap.add_argument('-r', '--retrain', type=int, default=1)
ap.add_argument('-m', '--mode', type=int, default=1, choices=[1, 3])
ap.add_argument('-q', '--queue', type=int, default=30)
ap.add_argument('-u', '--update', action='store_true')
ap.add_argument('-f', '--fifo', action='store_true')
ap.add_argument('--dump', default=None, help='write per-user-day decisions (uid, d, status, decision) as JSON lines')
ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--memory', default='2GB')
args = ap.parse_args()

threshold, anom_free_days, mode, queue = args.thresh, args.retrain, args.mode, args.queue
update, fifo = args.update, args.fifo
os.makedirs(args.outdir, exist_ok=True)
start_time = time.time()

# --- 1. pair selection (identical to generate_test_file.py) -----------------
users = load_users(args.stats)
log_lines = []
pairs, change_times = select_pairs(users, args.seed, log=lambda s: (print(s), log_lines.append(s)))
labels_path = os.path.join(args.outdir, 'labels.txt')
write_labels(labels_path, pairs, change_times)

# --- 2. injection + per-user runs in DuckDB ---------------------------------
t_sql = time.time()
con = duckdb.connect()
con.execute(f"SET threads={args.threads}; SET memory_limit='{args.memory}'; "
            f"SET temp_directory='{os.path.join(args.outdir, 'duck_tmp')}'")
con.execute("CREATE TEMP TABLE swaps(uid VARCHAR, partner VARCHAR, ct DOUBLE)")
con.executemany("INSERT INTO swaps VALUES (?, ?, ?)", [(u, pairs[u], t) for u, t in change_times.items()])
where = ''
if args.users:
    con.execute("CREATE TEMP TABLE keep(uid VARCHAR)")
    con.executemany("INSERT INTO keep VALUES (?)", [(u,) for u in open(args.users).read().split()])
    where = 'WHERE e.uid IN (SELECT uid FROM keep)'
# Injected event stream (swapped uid), as a view.
con.execute(f"""
CREATE TEMP VIEW e AS
SELECT e.rn, e.type, e.day,
       CASE WHEN s.uid IS NOT NULL AND epoch(e.ts) >= s.ct THEN s.partner ELSE e.uid END AS uid
FROM read_parquet('{args.parquet}') e LEFT JOIN swaps s ON e.uid = s.uid
{where}""")
# Fast path (hash aggregate, no sort of 50M rows): per (uid, day, type) counts
# plus each user-day's rn interval. If a user's day intervals never overlap,
# every day is exactly one contiguous run in file order -> runs = days by lo.
con.execute("""
CREATE TEMP TABLE udt AS
SELECT uid, day, type, count(*) AS cnt, min(rn) AS lo, max(rn) AS hi FROM e GROUP BY uid, day, type""")
con.execute("""
CREATE TEMP TABLE ud AS SELECT uid, day, min(lo) AS lo, max(hi) AS hi FROM udt GROUP BY uid, day""")
overlap = [r[0] for r in con.execute("""
SELECT DISTINCT uid FROM (
  SELECT uid, lo, max(hi) OVER (PARTITION BY uid ORDER BY lo
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_hi
  FROM ud) WHERE lo < prev_hi""").fetchall()]
con.execute("CREATE TEMP TABLE ov(uid VARCHAR)")
if overlap:
    con.executemany("INSERT INTO ov VALUES (?)", [(u,) for u in overlap])
rows = con.execute("""
SELECT t.uid, d.lo AS run, t.day, t.type, t.cnt
FROM udt t JOIN ud d USING (uid, day)
WHERE t.uid NOT IN (SELECT uid FROM ov)
ORDER BY t.uid, d.lo""").fetchall()
if overlap:
    # Exact fallback for users whose days interleave in file order: runs of
    # consecutive same-day events by rn (what detect.py actually scores).
    rows += con.execute("""
    WITH r AS (
      SELECT *, CASE WHEN day = lag(day) OVER (PARTITION BY uid ORDER BY rn) THEN 0 ELSE 1 END AS new_run
      FROM e WHERE uid IN (SELECT uid FROM ov)
    ), g AS (
      SELECT *, sum(new_run) OVER (PARTITION BY uid ORDER BY rn ROWS UNBOUNDED PRECEDING) AS run FROM r
    )
    SELECT uid, run, any_value(day), type, count(*) FROM g GROUP BY uid, run, type ORDER BY uid, run
    """).fetchall()
n_days = dict(con.execute("SELECT uid, count(*) FROM ud GROUP BY uid").fetchall())
con.close()
print(f'{len(n_days)} users, {len(overlap)} with interleaved days (exact fallback)', flush=True)
sql_time = time.time() - t_sql

runs = {}  # uid -> list of (day datetime, {type: count}) in run order
for uid, run, day, typ, cnt in rows:
    lst = runs.setdefault(uid, [])
    if not lst or lst[-1][0] != run:
        lst.append([run, datetime.datetime(day.year, day.month, day.day, 0, 0, 0, 0, pytz.UTC), {}])
    lst[-1][2][typ] = cnt

# --- 3. detect.py scoring loop (body unchanged) -----------------------------
# How model A decides, per user and per active day:
#   * the user's model is a list `dists[uid]` of up to `queue` (30) past daily
#     count vectors {event type: count};
#   * distance of today to one known day = sum |today - known| / sum max(today, known)
#     over all event types (0 = identical, 1 = no type in common);
#   * a = the smallest distance to any known day (nearest neighbour);
#   * a > threshold (0.6) -> "anomalous", otherwise "normal";
#   * retraining mode: after an anomaly the next `anom_free_days` (r = 1) day(s) are
#     "training" (not scored) and are added to the model, so the model learns the
#     new behaviour.
# Comments below explain the code; the code itself is Landauer's.
anomalous_users = {}
with open(labels_path) as f:
    for line in f:
        parts = line.split(',')
        ts = datetime.datetime.fromtimestamp(int(float(parts[1])), tz=pytz.UTC)
        anomalous_users[parts[0]] = datetime.datetime(ts.year, ts.month, ts.day, 0, 0, 0, 0, pytz.UTC)

detected_dist = {}
for uid, user_runs in runs.items():
    wait_days = {uid: anom_free_days}   # > 0 -> user is in training/retraining mode
    dists = {uid: []}                   # the user's model: known daily count vectors
    for _, day, counts in user_runs[:-1]:   # the last run is never closed in detect.py
        freq_day = {uid: counts}            # today's count vector
        # distance from today to every known day; keep the smallest (min_dist)
        last_active_day = {uid: day}
        min_dist = None
        min_known = None
        min_limit = None
        duplicate_check = []
        for known in dists[uid]:
            if known in duplicate_check:
                continue
            duplicate_check.append(known)
            manh = 0
            limit = 0
            # manh = sum |today - known|, limit = sum max(today, known); mode 1 uses raw counts
            for action_element in set(list(known.keys()) + list(freq_day[uid].keys())):
                idf_fact = 1
                norm_sum_known = 1
                norm_sum_freq = 1
                if mode == 3:
                    norm_sum_known = sum(known.values())
                    norm_sum_freq = sum(freq_day[uid].values())
                if action_element not in known:
                    manh += freq_day[uid][action_element] * idf_fact / norm_sum_freq
                    limit += freq_day[uid][action_element] * idf_fact / norm_sum_freq
                elif action_element not in freq_day[uid]:
                    manh += known[action_element] * idf_fact / norm_sum_known
                    limit += known[action_element] * idf_fact / norm_sum_known
                else:
                    manh += abs(freq_day[uid][action_element] * idf_fact / norm_sum_freq - known[action_element] * idf_fact / norm_sum_known)
                    limit += max(freq_day[uid][action_element] * idf_fact / norm_sum_freq, known[action_element] * idf_fact / norm_sum_known)
            if min_dist is None:
                min_dist = manh / limit
                min_known = known
                min_limit = limit
            else:
                if manh / limit < min_dist:
                    min_dist = manh / limit
                    min_known = known
                    min_limit = limit
            if min_dist == 0:
                break
        last_active_day_date = datetime.datetime(last_active_day[uid].year, last_active_day[uid].month, last_active_day[uid].day, 0, 0, 0, 0, pytz.UTC)
        sample = (uid, last_active_day_date)
        if sample not in detected_dist:
            detected_dist[sample] = []
        # status: is today scored ("detection") or a training day?
        if wait_days[uid] <= 0:
            detected_dist[sample].append("detection")
        else:
            detected_dist[sample].append("training")
        # decision: alarm if even the most similar known day differs by more than the threshold
        if min_dist is not None and min_dist > threshold:
            detected_dist[sample].append("anomalous")
            wait_days[uid] = anom_free_days         # enter retraining mode
        else:
            detected_dist[sample].append("normal")
            wait_days[uid] -= 1
            if fifo is False and min_known is not None:
                # move the matched day to the end so it is dropped last (least recently used)
                dists[uid].remove(min_known)
                dists[uid].append(min_known)
        # learn: add today to the model while in (re)training mode (or always with -u)
        if update is True or wait_days[uid] >= 0:
            dists[uid].append(freq_day[uid])
        # keep at most `queue` known days: drop the oldest
        if queue != -1 and len(dists[uid]) > queue:
            dists[uid] = dists[uid][1:]

if args.dump:
    with open(args.dump, 'w') as out:
        for (uid, day), detected in detected_dist.items():
            out.write(json.dumps([uid, int(day.timestamp()) // 86400] + detected) + '\n')

# --- 4. evaluation (same rules and formulas as detect.py get_eval_results) ---
c = dict(tp=0, tp_adj=0, fp=0, tn=0, fn=0, fn_adj=0, train=0, detect=0)
per_user = {uid: dict(tp_adj=0, fn=0) for uid in anomalous_users}
for (uid, day), detected in detected_dist.items():
    if "training" in detected:
        c['train'] += 1
    elif "detection" in detected:
        c['detect'] += 1
    if uid in anomalous_users and day.timestamp() == anomalous_users[uid].timestamp():
        if detected == ["detection", "anomalous"]:
            c['tp'] += 1; c['tp_adj'] += 1; per_user[uid]['tp_adj'] += 1
        elif detected == ["detection", "normal"] or detected == ["training", "normal"]:
            c['fn_adj'] += 1; c['fn'] += 1; per_user[uid]['fn'] += 1
        elif detected == ["training", "anomalous"]:
            c['tp_adj'] += 1; per_user[uid]['tp_adj'] += 1; c['fn'] += 1
    else:
        if detected == ["detection", "anomalous"]:
            c['fp'] += 1
        elif detected == ["detection", "normal"]:
            c['tn'] += 1

div = lambda a, b: a / b if b > 0 else float('nan')
m = {
    'seed': args.seed,
    'users': len(n_days), 'user_days': sum(n_days.values()),
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
    'detected_users': [u for u, pu in per_user.items() if pu == {'tp_adj': 1, 'fn': 0}],
    'missed_users': [u for u, pu in per_user.items() if pu == {'tp_adj': 0, 'fn': 1}],
    'eval_errors': [u for u, pu in per_user.items() if pu not in ({'tp_adj': 1, 'fn': 0}, {'tp_adj': 0, 'fn': 1})],
    'params': dict(threshold=threshold, retrain=anom_free_days, mode=mode, queue=queue, update=update, fifo=fifo),
    'sql_seconds': round(sql_time, 1), 'total_seconds': round(time.time() - start_time, 1),
}
for k, v in m.items():
    print(f'  {k} = {v}')
json.dump(m, open(os.path.join(args.outdir, 'metrics.json'), 'w'), indent=1)
