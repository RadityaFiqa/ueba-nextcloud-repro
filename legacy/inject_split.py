"""Pass 2: anomaly injection + split by user (replaces generate_test_file.py).

Pair selection is a line-for-line port of generate_test_file.py (same
defaults, same random call sequence), with two mechanical changes:
  * similarities are computed on demand instead of read from user_info.txt;
  * random.randint gets int() timestamps (Python >= 3.12 rejects floats;
    older Pythons converted integral floats identically).

Parsing uses orjson and datetime.fromisoformat (same values, ~3x faster).

Instead of writing one 14.5 GB clue_anomaly.json, each (swapped) event is
written as a minimal {"uid","type","time"} record - the only fields
detect.py reads - into chunk_<k>.json, k = crc32(new uid) % N. Event order
inside each chunk is the original file order.
"""
import argparse
import datetime
import json
import os
import random

import orjson
import pytz

from common import chunk_of, open_events

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--stats', default='user_stats.json')
ap.add_argument('--outdir', required=True)
ap.add_argument('--chunks', type=int, default=8)
ap.add_argument('-s', '--seed', type=int, default=1)
ap.add_argument('-p', '--pairs', type=int, default=10)
ap.add_argument('-m', '--min_sim', type=float, default=0.1)
ap.add_argument('-n', '--max_sim', type=float, default=0.6)
ap.add_argument('-b', '--days_before', type=int, default=25)
ap.add_argument('-a', '--days_after', type=int, default=2)
ap.add_argument('-c', '--min_count', type=int, default=100)
ap.add_argument('-u', '--min_unique', type=int, default=4)
ap.add_argument('--w_count', type=float, default=0.3)
ap.add_argument('--w_events', type=float, default=0.7)
args = ap.parse_args()

w_sum = args.w_count + args.w_events
w_count, w_events = args.w_count / w_sum, args.w_events / w_sum

random.seed(args.seed)
users = json.load(open(args.stats))['user_info']
for u in users.values():
    u['actions'] = set(u['actions'])


def similarity(a, b):
    # Same expression and rounding as get_user_similarity.py
    ua, ub = users[a], users[b]
    ra, rb = ua['cnt'] / ua['days'], ub['cnt'] / ub['days']
    cnt_sim = min(ra, rb) / max(ra, rb)
    action_sim = len(ua['actions'] & ub['actions']) / len(ua['actions'] | ub['actions'])
    return round(w_count * cnt_sim + w_events * action_sim, 3)


def eligible(u):
    return not (u['days'] <= args.days_before + args.days_after
                or u['cnt'] <= args.min_count
                or len(u['actions']) <= args.min_unique)


pairs, change_times = {}, {}
choices = list(users.keys())
while len(pairs) / 2 < args.pairs:
    if len(choices) == 0:
        raise SystemExit('Could not find user pair, try to extend allowed similarity range.')
    first_user = random.choice(choices)
    choices.remove(first_user)
    if not eligible(users[first_user]):
        continue
    choices_inner = list(choices)
    found = False
    second_user = change_time_first = change_time_second = None
    while found is False:
        if len(choices_inner) == 0:
            break
        user_inner = random.choice(choices_inner)
        choices_inner.remove(user_inner)
        if not eligible(users[user_inner]):
            continue
        sim = similarity(first_user, user_inner)
        if args.min_sim <= sim <= args.max_sim and first_user != user_inner:
            dl1, dl2 = users[first_user]['day_list'], users[user_inner]['day_list']
            earliest_change = max(dl1[args.days_before], dl2[args.days_before])
            latest_change = min(dl1[-args.days_after], dl2[-args.days_after])
            if earliest_change < latest_change:
                ctt = random.randint(int(earliest_change), int(latest_change))
                change_time_first = next(d for d in dl1 if ctt <= d)
                change_time_second = next(d for d in dl2 if ctt <= d)
                second_user = user_inner
                found = True
    if found:
        print(f'Pair (sim {similarity(first_user, second_user)}): '
              f'{first_user} @ {datetime.datetime.fromtimestamp(change_time_first, pytz.utc)} <-> '
              f'{second_user} @ {datetime.datetime.fromtimestamp(change_time_second, pytz.utc)}', flush=True)
        choices.remove(second_user)
        pairs[first_user], pairs[second_user] = second_user, first_user
        change_times[first_user], change_times[second_user] = change_time_first, change_time_second

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, 'labels.txt'), 'w') as labels:
    for uid, t in change_times.items():
        # pairs[uid]: we detect the first activity of the replacement user
        labels.write(str(pairs[uid]) + ',' + str(t) + '\n')

outs = [open(os.path.join(args.outdir, f'chunk_{k}.json'), 'w', encoding='utf-8', buffering=1 << 20)
        for k in range(args.chunks)]
cnt = 0
for line in open_events(args.data):
    cnt += 1
    if cnt % 5_000_000 == 0:
        print(f'{cnt:,} lines', flush=True)
    j = orjson.loads(line)
    uid = j['uid']
    # fromisoformat == dateutil isoparse for CLUE's "YYYY-MM-DDTHH:MM:SSZ" (Python >= 3.11)
    if uid in change_times and datetime.datetime.fromisoformat(j['time']).timestamp() >= change_times[uid]:
        uid = pairs[uid]
    outs[chunk_of(uid, args.chunks)].write(
        orjson.dumps({'uid': uid, 'type': j['type'], 'time': j['time']}).decode() + '\n')
for o in outs:
    o.close()
print(f'{cnt:,} lines split into {args.chunks} chunks in {args.outdir}')
