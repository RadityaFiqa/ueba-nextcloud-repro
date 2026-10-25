"""Pass 1: per-user statistics (replaces get_user_similarity.py).

Collects the same per-user fields as get_user_similarity.py (event types,
count, active days) in one streaming pass. The 5000x5000 similarity list is
NOT materialised; inject_split.py computes the few similarities it needs on
demand with the identical formula, so memory stays small.

Output: user_stats.json with users in first-appearance order (this order
drives random.choice in the injection and must match the original).
"""
import argparse
import datetime
import json

import pytz
from dateutil import parser

from common import open_events

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, help='clue.zip or clue.json')
ap.add_argument('--out', default='user_stats.json')
args = ap.parse_args()

users = {}
cnt = 0
for line in open_events(args.data):
    cnt += 1
    if cnt % 5_000_000 == 0:
        print(f'{cnt:,} lines', flush=True)
    j = json.loads(line)
    uid = j['uid']
    action = j['type']
    ts = parser.isoparse(j['time'])
    day = datetime.datetime(ts.year, ts.month, ts.day, 0, 0, 0, 0, pytz.UTC).timestamp()
    u = users.get(uid)
    if u is None:
        users[uid] = {'actions': {action}, 'cnt': 1, 'day_list': {day}}
    else:
        u['actions'].add(action)
        u['cnt'] += 1
        u['day_list'].add(day)

out = {}
for uid, u in users.items():
    out[uid] = {'actions': sorted(u['actions']), 'cnt': u['cnt'],
                'days': len(u['day_list']), 'day_list': sorted(u['day_list'])}
with open(args.out, 'w') as f:
    json.dump({'user_list': list(users.keys()), 'user_info': out}, f)
print(f'{cnt:,} lines, {len(users):,} users -> {args.out}')
