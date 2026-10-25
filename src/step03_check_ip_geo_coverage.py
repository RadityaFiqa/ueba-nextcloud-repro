"""STEP 3 - DATA CHECK: how many events carry IP / geolocation information.
  Input : data/clue.zip
  Output: results/ip_geo_coverage.json
  Run   : python src/step03_check_ip_geo_coverage.py --data data/clue.zip --out results/ip_geo_coverage.json

------------------------------------------------------------------------
How much of CLUE-LDS carries IP / geolocation information?

CLUE-LDS has no raw IP field. IP-related information exists only as the
top-level fields `isLocalIP` (bool) and `location` (GeoIP dict), and the
`uidType` field says whether `uid` is a user name or an (anonymised) IP.

Counts events by presence of these fields, overall, per uidType, per event
type and per year, plus how many users and subset users ever carry them.
Writes ip_geo_coverage.json next to the output path given.
"""
import argparse
import collections
import io
import json
import zipfile

import orjson

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--subset_users', default=None)
ap.add_argument('--out', default='ip_geo_coverage.json')
args = ap.parse_args()

subset = set(open(args.subset_users).read().split()) if args.subset_users else set()


def cat(j):
    has_ip, has_loc = 'isLocalIP' in j, 'location' in j
    if has_loc:
        return 'location'          # location implies isLocalIP in the samples
    if has_ip:
        return 'isLocalIP_only'
    return 'none'


total = collections.Counter()
by_uidtype = collections.defaultdict(collections.Counter)
by_type = collections.defaultdict(collections.Counter)
by_year = collections.defaultdict(collections.Counter)
local_ip = collections.Counter()
countries = collections.Counter()
users_any = collections.defaultdict(set)   # uidType -> uids seen
users_geo = collections.defaultdict(set)   # uidType -> uids with location
subset_cnt = collections.Counter()

z = zipfile.ZipFile(args.data)
with z.open('clue.json') as f:
    for n, line in enumerate(f, 1):
        if n % 5_000_000 == 0:
            print(f'{n:,}', flush=True)
        j = orjson.loads(line)
        c = cat(j)
        ut = j.get('uidType', '?')
        total[c] += 1
        by_uidtype[ut][c] += 1
        by_type[j['type']][c] += 1
        by_year[j['time'][:4]][c] += 1
        users_any[ut].add(j['uid'])
        if 'isLocalIP' in j:
            local_ip[bool(j['isLocalIP'])] += 1
        if c == 'location':
            users_geo[ut].add(j['uid'])
            countries[(j['location'] or {}).get('countryCode')] += 1
        if j['uid'] in subset:
            subset_cnt[c] += 1

res = {
    'total_events': sum(total.values()),
    'overall': dict(total),
    'by_uidType': {k: dict(v) for k, v in by_uidtype.items()},
    'by_year': {k: dict(v) for k, v in sorted(by_year.items())},
    'by_event_type': {k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values()))},
    'isLocalIP_values': {str(k): v for k, v in local_ip.items()},
    'top_countries': countries.most_common(15),
    'users_by_uidType': {k: len(v) for k, v in users_any.items()},
    'users_with_location_by_uidType': {k: len(v) for k, v in users_geo.items()},
    'subset_222_users': dict(subset_cnt),
}
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('total_events', 'overall', 'by_uidType', 'isLocalIP_values',
                                      'users_by_uidType', 'users_with_location_by_uidType', 'subset_222_users')}, indent=1))
