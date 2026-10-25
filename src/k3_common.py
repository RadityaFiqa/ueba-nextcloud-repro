"""K3 (M7 session hijack) attack generator, as fixed in pipeline/SPEC_K3_window_DRAFT.md
(approved 2026-10-24, before any K3 result).

Per seed: 20 victims drawn at random from the 174 users meeting Landauer's victim rules
(users with fewer than 30 active days are skipped: the day rule leaves them no attack day); attack day =
random active day 29..n-1 of the victim; session start uniform 00:00-21:00 UTC;
duration uniform 1-3 h. Victims, days and times are shared by the three intensities
of a seed (only volumes differ). Events: login_attempt + login_successful at the start,
file_accessed at k x the victim's median events per 30-min window (min 4 per 30 min),
s shared_link and d file_deleted; times uniform within the session; each path-bearing
event uses a folder the victim used before (p = 0.5) or a folder of other users the
victim never touched before the attack day.
"""
import config  # data / work / results paths (see config.py)
import json
import os
import random

import duckdb
import numpy as np

LEVELS = {'rendah': (2, 1, 0), 'sedang': (5, 3, 0), 'tinggi': (10, 5, 5)}
N_VICTIMS = 20
STATS = config.STATS
PARQUET = config.PARQUET


def eligible_users(stats_path=STATS):
    info = json.load(open(stats_path))['user_info']
    return sorted(u for u, v in info.items() if v['days'] > 27 and v['cnt'] > 100 and len(v['actions']) > 4)


def win30_counts(users, out=os.path.join(config.K3, 'win30.parquet')):
    """(uid, w, n): events per 30-min window (w = epoch // 1800) for the given users."""
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        con = duckdb.connect()
        con.execute("SET threads=2; SET memory_limit='2GB'")
        con.execute("CREATE TEMP TABLE el(uid VARCHAR)")
        con.executemany("INSERT INTO el VALUES (?)", [(u,) for u in users])
        con.execute(f"""COPY (SELECT uid, (epoch(ts) // 1800)::BIGINT AS w, count(*)::INT AS n
                         FROM read_parquet('{PARQUET}') WHERE uid IN (SELECT uid FROM el) GROUP BY ALL)
                         TO '{out}' (FORMAT parquet)""")
    con = duckdb.connect()
    res = {}
    for u, w, n in con.execute(f"SELECT uid, w, n FROM read_parquet('{out}') ORDER BY uid, w").fetchall():
        res.setdefault(u, []).append((w, n))
    return res


def plan_attacks(seed, base, w30, elig):
    """Victims, days and session times for one seed (shared by all intensities)."""
    rng = random.Random(1000 + seed)
    order = list(elig)
    rng.shuffle(order)
    # the day rule (active day 29..n-1) needs >= 30 active days; users with 28-29
    # days pass Landauer's rule but have no valid attack day and are skipped
    victims = [v for v in order if len(base[v]) >= 30][:N_VICTIMS]
    plans = []
    for v in victims:
        days = sorted(base[v])
        d = days[rng.randint(28, len(days) - 2)]
        start = d * 86400 + rng.uniform(0, 21 * 3600)
        dur = rng.uniform(3600, 3 * 3600)
        hist = [n for w, n in w30[v] if w * 1800 < d * 86400]
        plans.append(dict(victim=v, day=d, start=start, dur=dur, med30=float(np.median(hist))))
    return plans


def known_prefixes(base, v, d):
    s = set()
    for dd, r in base[v].items():
        if dd < d:
            s.update(r.pref)
    return sorted(s)


def attack_events(plan, level, seed, base, all_prefixes):
    """[(ts, type, prefix or None), ...] for one victim, one intensity."""
    k, s, dl = LEVELS[level]
    rng = random.Random(f"{1000 + seed}-{level}-{plan['victim']}")
    v, d, start, dur = plan['victim'], plan['day'], plan['start'], plan['dur']
    known = known_prefixes(base, v, d)
    known_set = set(known)

    def folder():
        if known and rng.random() < 0.5:
            return rng.choice(known)
        while True:
            p = rng.choice(all_prefixes)
            if p not in known_set:
                return p

    ev = [(start, 'login_attempt', None), (start + 1, 'login_successful', None)]
    n_access = int(round(max(4.0, k * plan['med30']) * dur / 1800))
    for typ, n in (('file_accessed', n_access), ('shared_link', s), ('file_deleted', dl)):
        for _ in range(n):
            ev.append((start + rng.uniform(0, dur), typ, folder()))
    return sorted(ev)
