"""uid-swap pair selection, line-for-line port of clue-lds generate_test_file.py.

Shared by inject_split.py (streaming pipeline) and duck/duck_seed.py (DuckDB
pipeline) so both select exactly the same pairs and switch times.
Changes vs. the original are mechanical only: similarities are computed on
demand (same formula and rounding as get_user_similarity.py) and
random.randint gets int() timestamps (Python >= 3.12 rejects floats).
"""
import datetime
import json
import random

import pytz

DEFAULTS = dict(pairs=10, min_sim=0.1, max_sim=0.6, days_before=25, days_after=2,
                min_count=100, min_unique=4, w_count=0.3, w_events=0.7)


def load_users(stats_path):
    users = json.load(open(stats_path))['user_info']
    for u in users.values():
        u['actions'] = set(u['actions'])
    return users


def select_pairs(users, seed, pairs=10, min_sim=0.1, max_sim=0.6, days_before=25, days_after=2,
                 min_count=100, min_unique=4, w_count=0.3, w_events=0.7, log=print):
    """Return (pairs, change_times): uid -> partner uid, uid -> switch day (epoch s)."""
    w_sum = w_count + w_events
    w_count, w_events = w_count / w_sum, w_events / w_sum
    random.seed(seed)

    def similarity(a, b):
        ua, ub = users[a], users[b]
        ra, rb = ua['cnt'] / ua['days'], ub['cnt'] / ub['days']
        cnt_sim = min(ra, rb) / max(ra, rb)
        action_sim = len(ua['actions'] & ub['actions']) / len(ua['actions'] | ub['actions'])
        return round(w_count * cnt_sim + w_events * action_sim, 3)

    def eligible(u):
        return not (u['days'] <= days_before + days_after or u['cnt'] <= min_count
                    or len(u['actions']) <= min_unique)

    out_pairs, change_times = {}, {}
    choices = list(users.keys())
    while len(out_pairs) / 2 < pairs:
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
            if min_sim <= sim <= max_sim and first_user != user_inner:
                dl1, dl2 = users[first_user]['day_list'], users[user_inner]['day_list']
                earliest_change = max(dl1[days_before], dl2[days_before])
                latest_change = min(dl1[-days_after], dl2[-days_after])
                if earliest_change < latest_change:
                    ctt = random.randint(int(earliest_change), int(latest_change))
                    change_time_first = next(d for d in dl1 if ctt <= d)
                    change_time_second = next(d for d in dl2 if ctt <= d)
                    second_user = user_inner
                    found = True
        if found:
            log(f'Pair (sim {similarity(first_user, second_user)}): '
                f'{first_user} @ {datetime.datetime.fromtimestamp(change_time_first, pytz.utc)} <-> '
                f'{second_user} @ {datetime.datetime.fromtimestamp(change_time_second, pytz.utc)}')
            choices.remove(second_user)
            out_pairs[first_user], out_pairs[second_user] = second_user, first_user
            change_times[first_user], change_times[second_user] = change_time_first, change_time_second
    return out_pairs, change_times


def write_labels(path, out_pairs, change_times):
    with open(path, 'w') as labels:
        for uid, t in change_times.items():
            # pairs[uid]: we detect the first activity of the replacement user
            labels.write(str(out_pairs[uid]) + ',' + str(t) + '\n')
