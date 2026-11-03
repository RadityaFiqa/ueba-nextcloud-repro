"""Order-of-activities feature trans_js (flag k6 of Model Utama).
trans_js = Jensen-Shannon divergence between today's event-type transitions (a -> b, consecutive events in log
order) and those of the user's previous 14 active days. Call init(TYPES, transitions_parquet) first."""
from collections import Counter

import duckdb
import numpy as np

import lib_models as lib

W = 14
S = {}


def init(types, trans_parquet):
    nt = len(types)
    tix = {t: i for i, t in enumerate(types)}
    con = duckdb.connect()
    trans = {}
    for u, d, a, b, n in con.execute(f"SELECT uid, d, a, b, n FROM read_parquet('{trans_parquet}')").fetchall():
        trans.setdefault((u, d), {})[tix[a] * nt + tix[b]] = n
    con.close()
    S.update(nt=nt, trans=trans)


def js(p, q):
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return lib.js_div(p.astype(float), q.astype(float))


def trans_js(base, recs, pairs, ct, trans=None):
    """One value per row of lib.compute_features(recs) (sorted uid, d); NaN on a user's first day.
    Transitions are looked up under the uid that really produced the day (before/after a uid swap);
    trans overrides the lookup (e.g. a ChainMap with injected hijack sessions)."""
    trans = S['trans'] if trans is None else trans
    out = []
    for uid in sorted(recs):
        lst = recs[uid]
        partner = pairs.get(uid)
        own_days = base.get(uid, {})

        def orig(d, uid=uid, partner=partner, own_days=own_days):
            if d in own_days and (uid not in ct or d * 86400 < ct[uid]):
                return uid
            return partner
        for i, r in enumerate(lst):
            if i == 0:
                out.append(np.nan)
                continue
            tt = trans.get((orig(r.d), r.d), {})
            th = Counter()
            for p in lst[max(0, i - W):i]:
                th.update(trans.get((orig(p.d), p.d), {}))
            keys = set(tt) | set(th)
            out.append(js(np.array([tt.get(k, 0) for k in keys], float), np.array([th.get(k, 0) for k in keys], float)) if keys else 0.0)
    return np.array(out)
