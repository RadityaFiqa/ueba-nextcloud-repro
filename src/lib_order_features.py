"""Order-of-activity and other user-relative extra features (no module-level side effects).
Used by step12: trans_js = Jensen-Shannon divergence between today's event-type transitions (a -> b, log order)
and those of the user's previous 14 active days (flag k6). Call init(BASE, TYPES, transitions_parquet) first;
pass peer=False (peer_gain compares with other users and is not used)."""
import bisect
from collections import Counter

import duckdb
import numpy as np

import lib_models as lib

W, NPEER, M = 14, 30, 5
S = {}


def init(base, types, trans_parquet):
    nt = len(types)
    tix = {t: i for i, t in enumerate(types)}
    con = duckdb.connect()
    trans = {}
    for u, d, a, b, n in con.execute(f"SELECT uid, d, a, b, n FROM read_parquet('{trans_parquet}')").fetchall():
        trans.setdefault((u, d), {})[tix[a] * nt + tix[b]] = n
    con.close()
    peers = [u for u, days in base.items() if len(days) >= 28]
    S.update(base=base, nt=nt, trans=trans, peers=peers, pidx={u: k for k, u in enumerate(peers)},
             pd={u: sorted(base[u]) for u in peers}, cache={},
             pc={u: np.vstack([np.zeros(nt), np.cumsum(np.array([base[u][d].types for d in sorted(base[u])], float), 0)]) for u in peers})


def peer_matrix(day):
    if day not in S['cache']:
        P = np.zeros((len(S['peers']), S['nt']))
        for u in S['peers']:
            k = bisect.bisect_left(S['pd'][u], day)
            if k >= M:
                P[S['pidx'][u]] = S['pc'][u][k] - S['pc'][u][max(0, k - NPEER)]
        s = P.sum(1, keepdims=True)
        S['cache'][day] = np.where(s > 0, P / np.where(s > 0, s, 1), np.nan)
    return S['cache'][day]


def js_rows(p, Q):
    m = 0.5 * (p + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum(1)
        b = np.where(Q > 0, Q * np.log2(Q / m), 0).sum(1)
    return 0.5 * a + 0.5 * b


def js(p, q):
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return lib.js_div(p.astype(float), q.astype(float))


def extra_features(f, recs, pairs, ct, peer=True, trans=None):
    """peer=False skips peer_gain (column 6 stays 0): it compares with other users' profiles.
    trans overrides the transition lookup (e.g. a ChainMap with injected sessions)."""
    base, trans = S['base'], (S['trans'] if trans is None else trans)
    out = np.full((len(f['d']), 7), np.nan)
    row = 0
    for uid in sorted(recs):
        lst = recs[uid]
        partner = pairs.get(uid)
        own_days = base.get(uid, {})

        def orig(d, uid=uid, partner=partner, own_days=own_days):
            if d in own_days and (uid not in ct or d * 86400 < ct[uid]):
                return uid
            return partner
        for i, r in enumerate(lst):
            if i > 0:
                hist = lst[max(0, i - W):i]
                Fi = f['F'][row]
                hh = lambda x: np.asarray(x.hist, float).reshape(24, 12).sum(1)
                hour_js = js(hh(r), sum(hh(p) for p in hist))
                tt = trans.get((orig(r.d), r.d), {})
                th = Counter()
                for p in hist:
                    th.update(trans.get((orig(p.d), p.d), {}))
                keys = set(tt) | set(th)
                trans_js = js(np.array([tt.get(k, 0) for k in keys], float), np.array([th.get(k, 0) for k in keys], float)) if keys else 0.0
                trans_new = (sum(n for k, n in tt.items() if k not in th) / sum(tt.values())) if tt else 0.0
                ft = set(r.pref)
                fh = set().union(*[set(p.pref) for p in hist])
                folder_jac = (1 - len(ft & fh) / len(ft | fh)) if ft else 0.0
                today = np.asarray(r.types, float)
                own = np.array([p.types for p in lst[max(0, i - NPEER):i]], float).sum(0)
                peer_gain = 0.0
                if peer:
                    own_js = js(today, own)
                    P = peer_matrix(r.d)
                    mask = ~np.isnan(P[:, 0])
                    if uid in S['pidx']:
                        mask[S['pidx'][uid]] = False
                    best = float(np.nanmin(js_rows(today / today.sum(), P[mask]))) if mask.any() and today.sum() > 0 else 1.0
                    peer_gain = own_js - best
                out[row] = (abs(Fi[0]), abs(Fi[4]), hour_js, trans_js, trans_new, folder_jac, peer_gain)
            row += 1
    return out


def c_k6(F, E):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    k = np.column_stack([g(np.abs(F[:, 0])) >= 3, (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2), g(F[:, 3]) >= 0.3,
                         g(F[:, 5]) >= 0.5, (g(F[:, 6]) >= 0.1) | (g(F[:, 7]) >= 3), g(E[:, 3]) >= 0.5])
    return (k * np.array([2, 2, 1, 2, 1, 2])).sum(1) / 10.0
