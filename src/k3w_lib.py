"""Shared code for per-window K3 evaluation (K3-b v2, pipeline/SPEC_K3_window_v2.md).

Win / load_windows / inject / user_features / calib_risk / GlobalIF are the K3-b versions
(k3_window.py), unchanged. New: user_features_v2 (EWMA + hour-profile baseline),
alert_stream (accumulation over K calendar windows + per-user suppression) and evaluate2.
"""
import config  # data / work / results paths (see config.py)
import math
import os
from collections import deque

import duckdb
import numpy as np
from sklearn.ensemble import IsolationForest

from base_daily import SENSITIVE

PARQUET = config.PARQUET
SENS = set(SENSITIVE)
DAYS = 1910


class Win:
    __slots__ = ('w', 'types', 'total', 'sens', 'pathcnt', 'pref', 'hour')


def load_windows(users, W, types):
    """uid -> list of Win sorted by w (w = epoch // (60 W))."""
    path = f'{config.K3}/win{W}'
    os.makedirs(path, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='2GB'")
    con.execute("CREATE TEMP TABLE el(uid VARCHAR)")
    con.executemany("INSERT INTO el VALUES (?)", [(u,) for u in users])
    sens = ', '.join(f"'{s}'" for s in SENSITIVE)
    if not os.path.exists(f'{path}/types.parquet'):
        con.execute(f"""CREATE TEMP VIEW ev AS SELECT uid, (epoch(ts) // {60 * W})::BIGINT AS w, type,
            coalesce(json_extract_string(params, '$.path'), json_extract_string(params, '$.newpath'),
                     json_extract_string(params, '$.fileTarget'), json_extract_string(params, '$.filePath'),
                     json_extract_string(params, '$.itemTarget')) AS path
            FROM read_parquet('{PARQUET}') WHERE uid IN (SELECT uid FROM el)""")
        con.execute(f"COPY (SELECT uid, w, type, count(*)::INT cnt FROM ev GROUP BY ALL) TO '{path}/types.parquet' (FORMAT parquet)")
        con.execute(f"""COPY (SELECT uid, w, count(*)::INT total, sum(CASE WHEN type IN ({sens}) THEN 1 ELSE 0 END)::INT sens,
            sum(CASE WHEN path IS NOT NULL THEN 1 ELSE 0 END)::INT pathcnt FROM ev GROUP BY uid, w) TO '{path}/stat.parquet' (FORMAT parquet)""")
        con.execute(f"""COPY (SELECT uid, w, regexp_extract(path, '^(/[^/]*(/[^/]*)?)', 1) prefix, count(*)::INT cnt
            FROM ev WHERE path IS NOT NULL GROUP BY ALL) TO '{path}/prefix.parquet' (FORMAT parquet)""")
    tix = {t: i for i, t in enumerate(types)}
    wins = {}
    for uid, w, total, s, pc in con.execute(f"SELECT uid, w, total, sens, pathcnt FROM read_parquet('{path}/stat.parquet')").fetchall():
        x = Win()
        x.w, x.total, x.sens, x.pathcnt = w, total, s, pc
        x.types = np.zeros(len(types))
        x.pref = {}
        x.hour = int((w * 60 * W) % 86400 // 3600)
        wins.setdefault(uid, {})[w] = x
    for uid, w, t, c in con.execute(f"SELECT uid, w, type, cnt FROM read_parquet('{path}/types.parquet')").fetchall():
        wins[uid][w].types[tix[t]] = c
    for uid, w, p, c in con.execute(f"SELECT uid, w, prefix, cnt FROM read_parquet('{path}/prefix.parquet')").fetchall():
        wins[uid][w].pref[p] = c
    return {u: [d[w] for w in sorted(d)] for u, d in wins.items()}


def inject(wins_u, events, W, tix):
    byw = {x.w: x for x in wins_u}
    new = {}
    for ts, typ, pref in events:
        w = int(ts // (60 * W))
        if w not in new:
            o = byw.get(w)
            x = Win()
            x.w = w
            x.hour = int((w * 60 * W) % 86400 // 3600)
            if o is None:
                x.types, x.total, x.sens, x.pathcnt, x.pref = np.zeros(len(tix)), 0, 0, 0, {}
            else:
                x.types, x.total, x.sens, x.pathcnt, x.pref = o.types.copy(), o.total, o.sens, o.pathcnt, dict(o.pref)
            new[w] = x
        x = new[w]
        x.types[tix[typ]] += 1
        x.total += 1
        x.sens += typ in SENS
        if pref is not None:
            x.pathcnt += 1
            x.pref[pref] = x.pref.get(pref, 0) + 1
    byw.update(new)
    return [byw[w] for w in sorted(byw)]


def _context(x, seen, hourh, hcount, sratio, M):
    c2 = (sum(c for p, c in x.pref.items() if p not in seen) / sum(x.pref.values())) if x.pref else 0.0
    c3 = max(0.0, x.sens / x.total - (sum(sratio) / len(sratio)))
    c4 = 0.0
    if len(hourh) >= M:
        cum = np.cumsum(hcount); tot = cum[-1]
        lo_h = int(np.searchsorted(cum, 0.025 * tot)); hi_h = int(np.searchsorted(cum, 0.975 * tot))
        c4 = float(not (lo_h <= x.hour <= hi_h))
    return c2, c3, c4


def _push(x, seen, hourh, hcount, sratio, NH):
    seen.update(x.pref)
    hourh.append((x.hour, x.total)); hcount[x.hour] += x.total
    if len(hourh) > NH:
        h0, t0 = hourh.popleft(); hcount[h0] -= t0
    sratio.append(x.sens / x.total)


def user_features(wl, N=30, M=5, NH=200):
    """K3-b X4-win features (unchanged). Returns (X, c, a_landauer)."""
    n = len(wl)
    T = np.array([x.types for x in wl])
    L = np.log1p(T)
    X = np.full((n, T.shape[1] + 4), np.nan, dtype=np.float32)
    flags = np.zeros(n)
    aL = np.full(n, np.nan)
    seen, hourh, hcount = set(), deque(), np.zeros(24)
    logtot, sratio = deque(maxlen=N), deque(maxlen=N)
    for i, x in enumerate(wl):
        if i > 0:
            lo = max(0, i - N)
            X[i, :T.shape[1]] = L[i] - np.log1p(np.median(T[lo:i], axis=0))
            lt = math.log1p(x.total)
            if len(logtot) >= M:
                v = np.asarray(logtot); med = np.median(v); mad = np.median(np.abs(v - med))
                c1 = float(np.clip((lt - med) / (1.4826 * mad + 0.5), -10, 10))
            else:
                c1 = 0.0
            c2, c3, c4 = _context(x, seen, hourh, hcount, sratio, M)
            X[i, T.shape[1]:] = (c1, c2, c3, c4)
            flags[i] = (c1 >= 3) + (c2 >= 0.5) + (c3 >= 0.1) + (c4 >= 1)
            Q = T[lo:i]
            aL[i] = float(np.min(np.abs(Q - T[i]).sum(1) / np.maximum(Q, T[i]).sum(1)))
        _push(x, seen, hourh, hcount, sratio, NH)
        logtot.append(math.log1p(x.total))
    return X, flags / 4.0, aL


def user_features_v2(wl, alpha=0.1, alpha_h=0.2, N=30, M=5, NH=200):
    """X5 baseline: EWMA per type (alpha) and hour-of-day EWMA of the volume (alpha_h)."""
    n = len(wl)
    T = np.array([x.types for x in wl])
    X = np.full((n, T.shape[1] + 4), np.nan, dtype=np.float32)
    flags = np.zeros(n)
    seen, hourh, hcount = set(), deque(), np.zeros(24)
    sratio = deque(maxlen=N)
    m = T[0].astype(float).copy()
    mh = np.full(24, np.nan)
    mtot = float(wl[0].total)
    mh[wl[0].hour] = mtot
    _push(wl[0], seen, hourh, hcount, sratio, NH)
    for i in range(1, n):
        x = wl[i]
        X[i, :T.shape[1]] = np.log1p(T[i]) - np.log1p(m)
        ref = mh[x.hour] if not np.isnan(mh[x.hour]) else mtot
        c1 = math.log1p(x.total) - math.log1p(ref)
        c2, c3, c4 = _context(x, seen, hourh, hcount, sratio, M)
        X[i, T.shape[1]:] = (c1, c2, c3, c4)
        flags[i] = (c1 >= math.log(4)) + (c2 >= 0.5) + (c3 >= 0.1) + (c4 >= 1)
        m = (1 - alpha) * m + alpha * T[i]
        mtot = (1 - alpha) * mtot + alpha * x.total
        mh[x.hour] = x.total if np.isnan(mh[x.hour]) else (1 - alpha_h) * mh[x.hour] + alpha_h * x.total
        _push(x, seen, hourh, hcount, sratio, NH)
    return X, flags / 4.0


def calib_risk(a, c, N=30, M=5, w=(0.5, 0.4, 0.1), lam=0.3):
    a_cal = a.copy()
    hist = []
    R = np.full(len(a), np.nan)
    h = 0.0
    for i in range(len(a)):
        if np.isnan(a[i]):
            continue
        if len(hist) >= M:
            q = np.asarray(hist[-N:])
            a_cal[i] = (np.sum(q < a[i]) + 0.5 * np.sum(q == a[i])) / len(q)
        hist.append(a[i])
        R[i] = w[0] * a_cal[i] + w[1] * c[i] + w[2] * h
        h = lam * h + (1 - lam) * R[i]
    return R


class GlobalIF:
    def __init__(self, rows_t, X, W, T=60, step=7, first_after=7, ref_max=20000):
        day = rows_t * 60 * W // 86400
        valid = ~np.isnan(X).any(1)
        self.day0 = int(day[valid].min())
        self.step = step
        self.models = {}
        rng = np.random.default_rng(0)
        for r in range(self.day0 + first_after, int(day.max()) + 1, step):
            tr = np.flatnonzero(valid & (day >= max(self.day0, r - T)) & (day < r))
            if len(tr) < 25:
                continue
            m = IsolationForest(n_estimators=200, max_samples=min(256, len(tr)), random_state=0, n_jobs=2).fit(X[tr])
            ref = tr if len(tr) <= ref_max else rng.choice(tr, ref_max, replace=False)
            self.models[r] = (m, np.sort(m.score_samples(X[ref])))
        self.W = W

    def score(self, rows_t, X):
        day = rows_t * 60 * self.W // 86400
        a = np.full(len(X), np.nan)
        keys = (day - self.day0 - 7) // self.step * self.step + self.day0 + 7
        ok = ~np.isnan(X).any(1)
        for r in np.unique(keys[ok]):
            if r not in self.models:
                continue
            idx = np.flatnonzero(ok & (keys == r))
            m, ref = self.models[r]
            s = m.score_samples(X[idx])
            a[idx] = (len(ref) - np.searchsorted(ref, s, side='right')) / len(ref)
        return a


def alert_stream(ws, s, theta, strict=False, K=1, suppress=0):
    """Indices of emitted alerts. s may hold NaN (no score -> no alert).
    K > 1: score = sum of s over calendar windows (w-K, w]; suppress > 0: after an emitted
    alert, the user's alerts in the next `suppress` windows are held back."""
    ok = ~np.isnan(s)
    v = np.where(ok, s, 0.0)
    if K > 1:
        acc = np.empty(len(v))
        j, run = 0, 0.0
        for i in range(len(v)):
            run += v[i]
            while ws[j] <= ws[i] - K:
                run -= v[j]; j += 1
            acc[i] = run
        v = acc
    hit = ok & ((v > theta) if strict else (v >= theta))
    idx = np.flatnonzero(hit)
    if suppress <= 0:
        return idx
    out, last = [], None
    for i in idx:
        if last is None or ws[i] - last >= suppress:
            out.append(i); last = ws[i]
    return np.asarray(out, dtype=int)


def evaluate2(scores, wl_by_user, plans, W, theta, strict=False, K=1, suppress=0):
    """-> (detected per victim dict {uid: latency_min or None}, false alarms per day)."""
    att = {p['victim']: (int(p['start'] // (60 * W)), int((p['start'] + p['dur'] - 1e-9) // (60 * W)), p['start']) for p in plans}
    fp = 0
    det = {}
    for u, s in scores.items():
        ws = np.array([x.w for x in wl_by_user[u]])
        idx = alert_stream(ws, s, theta, strict, K, suppress)
        if u in att:
            w0, w1, st = att[u]
            w1k = w1 + K - 1
            in_att = (ws[idx] >= w0) & (ws[idx] <= w1k)
            fp += int(np.sum(~in_att))
            hits = idx[in_att]
            det[u] = (((ws[hits[0]] + 1) * 60 * W - st) / 60) if len(hits) else None
        else:
            fp += len(idx)
    return det, fp / DAYS
