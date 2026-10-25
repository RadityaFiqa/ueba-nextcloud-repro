"""STEP 10 - TEST 3 (K3 session hijack, per 15/30-min window): models A-win and X4-win
on the 174 users that meet Landauer's victim rules; budget <= 1.75 false alarms per day.
  Attack: lib_k3_attack.py
  Output: results/k3_window.json (key test_seed1 = reported numbers)
  Run   : python -W ignore src/step10_test3_k3_window_Awin_X4win.py

------------------------------------------------------------------------
K3-b: per-window detection (15 and 30 min) on K3 session hijacks, as fixed in
docs/SPEC_K3_window.md (approved 2026-10-24).

Scope: the 174 users meeting Landauer's victim rules. Unit: an active user-window.
Models
  A-win : Landauer's detector on window count vectors in its update + FIFO mode
          (detect.py -u -f): queue = the user's last 30 windows; a = min L1/limit.
  X4-win: relative Set A per window (log1p(x) - log1p(median of the user's previous 30
          windows)) + 4 window context features; one global IF (200 trees, 256 samples,
          60-day window, retrain 7 days; percentile reference <= 20,000 training windows);
          per-user calibration (previous 30 scored windows); R = 0.5a + 0.4c + 0.1h, lambda 0.3.
          Context: c1 volume robust z (log1p, MAD floor 0.5, clip 10), c2 share of path
          events in folders the user never touched, c3 excess sensitive share, c4 window hour
          outside the user's 2.5-97.5 percentile hours (previous 200 windows). Flags:
          c1>=3, c2>=0.5, c3>=0.1, c4=1, weight 2 each -> c = flags/4.
Implementation: IF models are trained once on clean data per window size; attack windows
are scored by models trained before the attack, so their scores equal a per-run retrain;
injected windows are only left out of later training sets.
Threshold per model: on seeds 0+2 (3 intensities each), the threshold with false alarms
<= 1.75 per day that detects the most sessions (ties: lower median latency). Reported on seed 1.
"""
import config  # data / work / results paths (see config.py)
import json
import math
import os
import time
from collections import Counter, deque

import duckdb
import numpy as np
from sklearn.ensemble import IsolationForest

import lib_k3_attack as k3  # K3 session-hijack generator
import lib_models as lib  # features F1-F8, models B/C/D/X4, Landauer detector, evaluator
from step04_daily_tables import SENSITIVE

PARQUET = config.PARQUET
DAYS = 1910
BUDGET = 1.75
VAL, TEST = [0, 2], 1
OUT = os.path.join(config.RESULTS, 'k3_window.json')
LOG = open(os.path.join(config.RESULTS, 'k3_window.log'), 'a')
SENS = set(SENSITIVE)


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


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
    """New window list for one user with the attack events added."""
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


def user_features(wl, N=30, M=5, NH=200):
    """Features for one user's windows. Returns (X rows, c, a_landauer). Row 0 = NaN."""
    n = len(wl)
    T = np.array([x.types for x in wl])
    L = np.log1p(T)
    X = np.full((n, T.shape[1] + 4), np.nan, dtype=np.float32)
    flags = np.zeros(n)
    aL = np.full(n, np.nan)
    seen = set()
    hourh = deque()
    hcount = np.zeros(24)
    logtot = deque(maxlen=N)
    sratio = deque(maxlen=N)
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
            c2 = (sum(c for p, c in x.pref.items() if p not in seen) / sum(x.pref.values())) if x.pref else 0.0
            c3 = max(0.0, x.sens / x.total - (sum(sratio) / len(sratio)))
            c4 = 0.0
            if len(hourh) >= M:
                cum = np.cumsum(hcount); tot = cum[-1]
                lo_h = int(np.searchsorted(cum, 0.025 * tot)); hi_h = int(np.searchsorted(cum, 0.975 * tot))
                c4 = float(not (lo_h <= x.hour <= hi_h))
            X[i, T.shape[1]:] = (c1, c2, c3, c4)
            flags[i] = (c1 >= 3) + (c2 >= 0.5) + (c3 >= 0.1) + (c4 >= 1)
            Q = T[lo:i]
            aL[i] = float(np.min(np.abs(Q - T[i]).sum(1) / np.maximum(Q, T[i]).sum(1)))
        seen.update(x.pref)
        hourh.append((x.hour, x.total)); hcount[x.hour] += x.total
        if len(hourh) > NH:
            h0, t0 = hourh.popleft(); hcount[h0] -= t0
        logtot.append(math.log1p(x.total)); sratio.append(x.sens / x.total)
    return X, flags / 4.0, aL


def calib_risk(a, c, N=30, M=5, w=(0.5, 0.4, 0.1), lam=0.3):
    """Per-user calibration of a, then R with history h (one user's sequence)."""
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
    """IF models per weekly retrain on clean windows; score any row by its window time."""

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


def evaluate(scores, wl_by_user, plans, W, theta, strict):
    """Sessions detected, latencies (min) and false alarms per day for one run."""
    fp = 0
    att = {p['victim']: (int(p['start'] // (60 * W)), int((p['start'] + p['dur'] - 1e-9) // (60 * W)), p['start']) for p in plans}
    det, lat = 0, []
    for u, s in scores.items():
        alert = (s > theta) if strict else (s >= theta)
        alert &= ~np.isnan(s)
        ws = np.array([x.w for x in wl_by_user[u]])
        if u in att:
            w0, w1, st = att[u]
            in_att = (ws >= w0) & (ws <= w1)
            fp += int(np.sum(alert & ~in_att))
            hit = np.flatnonzero(alert & in_att)
            if len(hit):
                det += 1
                lat.append(((ws[hit[0]] + 1) * 60 * W - st) / 60)
        else:
            fp += int(np.sum(alert))
    return det, lat, fp / DAYS


T0 = time.time()
BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
ELIG = k3.eligible_users()
W30 = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = {s: k3.plan_attacks(s, BASE, W30, ELIG) for s in VAL + [TEST]}
EVENTS = {(s, lv): {p['victim']: k3.attack_events(p, lv, s, BASE, ALL_PREFIX) for p in PLANS[s]}
          for s in VAL + [TEST] for lv in k3.LEVELS}
GRID_A = [round(x, 3) for x in np.arange(0.30, 0.995, 0.01)] + [0.995, 0.999]
GRID_R = GRID_A
res = {'meta': __doc__.strip()}

for W in (30, 15):
    t = time.time()
    WINS = load_windows(ELIG, W, TYPES)
    n_rows = sum(len(v) for v in WINS.values())
    log(f'W={W}: {len(WINS)} users, {n_rows:,} active windows, loaded {time.time() - t:.0f}s')
    t = time.time()
    FE = {u: user_features(wl) for u, wl in WINS.items()}
    log(f'W={W}: clean features {time.time() - t:.0f}s')
    t = time.time()
    users = sorted(WINS)
    rows_t = np.concatenate([[x.w for x in WINS[u]] for u in users])
    Xall = np.vstack([FE[u][0] for u in users])
    FE = {u: (None, FE[u][1], FE[u][2]) for u in users}   # free the per-user copies
    gif = GlobalIF(rows_t, Xall, W)
    a_all = gif.score(rows_t, Xall)
    del Xall
    log(f'W={W}: global IF ({len(gif.models)} models) + scoring {time.time() - t:.0f}s')
    CLEAN_A, CLEAN_R, off = {}, {}, 0
    for u in users:
        n = len(WINS[u])
        CLEAN_A[u] = FE[u][2]
        CLEAN_R[u] = calib_risk(a_all[off:off + n], FE[u][1])
        off += n

    def run_scores(seed, level):
        sA, sR, wl_by = dict(CLEAN_A), dict(CLEAN_R), dict(WINS)
        for v, ev in EVENTS[(seed, level)].items():
            wl = inject(WINS[v], ev, W, TIX)
            Xv, cv, aLv = user_features(wl)
            av = gif.score(np.array([x.w for x in wl]), Xv)
            sA[v], sR[v], wl_by[v] = aLv, calib_risk(av, cv), wl
        return sA, sR, wl_by

    t = time.time()
    RUNS = {(s, lv): run_scores(s, lv) for s in VAL + [TEST] for lv in k3.LEVELS}
    log(f'W={W}: per-run victim rescoring {time.time() - t:.0f}s')
    out = {}
    # A-win = Landauer's distance per window (score index 0), X4-win = calibrated risk R (index 1);
    # each gets the threshold that detects most sessions on seeds 0+2 within the alarm budget
    for name, key, grid, strict in (('A-win', 0, GRID_A, True), ('X4-win', 1, GRID_R, False)):
        cands = []
        for th in grid:
            dets, lats, fps = 0, [], []
            for s in VAL:
                for lv in k3.LEVELS:
                    sc = RUNS[(s, lv)]
                    d_, l_, f_ = evaluate(sc[key], sc[2], PLANS[s], W, th, strict)
                    dets += d_; lats += l_; fps.append(f_)
            cands.append((th, dets, float(np.median(lats)) if lats else float('inf'), float(np.mean(fps))))
        ok = [c for c in cands if c[3] <= BUDGET]
        th, vdet, vlat, vfp = (max(ok, key=lambda c: (c[1], -c[2], -c[3])) if ok else min(cands, key=lambda c: c[3]))
        test = {}
        for lv in k3.LEVELS:
            sc = RUNS[(TEST, lv)]
            d_, l_, f_ = evaluate(sc[key], sc[2], PLANS[TEST], W, th, strict)
            test[lv] = {'detected': d_, 'of': len(PLANS[TEST]), 'latency_median_min': float(np.median(l_)) if l_ else None,
                        'latency_p90_min': float(np.percentile(l_, 90)) if l_ else None, 'false_alarms_per_day': f_,
                        'latencies_min': [round(x, 1) for x in l_]}
        out[name] = {'theta': th, 'validation': {'detected_of_120': vdet, 'latency_median_min': vlat, 'false_alarms_per_day': vfp},
                     'test_seed1': test}
        log(f"W={W} {name}: theta={th} | val det {vdet}/120 lat {vlat:.0f} min fa/day {vfp:.2f} | TEST " +
            ' | '.join(f"{lv}: {v['detected']}/20 lat {v['latency_median_min']} fa/day {v['false_alarms_per_day']:.2f}" for lv, v in test.items()))
    res[f'W{W}'] = {'active_windows': n_rows, 'models': out}
    json.dump(res, open(OUT, 'w'), indent=1, default=str)
log(f'K3_WINDOW_DONE in {time.time() - T0:.0f}s')
