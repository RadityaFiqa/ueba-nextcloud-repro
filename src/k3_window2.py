"""K3-b v2 (pipeline/SPEC_K3_window_v2.md, fixed 2026-10-24 before running).
Models A-win, X4-win (unchanged), X5-base, X5-acc, X5-full on 30- and 60-min windows.
Tuning on seeds 0+2 (budget <= 1.75 false alarms/day); main test on unseen seeds 3-9
(420 sessions); seed 1 reported separately (already seen)."""
import os
import config  # data / work / results paths (see config.py)
import json
import time

import numpy as np
from scipy.stats import binomtest

import k3_common as k3
import lib
from k3w_lib import GlobalIF, calib_risk, inject, load_windows, user_features, user_features_v2

VAL, SEED1, TEST = [0, 2], [1], list(range(3, 10))
SEEDS = VAL + SEED1 + TEST
BUDGET, DAYS = 1.75, 1910
OUT = os.path.join(config.RESULTS, 'k3_window2.json')
LOG = open(os.path.join(config.RESULTS, 'k3_window2.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def accumulate(ws, s, K):
    ok = ~np.isnan(s)
    if K <= 1:
        return s
    v = np.where(ok, s, 0.0)
    acc = np.empty(len(v))
    j, run = 0, 0.0
    for i in range(len(v)):
        run += v[i]
        while ws[j] <= ws[i] - K:
            run -= v[j]; j += 1
        acc[i] = run
    acc[~ok] = np.nan
    return acc


def emitted(ws, v, theta, strict, sup):
    ok = ~np.isnan(v)
    hit = ok & ((v > theta) if strict else (v >= theta))
    idx = np.flatnonzero(hit)
    if sup <= 0 or len(idx) == 0:
        return idx
    out, last = [], None
    for i in idx:
        if last is None or ws[i] - last >= sup:
            out.append(i); last = ws[i]
    return np.asarray(out, dtype=int)


def evaluate(V, WS, plans, W, theta, strict, K, sup):
    att = {p['victim']: (int(p['start'] // (60 * W)), int((p['start'] + p['dur'] - 1e-9) // (60 * W)), p['start']) for p in plans}
    fp, det = 0, {}
    for u, v in V.items():
        ws = WS[u]
        idx = emitted(ws, v, theta, strict, sup)
        if u in att:
            w0, w1, st = att[u]
            m = (ws[idx] >= w0) & (ws[idx] <= w1 + K - 1)
            fp += int(np.sum(~m))
            hits = idx[m]
            det[u] = (((ws[hits[0]] + 1) * 60 * W - st) / 60) if len(hits) else None
        else:
            fp += len(idx)
    return det, fp / DAYS


T0 = time.time()
BASE, TYPES = lib.load_base(config.BASE)
TIX = {t: i for i, t in enumerate(TYPES)}
ELIG = k3.eligible_users()
W30C = k3.win30_counts(ELIG)
ALL_PREFIX = sorted({p for days in BASE.values() for r in days.values() for p in r.pref})
PLANS = {s: k3.plan_attacks(s, BASE, W30C, ELIG) for s in SEEDS}
EVENTS = {(s, lv): {p['victim']: k3.attack_events(p, lv, s, BASE, ALL_PREFIX) for p in PLANS[s]}
          for s in SEEDS for lv in k3.LEVELS}
log(f'setup {time.time() - T0:.0f}s; {len(SEEDS)} seeds x {len(k3.LEVELS)} levels')
res = {'meta': __doc__.strip()}

for W in (30, 60):
    K, SUP = (3, 8) if W == 30 else (2, 4)
    t = time.time()
    WINS = load_windows(ELIG, W, TYPES)
    users = sorted(WINS)
    log(f'W={W}: {sum(len(v) for v in WINS.values()):,} windows, loaded {time.time() - t:.0f}s')
    t = time.time()
    F4 = {u: user_features(WINS[u]) for u in users}
    F5 = {u: user_features_v2(WINS[u]) for u in users}
    rows_t = np.concatenate([[x.w for x in WINS[u]] for u in users])
    log(f'W={W}: clean features {time.time() - t:.0f}s')
    t = time.time()
    X4 = np.vstack([F4[u][0] for u in users]); g4 = GlobalIF(rows_t, X4, W); a4 = g4.score(rows_t, X4); del X4
    X5 = np.vstack([F5[u][0] for u in users]); g5 = GlobalIF(rows_t, X5, W); a5 = g5.score(rows_t, X5); del X5
    log(f'W={W}: two global IFs + scoring {time.time() - t:.0f}s')
    CLEAN = {'A': {}, 'X4': {}, 'X5': {}}
    WS0 = {}
    off = 0
    for u in users:
        n = len(WINS[u])
        WS0[u] = np.array([x.w for x in WINS[u]])
        CLEAN['A'][u] = F4[u][2]
        CLEAN['X4'][u] = calib_risk(a4[off:off + n], F4[u][1])
        CLEAN['X5'][u] = calib_risk(a5[off:off + n], F5[u][1])
        off += n
    del F4, F5
    CLEAN['X5acc'] = {u: accumulate(WS0[u], CLEAN['X5'][u], K) for u in users}

    t = time.time()
    RUNS = {}
    for s in SEEDS:
        for lv in k3.LEVELS:
            ov = {'A': {}, 'X4': {}, 'X5': {}, 'X5acc': {}, 'WS': {}}
            for v, ev in EVENTS[(s, lv)].items():
                wl = inject(WINS[v], ev, W, TIX)
                ws = np.array([x.w for x in wl])
                X4v, c4v, aLv = user_features(wl)
                X5v, c5v = user_features_v2(wl)
                ov['A'][v] = aLv
                ov['X4'][v] = calib_risk(g4.score(ws, X4v), c4v)
                ov['X5'][v] = calib_risk(g5.score(ws, X5v), c5v)
                ov['X5acc'][v] = accumulate(ws, ov['X5'][v], K)
                ov['WS'][v] = ws
            RUNS[(s, lv)] = ov
    log(f'W={W}: victim rescoring for {len(RUNS)} runs {time.time() - t:.0f}s')

    def view(s, lv, key):
        ov = RUNS[(s, lv)]
        V = dict(CLEAN[key]); V.update(ov[key])
        WS = dict(WS0); WS.update(ov['WS'])
        return V, WS

    MODELS = {  # name: (score key, strict, K, suppress, grid)
        'A-win': ('A', True, 1, 0, 'single'), 'X4-win': ('X4', False, 1, 0, 'single'),
        'X5-base': ('X5', False, 1, 0, 'single'), 'X5-acc': ('X5acc', False, K, 0, 'acc'),
        'X5-full': ('X5acc', False, K, SUP, 'acc')}
    GRIDS = {'single': [round(x, 3) for x in np.arange(0.30, 0.995, 0.01)] + [0.995, 0.999],
             'acc': [round(x, 3) for x in np.linspace(0.30, float(K), 136)]}
    out, DET = {}, {}
    for name, (key, strict, KK, sup, grid) in MODELS.items():
        views = {(s, lv): view(s, lv, key) for s in SEEDS for lv in k3.LEVELS}
        cands = []
        for th in GRIDS[grid]:
            dets, lats, fas = 0, [], []
            for s in VAL:
                for lv in k3.LEVELS:
                    V, WS = views[(s, lv)]
                    d, fa = evaluate(V, WS, PLANS[s], W, th, strict, KK, sup)
                    dets += sum(x is not None for x in d.values()); lats += [x for x in d.values() if x is not None]; fas.append(fa)
            cands.append((th, dets, float(np.median(lats)) if lats else 1e9, float(np.mean(fas))))
        ok = [c for c in cands if c[3] <= BUDGET]
        th, vdet, vlat, vfa = max(ok, key=lambda c: (c[1], -c[2], -c[3])) if ok else min(cands, key=lambda c: c[3])

        def summarize(seeds):
            per_lv, flat, fas, lat_all = {}, {}, [], []
            for lv in k3.LEVELS:
                dl, ll = 0, []
                for s in seeds:
                    V, WS = views[(s, lv)]
                    d, fa = evaluate(V, WS, PLANS[s], W, th, strict, KK, sup)
                    fas.append(fa)
                    for u, x in d.items():
                        flat[(s, lv, u)] = x is not None
                        if x is not None:
                            dl += 1; ll.append(x)
                per_lv[lv] = {'detected': dl, 'of': 20 * len(seeds), 'latency_median_min': float(np.median(ll)) if ll else None}
                lat_all += ll
            return {'detected': sum(flat.values()), 'of': len(flat), 'latency_median_min': float(np.median(lat_all)) if lat_all else None,
                    'latency_p90_min': float(np.percentile(lat_all, 90)) if lat_all else None,
                    'false_alarms_per_day': float(np.mean(fas)), 'per_level': per_lv}, flat

        test, flat = summarize(TEST)
        s1, _ = summarize(SEED1)
        DET[name] = flat
        out[name] = {'theta': th, 'validation': {'detected_of_120': vdet, 'latency_median_min': vlat, 'false_alarms_per_day': vfa},
                     'test_seeds_3_9': test, 'seed1': s1}
        log(f"W={W} {name}: theta={th} | val {vdet}/120 lat {vlat:.0f} fa {vfa:.2f} | TEST 3-9 {test['detected']}/{test['of']} "
            f"lat med {test['latency_median_min']} p90 {test['latency_p90_min']} fa {test['false_alarms_per_day']:.2f} | seed1 {s1['detected']}/60")
    sign = {}
    for other in ('X4-win', 'A-win', 'X5-acc'):
        keys = DET['X5-full'].keys()
        b = sum(DET['X5-full'][k] and not DET[other][k] for k in keys)
        c = sum(DET[other][k] and not DET['X5-full'][k] for k in keys)
        sign[f'X5-full vs {other}'] = {'only_X5full': b, f'only_{other}': c, 'p_value': binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}
        log(f'W={W} sign test X5-full vs {other}: {b} vs {c}, p={sign[f"X5-full vs {other}"]["p_value"]:.3g}')
    res[f'W{W}'] = {'K': K, 'suppress_windows': SUP, 'models': out, 'sign_tests_seeds_3_9': sign}
    json.dump(res, open(OUT, 'w'), indent=1, default=str)
log(f'K3_WINDOW2_DONE in {time.time() - T0:.0f}s')
