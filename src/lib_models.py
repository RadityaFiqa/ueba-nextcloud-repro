"""LIBRARY for the three models of the thesis (imported as `lib`).
  base data and uid swap ..... load_base(), seed_swaps(), relabel(), labels_from_swaps()
  context features F1-F8 ..... compute_features()        (IF Personal, Model Utama)
  49 user-relative counts .... relative_A()              (IF Personal, Model Utama)
  Isolation Forest score ..... iforest_a()               (IF Personal, Model Utama)
  per-user ranking a ......... calibrate_per_user()      (IF Personal, Model Utama)
  risk score R, history h .... risk()                    (Model Utama; flags k1-k6 are in step12_final_eval.py)
  Landauer detector .......... landauer_detect()         (Pembanding Landauer)
  Landauer evaluation ........ evaluate(), metrics()     (all models)

Implementation details:
  * hour-of-day uses 5-minute bins (F4 percentiles at 5-min resolution);
  * F8 peak uses fixed 10-minute bins (pairs of 5-min bins);
  * F1, F5 and F8 use a robust z (median / MAD, clip +-10).
"""
import math
from collections import Counter, deque

import duckdb
import numpy as np
from sklearn.ensemble import IsolationForest

from lib_uid_swap import load_users, select_pairs  # noqa: E402

NBIN = 288   # 5-minute bins per day
# Contextual features (all compare today with the user's own previous W active days):
#   F1 volume (robust z of log events)      F5 active span (robust z)
#   F2 action mix change (Jensen-Shannon)   F6 share of events in folders never used before
#   F3 share of never-used action types     F7 extra share of sensitive actions (share, delete, ...)
#   F4 share of events outside usual hours  F8 burst: busiest 10 minutes (robust z)
FNAMES = ['F1_vol_z', 'F2_type_js', 'F3_new_type_share', 'F4_off_hours', 'F5_span_z',
          'F6_new_prefix_share', 'F7_sens_excess', 'F8_burst_z']


# ---------------------------------------------------------------- base data
class Rec:
    __slots__ = ('d', 'types', 'total', 'first', 'last', 'sens', 'pathcnt', 'hist', 'pref', 'peak10')


def load_base(base_dir):
    """uid -> {d: Rec} from the base tables (original uids, no swap)."""
    con = duckdb.connect()
    con.execute("SET threads=2")
    types = [r[0] for r in con.execute(
        f"SELECT DISTINCT type FROM read_parquet('{base_dir}/types.parquet') ORDER BY 1").fetchall()]
    con.execute(f"""CREATE TEMP TABLE ds AS
        SELECT row_number() OVER (ORDER BY uid, d) - 1 AS idx, * FROM read_parquet('{base_dir}/daystat.parquet')""")
    con.execute("CREATE TEMP TABLE ty AS SELECT * FROM (VALUES " +
                ', '.join(f"('{t}', {i})" for i, t in enumerate(types)) + ") v(type, ti)")
    ds = con.execute("SELECT idx, uid, d, total, first_mi, last_mi, sens, pathcnt FROM ds ORDER BY idx").fetchall()
    n = len(ds)
    TY = np.zeros((n, len(types)))
    q = con.execute(f"""SELECT ds.idx, ty.ti, t.cnt FROM read_parquet('{base_dir}/types.parquet') t
                        JOIN ds USING (uid, d) JOIN ty USING (type)""").fetchnumpy()
    np.add.at(TY, (q['idx'], q['ti']), q['cnt'])
    HI = np.zeros((n, NBIN))
    q = con.execute(f"""SELECT ds.idx, h.bin, h.cnt FROM read_parquet('{base_dir}/hist.parquet') h
                        JOIN ds USING (uid, d)""").fetchnumpy()
    np.add.at(HI, (q['idx'], q['bin'].astype(np.int64)), q['cnt'])
    del q
    PREF = [dict() for _ in range(n)]
    for idx, p, c in con.execute(f"""SELECT ds.idx, p.prefix, p.cnt FROM read_parquet('{base_dir}/prefix.parquet') p
                                     JOIN ds USING (uid, d)""").fetchall():
        PREF[idx][p] = c
    PEAK = HI.reshape(n, -1, 2).sum(2).max(1)
    base = {}
    for idx, uid, d, total, fmi, lmi, sens, pc in ds:
        r = Rec()
        r.d, r.total, r.first, r.last, r.sens, r.pathcnt = d, total, fmi, lmi, sens, pc
        r.types, r.hist, r.pref, r.peak10 = TY[idx], HI[idx], PREF[idx], float(PEAK[idx])
        base.setdefault(uid, {})[d] = r
    return base, types


def seed_swaps(stats_path, seed):
    users = load_users(stats_path)
    pairs, ct = select_pairs(users, seed, log=lambda s: None)
    return pairs, ct


def relabel(base, pairs, ct):
    """Apply the seed's uid swap at day level -> uid -> list of Rec sorted by d."""
    out = {}
    for uid, days in base.items():
        for d, r in days.items():
            new = pairs[uid] if uid in ct and d * 86400 >= ct[uid] else uid
            tgt = out.setdefault(new, {})
            if d in tgt:  # never happens for Landauer swaps (see notes), merged for safety
                m = tgt[d]
                n = Rec()
                n.d = d
                n.types, n.hist = m.types + r.types, m.hist + r.hist
                n.total, n.sens, n.pathcnt = m.total + r.total, m.sens + r.sens, m.pathcnt + r.pathcnt
                n.first, n.last = min(m.first, r.first), max(m.last, r.last)
                n.pref = dict(Counter(m.pref) + Counter(r.pref))
                n.peak10 = float(n.hist.reshape(-1, 2).sum(1).max())
                tgt[d] = n
            else:
                tgt[d] = r
    return {u: [days[d] for d in sorted(days)] for u, days in out.items()}


def read_labels(path):
    lab = {}
    for line in open(path):
        if line.strip():
            u, t = line.strip().split(',')
            lab[u] = int(float(t)) // 86400
    return lab


def labels_from_swaps(pairs, ct):
    return {pairs[u]: int(t) // 86400 for u, t in ct.items()}


# ---------------------------------------------------------------- features
def robust_z(x, hist_vals, M):
    if len(hist_vals) < M:
        return 0.0
    v = np.asarray(hist_vals)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return float(np.clip((x - med) / (1.4826 * mad + 1e-6), -10, 10))


def js_div(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_features(user_recs, W=30, M=5):
    """Causal features per user-day. Returns dict of aligned arrays (sorted uid, d)."""
    uids, ds, first, last, A, F = [], [], [], [], [], []
    for uid in sorted(user_recs):
        recs = user_recs[uid]
        win = deque()
        tsum = np.zeros_like(recs[0].types)
        hsum = np.zeros(NBIN)
        pref = Counter()
        logtot, span, peak, sratio = deque(), deque(), deque(), deque()
        n = len(recs)
        for i, r in enumerate(recs):
            f = np.full(8, np.nan)
            if win:   # the user's first day has no history -> features stay NaN
                lt = math.log1p(r.total)
                f[0] = robust_z(lt, logtot, M)
                f[1] = js_div(r.types, tsum)
                f[2] = float(r.types[tsum == 0].sum() / r.total)
                cum = np.cumsum(hsum)
                N = cum[-1]
                lo = int(np.searchsorted(cum, 0.025 * N))
                hi = int(np.searchsorted(cum, 0.975 * N))
                inside = r.hist[lo:hi + 1].sum()
                f[3] = float((r.total - inside) / r.total)
                f[4] = robust_z((r.last - r.first) / 60.0, span, M)
                if r.pathcnt > 0 and r.pref:
                    novel = sum(c for p, c in r.pref.items() if pref.get(p, 0) <= 0)
                    f[5] = novel / sum(r.pref.values())
                else:
                    f[5] = 0.0
                rt = r.sens / r.total
                f[6] = max(0.0, rt - (sum(sratio) / len(sratio)))
                f[7] = robust_z(r.peak10, peak, M)
            uids.append(uid); ds.append(r.d); first.append(i == 0); last.append(i == n - 1)
            A.append(r.types); F.append(f)
            # push today into history, pop beyond W active days
            win.append(r); tsum = tsum + r.types; hsum += r.hist; pref.update(r.pref)
            logtot.append(math.log1p(r.total)); span.append((r.last - r.first) / 60.0)
            peak.append(r.peak10); sratio.append(r.sens / r.total)
            if len(win) > W:
                o = win.popleft()
                tsum = tsum - o.types; hsum -= o.hist; pref.subtract(o.pref)
                logtot.popleft(); span.popleft(); peak.popleft(); sratio.popleft()
    return dict(uid=np.array(uids, dtype=object), d=np.array(ds), first=np.array(first), last=np.array(last),
                A=np.array(A, dtype=np.float32), F=np.array(F, dtype=np.float32))


# ---------------------------------------------------------------- Isolation Forest
def iforest_a(feat, X, T=60, step=7, first_after=7, random_state=0):
    """a_raw in [0,1]: share of the training window's days that are less anomalous than the day. NaN = no model yet.

    ONE Isolation Forest (200 trees, 256 samples) on the user-relative inputs of all users, trained on the
    previous T = 60 days and retrained every `step` = 7 days. Every input row is already relative to its
    own user's history, so the model compares deviations, not users."""
    d = feat['d']
    valid = ~feat['first']  # first days have no features
    a = np.full(len(d), np.nan)
    d0 = int(d.min())
    for r in range(d0 + first_after, int(d.max()) + 1, step):
        tr = valid & (d >= max(d0, r - T)) & (d < r)
        te = valid & (d >= r) & (d < r + step)
        if tr.sum() < 2 or te.sum() == 0:
            continue
        m = IsolationForest(n_estimators=200, max_samples=min(256, int(tr.sum())),
                            random_state=random_state, n_jobs=2).fit(X[tr])
        s_tr = np.sort(m.score_samples(X[tr]))
        s_te = m.score_samples(X[te])
        # share of training days that are less anomalous (higher score) than the test day
        a[te] = (len(s_tr) - np.searchsorted(s_tr, s_te, side='right')) / len(s_tr)
    return a


# ---------------------------------------------------------------- risk scoring
def risk(feat, a, c, w=(0.5, 0.4, 0.1), lam=0.3):
    """Model Utama: R = 0.5*a + 0.4*c + 0.1*h, h_t = 0.3*h_{t-1} + 0.7*R_{t-1} per user (active days).
    Other w are used only by the RM2 ablation (without c, without h)."""
    wa, wc, wh = (x / sum(w) for x in w)
    uid, first = feat['uid'], feat['first']
    R = np.full(len(a), np.nan)
    h = 0.0
    for i in range(len(a)):
        if first[i]:
            h = 0.0
            continue
        ai = 0.0 if wa == 0 else a[i]
        if np.isnan(ai):
            continue
        R[i] = wa * ai + wc * c[i] + wh * h
        h = lam * h + (1 - lam) * R[i]
    return R


# ---------------------------------------------------------------- user-relative inputs and per-user ranking
def relative_A(feat, W=14):
    """The 49 daily event-type counts made relative to the user's own history (causal):
    rel[t, k] = log1p(count_t,k) - log1p(median of count_k over the user's previous
    W active days). NaN on the user's first day (no history)."""
    A = feat['A'].astype(np.float64)
    out = np.full_like(A, np.nan)
    starts = np.flatnonzero(feat['first'])
    for s, e in zip(starts, list(starts[1:]) + [len(A)]):
        L = np.log1p(A[s:e])
        for i in range(1, e - s):
            out[s + i] = L[i] - np.log1p(np.median(A[s + max(0, i - W):s + i], axis=0))
    return out


def calibrate_per_user(feat, a, N=30, M=5):
    """a = mid-rank of today's a among the user's previous
    N scored days (strictly before t). With < M previous scores, a = a_raw (fallback)."""
    out = a.copy()
    starts = np.flatnonzero(feat['first'])
    for s, e in zip(starts, list(starts[1:]) + [len(a)]):
        hist = []
        for i in range(s, e):
            if np.isnan(a[i]):
                continue
            if len(hist) >= M:
                h = np.asarray(hist[-N:])
                out[i] = (np.sum(h < a[i]) + 0.5 * np.sum(h == a[i])) / len(h)
            hist.append(a[i])
    return out


# ---------------------------------------------------------------- Landauer detector
def landauer_detect(feat, theta=0.6, r=1, queue=30, labels=None):
    """Pembanding Landauer (detect.py mode 1): a = min L1 distance / limit between today's 49 counts and the
    user's queue of up to `queue` daily vectors; alarm if a > theta. The decision drives Landauer's retraining
    and queue updates exactly as detect.py does (same TP/FP as step 5, check V4).
    Returns (status, decision, metrics, a)."""
    A = feat['A'].astype(np.float64)    # float64 like detect.py's Python floats (thresholds are exact)
    first, last = feat['first'], feat['last']
    n = len(A)
    status = np.full(n, '', dtype=object)
    dec = np.full(n, '', dtype=object)
    a_out = np.full(n, np.nan)
    i = 0
    while i < n:
        j = i + 1
        while j < n and not first[j]:
            j += 1
        Q = []                      # user's model: list of count vectors (oldest first)
        wait = r
        for t in range(i, j):
            if last[t]:
                break               # detect.py never closes the last day
            x = A[t]
            a = None
            mi = None
            if Q:
                M = np.vstack(Q)
                manh = np.abs(M - x).sum(1)
                limit = np.maximum(M, x).sum(1)
                dist = manh / limit
                mi = int(np.argmin(dist))       # first minimum == detect.py's strict '<' scan
                a = float(dist[mi])
            status[t] = 'detection' if wait <= 0 else 'training'
            alert = a is not None and a > theta  # empty model -> normal (as detect.py)
            if a is not None:
                a_out[t] = a
            if alert:
                dec[t] = 'anomalous'
                wait = r
            else:
                dec[t] = 'normal'
                wait -= 1
                if mi is not None:              # move matching vector to the end (fifo=False)
                    Q.append(Q.pop(mi))
            if wait >= 0:                       # re-training: add today's vector
                Q.append(x)
            if queue != -1 and len(Q) > queue:
                Q.pop(0)
        i = j
    m = metrics(feat['uid'], feat['d'], status, dec, labels) if labels is not None else None
    return status, dec, m, a_out


# ---------------------------------------------------------------- evaluation
def evaluate(feat, alert, avail, labels, r=1):
    """Landauer's protocol on per-user-day decisions.

    alert: bool array (anomalous decision); avail: bool array, False where the
    model has no decision (forced training day, e.g. before the first IF model);
    a user's first day is always 'normal' (no history), the last day is never
    scored (detect.py never closes it). Retraining mode as detect.py: anomalous
    -> wait = r, normal -> wait -= 1, status training while wait > 0.
    """
    uid, d, first, last = feat['uid'], feat['d'], feat['first'], feat['last']
    status = np.full(len(d), '', dtype=object)
    dec = np.full(len(d), '', dtype=object)
    wait = r
    for i in range(len(d)):
        if first[i]:
            wait = r
        if last[i]:
            continue
        forced = (not first[i]) and (not avail[i])
        status[i] = 'detection' if (wait <= 0 and not forced) else 'training'
        if (not first[i]) and avail[i] and alert[i]:
            dec[i] = 'anomalous'
            wait = r
        else:
            dec[i] = 'normal'
            wait -= 1
    return status, dec, metrics(uid, d, status, dec, labels)


def metrics(uid, d, status, dec, labels):
    c = dict(tp=0, tp_adj=0, fp=0, tn=0, fn=0, fn_adj=0, train=0, detect=0)
    for i in range(len(d)):
        s = status[i]
        if not s:
            continue
        if s == 'training':
            c['train'] += 1
        else:
            c['detect'] += 1
        pos = labels.get(uid[i]) == d[i]
        if pos:
            if s == 'detection' and dec[i] == 'anomalous':
                c['tp'] += 1; c['tp_adj'] += 1
            elif dec[i] == 'normal':
                c['fn'] += 1; c['fn_adj'] += 1
            else:  # training + anomalous
                c['tp_adj'] += 1; c['fn'] += 1
        elif s == 'detection':
            if dec[i] == 'anomalous':
                c['fp'] += 1
            else:
                c['tn'] += 1
    div = lambda x, y: x / y if y else float('nan')
    alerts = c['tp'] + c['fp']
    return dict(TP=c['tp'], FN=c['fn'], FP=c['fp'], TN=c['tn'], TP_adj=c['tp_adj'], FN_adj=c['fn_adj'],
                train=c['train'], detect=c['detect'],
                TPR=div(c['tp'], c['tp'] + c['fn']), TPR_adj=div(c['tp_adj'], c['tp_adj'] + c['fn_adj']),
                FPR=div(c['fp'], c['fp'] + c['tn']), Prec=div(c['tp_adj'], c['tp_adj'] + c['fp']),
                R=div(c['train'], c['train'] + c['detect']), alerts=alerts, alerts_per_day=alerts / 1910)
