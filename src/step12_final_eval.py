"""STEP 12b - FINAL experiment, evaluation (reads work/final/*.npz from step12_final_prep.py).

Models (all user-relative, no labels, no IP):
  Pembanding Landauer  nearest-day distance of the 49 daily counts, alarm if > 0.6 (Landauer et al., 2022)
  IF Personal          Isolation Forest on 49 user-relative counts + F1..F8, score ranked against the user's own
                       previous 30 days (a); alarm if a >= theta
  Model Utama          IF Personal + risk score R = .5a + .4c + .1h (lambda 0.3), c = raised flag weights / 10:
                       k1 |F1|>=3 (2), k2 F2>=.5 or F3>=.2 (2), k3 F4>=.3 (1), k4 F6>=.5 (2),
                       k5 F7>=.1 or F8>=3 (1), k6 trans_js>=.5 (2); alarm if R >= theta
Threshold rule (validation seeds 0+2, uid swaps and session hijacks): false alarms <= 1.75/day (Landauer's
budget) on the validation swaps, then highest mean of (swap TPR, hijack TPR); ties -> lower FPR.
Metrics: TPR, FPR (Landauer's protocol: adaptation days not scored), TPR_adj (attack day in adaptation still
counts when the decision was an alarm), FPR_adj (false alarms on every scored-or-adaptation normal day).
Outputs results/final.json and results/final_curve.csv:
  E1 thresholds, E2 seed 1 (both attacks, McNemar, bootstrap), E3 RM1 ablation, E4 RM2 ablation,
  E5 threshold curves, E6 uid swap on seeds 0-9 (Wilcoxon on 3-9).
  Run: python -W ignore src/step12_final_eval.py
"""
import csv
import json
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon

import config
import lib_models as lib

DIR = os.path.join(config.WORK, 'final')
DAYS, LEVELS, BUDGET, LAM = 1910, ['rendah', 'sedang', 'tinggi'], 1.75, 0.3
TH = [round(x, 2) for x in np.arange(0.40, 0.995, 0.01)]
L_TH = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
W_MAIN = (0.5, 0.4, 0.1)
FL = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6']
FW = {'k1': 2, 'k2': 2, 'k3': 1, 'k4': 2, 'k5': 1, 'k6': 2}
rng = np.random.default_rng(0)


def load(name):
    z = dict(np.load(os.path.join(DIR, name + '.npz')))
    z['feat'] = {'uid': z['uid'], 'd': z['d'], 'first': z['first'], 'last': z['last']}
    z['lab'] = dict(zip(z['lab_u'].tolist(), z['lab_d'].tolist()))
    z['pos'] = np.array([z['lab'].get(u) == d for u, d in zip(z['uid'], z['d'])])
    return z


def flags(z, drop=(), k5=('F7', 'F8')):
    g = lambda x: np.nan_to_num(x, nan=-np.inf)
    F, t = z['F'], z['trans_js']
    k5v = np.zeros(len(F), bool)
    if 'F7' in k5:
        k5v |= g(F[:, 6]) >= 0.1
    if 'F8' in k5:
        k5v |= g(F[:, 7]) >= 3
    K = {'k1': g(np.abs(F[:, 0])) >= 3, 'k2': (g(F[:, 1]) >= 0.5) | (g(F[:, 2]) >= 0.2), 'k3': g(F[:, 3]) >= 0.3,
         'k4': g(F[:, 5]) >= 0.5, 'k5': k5v, 'k6': g(t) >= 0.5}
    keep = [k for k in FL if k not in drop and not (k == 'k5' and not k5)]
    return sum(K[k] * FW[k] for k in keep) / float(sum(FW[k] for k in keep))


def risk(z, a, c, w=W_MAIN):
    return np.nan_to_num(lib.risk(z['feat'], a, c, w, LAM), nan=-1)


def metrics_from(z, alarm, train, valid):
    """TPR/FPR as Landauer (detection days), TPR_adj/FPR_adj (adaptation days included)."""
    pos, neg = z['pos'] & valid, ~z['pos'] & valid
    det = valid & ~train
    tp, tp_adj, npos = int((pos & det & alarm).sum()), int((pos & alarm).sum()), int(pos.sum())
    fp, tn = int((neg & det & alarm).sum()), int((neg & det & ~alarm).sum())
    fp_adj, n_adj = int((neg & alarm).sum()), int(neg.sum())
    return {'TP': tp, 'TP_adj': tp_adj, 'P': npos, 'TPR': tp / npos if npos else float('nan'), 'TPR_adj': tp_adj / npos if npos else float('nan'),
            'FP': fp, 'TN': tn, 'FPR': fp / (fp + tn), 'FP_adj': fp_adj, 'FPR_adj': fp_adj / n_adj, 'fp_per_day': fp / DAYS}


def ev(z, score, th):
    avail = ~np.isnan(z['a']) & ~z['first']
    st, dc, _ = lib.evaluate(z['feat'], score >= th, avail, z['lab'], r=1)
    alarm, train, valid = dc == 'anomalous', st == 'training', st != ''
    return metrics_from(z, alarm, train, valid), alarm, train, valid


def ev_landauer(z, th=0.6):
    alarm, train = z[f'L_alarm_{th}'], z[f'L_train_{th}']
    valid = ~z['last']
    return metrics_from(z, alarm, train, valid), alarm, train, valid


def r4(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def scores(z):
    a = z['a']
    return {'IF Personal': np.nan_to_num(a, nan=-1), 'Model Utama': risk(z, a, flags(z))}


# ---------------------------------------------------------------- E1 thresholds on seeds 0+2
VSW = {s: load(f'swap_s{s}') for s in (0, 2)}
VHJ = {(s, lv): load(f'hijack_s{s}_{lv}') for s in (0, 2) for lv in LEVELS}
VS = {k: scores(z) for k, z in list(VSW.items()) + list(VHJ.items())}
OUT = {'meta': __doc__.strip(), 'E1_thresholds': {}}
THETA = {}
for model in ['IF Personal', 'Model Utama']:
    best = None
    for th in TH:
        sw = [ev(VSW[s], VS[s][model], th)[0] for s in VSW]
        fpd = sum(m['FP'] for m in sw) / (2 * DAYS)
        if fpd > BUDGET:
            continue
        tpr_s = sum(m['TP'] for m in sw) / 40
        hj = sum(ev(VHJ[k], VS[k][model], th)[0]['TP'] for k in VHJ)
        fpr = sum(m['FP'] for m in sw) / sum(m['FP'] + m['TN'] for m in sw)
        cand = (((tpr_s + hj / 120) / 2), -fpr, th, {'swap_TP_val': int(tpr_s * 40), 'hijack_TP_val': hj, 'fp_per_day_val': round(fpd, 3)})
        if best is None or cand[:2] > best[:2]:
            best = cand
    THETA[model] = best[2]
    OUT['E1_thresholds'][model] = {'theta': best[2], **best[3]}
    print('E1', model, OUT['E1_thresholds'][model], flush=True)
del VSW, VHJ, VS

# ---------------------------------------------------------------- E2 seed 1, both attacks
SW1 = load('swap_s1')
HJ1 = {lv: load(f'hijack_s1_{lv}') for lv in LEVELS}
S1 = {'swap': scores(SW1), **{lv: scores(HJ1[lv]) for lv in LEVELS}}
E2, DEC = {}, {}
for model in ['Pembanding Landauer', 'IF Personal', 'Model Utama']:
    if model == 'Pembanding Landauer':
        m, al, tr, va = ev_landauer(SW1)
        hj = [ev_landauer(HJ1[lv]) for lv in LEVELS]
    else:
        m, al, tr, va = ev(SW1, S1['swap'][model], THETA[model])
        hj = [ev(HJ1[lv], S1[lv][model], THETA[model]) for lv in LEVELS]
    DEC[model] = {'swap': (al, tr, va), 'hijack': [(x[1], x[2], x[3]) for x in hj]}
    hm = [x[0] for x in hj]
    E2[model] = {'theta': 0.6 if model == 'Pembanding Landauer' else THETA[model], 'swap': r4(m),
                 'hijack': {'TP': sum(x['TP'] for x in hm), 'TP_adj': sum(x['TP_adj'] for x in hm), 'P': 60,
                            'TPR': round(sum(x['TP'] for x in hm) / 60, 4), 'TPR_adj': round(sum(x['TP_adj'] for x in hm) / 60, 4),
                            'per_level': {lv: x['TP'] for lv, x in zip(LEVELS, hm)},
                            'FPR': round(float(np.mean([x['FPR'] for x in hm])), 4), 'FPR_adj': round(float(np.mean([x['FPR_adj'] for x in hm])), 4),
                            'fp_per_day': round(float(np.mean([x['fp_per_day'] for x in hm])), 3)}}
    print('E2', model, 'swap', {k: E2[model]['swap'][k] for k in ('TP', 'TP_adj', 'TPR', 'FPR', 'TPR_adj', 'FPR_adj', 'fp_per_day')},
          '| hijack', {k: E2[model]['hijack'][k] for k in ('TP', 'TP_adj', 'per_level', 'FPR', 'FPR_adj')}, flush=True)


def mcnemar(x, y):
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    return {'hanya_model': b, 'hanya_landauer': c, 'p': float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def detected(model, kind):
    if kind == 'swap':
        al, tr, va = DEC[model]['swap']
        return al & ~tr & va & SW1['pos']
    return np.concatenate([al & ~tr & va & HJ1[lv]['pos'] for lv, (al, tr, va) in zip(LEVELS, DEC[model]['hijack'])])


STATS = {}
for model in ['IF Personal', 'Model Utama']:
    pos_sw = SW1['pos']
    pos_hj = np.concatenate([HJ1[lv]['pos'] for lv in LEVELS])
    al_m, tr_m, va_m = DEC[model]['swap']
    al_l, tr_l, va_l = DEC['Pembanding Landauer']['swap']
    neg = ~pos_sw & va_m & ~tr_m & va_l & ~tr_l
    STATS[model] = {'swap_deteksi': mcnemar(detected(model, 'swap')[pos_sw], detected('Pembanding Landauer', 'swap')[pos_sw]),
                    'hijack_deteksi': mcnemar(detected(model, 'hijack')[pos_hj], detected('Pembanding Landauer', 'hijack')[pos_hj]),
                    'swap_alarm_palsu': mcnemar(al_m[neg], al_l[neg])}
    # bootstrap (users) of the FPR difference on the uid-swap data, days scored by both models
    users = np.unique(SW1['uid'][neg])
    idx = {u: i for i, u in enumerate(users)}
    ui = np.array([idx[u] for u in SW1['uid'][neg]])
    fm, fl, n = np.bincount(ui, al_m[neg], len(users)), np.bincount(ui, al_l[neg], len(users)), np.bincount(ui, None, len(users))
    diffs = []
    for _ in range(2000):
        bs = rng.integers(0, len(users), len(users))
        diffs.append(fm[bs].sum() / n[bs].sum() - fl[bs].sum() / n[bs].sum())
    STATS[model]['selisih_FPR_vs_landauer'] = {'nilai': round(float(fm.sum() / n.sum() - fl.sum() / n.sum()), 4),
                                               'ci95': [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}
    print('STAT', model, STATS[model], flush=True)
OUT['E2_seed1'] = E2
OUT['E2_stats'] = STATS

# ---------------------------------------------------------------- E3 RM1 and E4 RM2 ablations (seed 1, theta of Model Utama)
th = THETA['Model Utama']


def ablate(make):
    sw = ev(SW1, make(SW1), th)[0]
    hj = [ev(HJ1[lv], make(HJ1[lv]), th)[0] for lv in LEVELS]
    return {'swap_TP': sw['TP'], 'swap_TP_adj': sw['TP_adj'], 'swap_TPR': round(sw['TPR'], 4), 'swap_FPR': round(sw['FPR'], 4),
            'swap_TPR_adj': round(sw['TPR_adj'], 4), 'swap_FPR_adj': round(sw['FPR_adj'], 4), 'fp_per_day': round(sw['fp_per_day'], 3),
            'hijack_TP': sum(x['TP'] for x in hj), 'hijack_TP_adj': sum(x['TP_adj'] for x in hj)}


RM1 = {'Model Utama lengkap': ablate(lambda z: risk(z, z['a'], flags(z)))}
for k in FL:
    RM1[f'tanpa tanda {k}'] = ablate(lambda z, k=k: risk(z, z['a'], flags(z, drop=(k,))))
GROUPS = {'jumlah (F1, k1)': ('a_tanpa_jumlah', dict(drop=('k1',))), 'pola (F2-F3, k2)': ('a_tanpa_pola', dict(drop=('k2',))),
          'waktu (F4-F5, k3)': ('a_tanpa_waktu', dict(drop=('k3',))), 'folder (F6, k4)': ('a_tanpa_folder', dict(drop=('k4',))),
          'tindakan berisiko (F7)': ('a_tanpa_risiko', dict(k5=('F8',))), 'lonjakan singkat (F8)': ('a_tanpa_lonjakan', dict(k5=('F7',))),
          '49 jumlah relatif': ('a_tanpa_jumlah_relatif', {})}
for g, (akey, fk) in GROUPS.items():
    RM1[f'tanpa fitur {g}'] = ablate(lambda z, akey=akey, fk=fk: risk(z, z[akey], flags(z, **fk)))
RM2 = {'Model Utama lengkap': RM1['Model Utama lengkap'],
       'IF Personal (skor a saja, batas sendiri)': None,
       'tanpa skor konteks c': ablate(lambda z: risk(z, z['a'], flags(z), (0.5, 0.0, 0.1))),
       'tanpa riwayat h': ablate(lambda z: risk(z, z['a'], flags(z), (0.5, 0.4, 0.0))),
       'tanpa kalibrasi per pengguna (a mentah)': ablate(lambda z: risk(z, z['a_raw'], flags(z))),
       'jumlah mentah, bukan relatif': ablate(lambda z: risk(z, z['a_jumlah_mentah'], flags(z)))}
ifp = E2['IF Personal']
RM2['IF Personal (skor a saja, batas sendiri)'] = {'swap_TP': ifp['swap']['TP'], 'swap_TP_adj': ifp['swap']['TP_adj'], 'swap_TPR': ifp['swap']['TPR'],
                                                   'swap_FPR': ifp['swap']['FPR'], 'swap_TPR_adj': ifp['swap']['TPR_adj'], 'swap_FPR_adj': ifp['swap']['FPR_adj'],
                                                   'fp_per_day': ifp['swap']['fp_per_day'], 'hijack_TP': ifp['hijack']['TP'], 'hijack_TP_adj': ifp['hijack']['TP_adj']}
OUT['E3_RM1'], OUT['E4_RM2'] = RM1, RM2
for k, v in list(RM1.items()) + list(RM2.items()):
    print('ABL', f'{k:45s}', v, flush=True)

# ---------------------------------------------------------------- E5 threshold curves (seed 1)
rows = []
for model in ['IF Personal', 'Model Utama']:
    for t in TH:
        sw = ev(SW1, S1['swap'][model], t)[0]
        hj = [ev(HJ1[lv], S1[lv][model], t)[0] for lv in LEVELS]
        rows.append([model, t, sw['TPR'], sw['FPR'], sw['TPR_adj'], sw['FPR_adj'], sw['fp_per_day'],
                     sum(x['TP'] for x in hj) / 60, sum(x['TP_adj'] for x in hj) / 60, np.mean([x['fp_per_day'] for x in hj])])
for t in L_TH:
    sw = ev_landauer(SW1, t)[0]
    hj = [ev_landauer(HJ1[lv], t)[0] for lv in LEVELS]
    rows.append(['Pembanding Landauer', t, sw['TPR'], sw['FPR'], sw['TPR_adj'], sw['FPR_adj'], sw['fp_per_day'],
                 sum(x['TP'] for x in hj) / 60, sum(x['TP_adj'] for x in hj) / 60, np.mean([x['fp_per_day'] for x in hj])])
with open(os.path.join(config.RESULTS, 'final_curve.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['model', 'theta', 'swap_TPR', 'swap_FPR', 'swap_TPR_adj', 'swap_FPR_adj', 'swap_fp_per_day', 'hijack_TPR', 'hijack_TPR_adj', 'hijack_fp_per_day'])
    for r in rows:
        w.writerow([r[0], r[1]] + [round(float(x), 4) for x in r[2:]])
print('E5 curve rows', len(rows), flush=True)

# ---------------------------------------------------------------- E6 uid swap on seeds 0-9
E6 = {}
for s in range(10):
    z = SW1 if s == 1 else load(f'swap_s{s}')
    sc = scores(z)['Model Utama']
    E6[s] = {'Pembanding Landauer': r4(ev_landauer(z)[0]), 'Model Utama': r4(ev(z, sc, THETA['Model Utama'])[0])}
    print('E6 seed', s, 'Landauer', E6[s]['Pembanding Landauer']['TP'], E6[s]['Pembanding Landauer']['FPR'],
          '| Model Utama', E6[s]['Model Utama']['TP'], E6[s]['Model Utama']['FPR'], flush=True)
tot = {}
for grp, seeds in (('0-9', range(10)), ('3-9', range(3, 10)), ('1 dan 3-9', [1] + list(range(3, 10)))):
    t = {}
    for model in ('Pembanding Landauer', 'Model Utama'):
        t[model] = {'TP': sum(E6[s][model]['TP'] for s in seeds), 'TP_adj': sum(E6[s][model]['TP_adj'] for s in seeds), 'P': 20 * len(list(seeds)),
                    'FPR_rata2': round(float(np.mean([E6[s][model]['FPR'] for s in seeds])), 4),
                    'FPR_adj_rata2': round(float(np.mean([E6[s][model]['FPR_adj'] for s in seeds])), 4)}
    x = [E6[s]['Model Utama']['TP'] for s in seeds]
    y = [E6[s]['Pembanding Landauer']['TP'] for s in seeds]
    t['wilcoxon_p_satu_arah'] = round(float(wilcoxon(x, y, alternative='greater').pvalue), 4) if any(a != b for a, b in zip(x, y)) else 1.0
    tot[grp] = t
OUT['E6_10_putaran'] = {'per_putaran': E6, 'total': tot}
print('E6 total', tot, flush=True)
json.dump(OUT, open(os.path.join(config.RESULTS, 'final.json'), 'w'), indent=1, default=str)
print('DONE')
