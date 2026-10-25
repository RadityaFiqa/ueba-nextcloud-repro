"""Fill Status / Hasil / Catatan (columns J, K, L, the yellow input cells) of
Skenario_Pengujian.xlsx from the results. Nothing else in the workbook is touched.

Usage: python fill_workbook.py SRC.xlsx DST.xlsx results.json v1_v2.json v3.json
"""
import json
import sys

import openpyxl

src, dst, res_p, v12_p, v3_p = sys.argv[1:6]
R = json.load(open(res_p, encoding='utf-8'))
V12 = json.load(open(v12_p, encoding='utf-8'))
V3 = json.load(open(v3_p, encoding='utf-8'))
var = R['variants']
A = R['E1_A']['test_seed1']
D = var['E5_D']['test_seed1']
C = var['E4_C']['test_seed1']


def p(x, n=3):
    return f'{x:.{n}f}'.replace('.', ',')


def line(m, extra=''):
    s = (f"TP {m['TP']}/20, TPR {p(m['TPR'], 2)}, TPR_adj {p(m['TPR_adj'], 2)}, FPR {p(m['FPR'], 4)}, "
         f"R {p(m['R'], 3)}, alert/hari {p(m['alerts_per_day'], 2)}")
    return s + extra


def vline(name, ref=None, ref_name=''):
    v = var[name]
    m = v['test_seed1']
    th = v['theta']
    th = f"(a≥{p(th[0], 2)}, c≥{p(th[1], 3)})" if isinstance(th, list) else p(th, 2)
    d = f"; ΔFPR vs {ref_name} {'+' if m['FPR'] - ref['FPR'] >= 0 else ''}{p(m['FPR'] - ref['FPR'], 4)}" if ref else ''
    return f"θ={th}: " + line(m) + d


SEED = 'Seed 1 (run tunggal); parameter/threshold dipilih di seed 0+2'
P5 = R['P5']
p5f = [m['FPR'] for k, m in P5.items() if k != 'base']
p5t = [m['TPR'] for k, m in P5.items() if k != 'base']
e6 = R['E6']
k = R['K1_K2']
h2 = R['H2']

rows = {
    'V1': ('Selesai',
           f"type & time 100%; path {p(V12['V1']['all']['pct_events']['path'], 1)}% event / "
           f"{p(V12['V1']['all']['pct_user_days']['path'], 1)}% user-hari; isLocalIP {p(V12['V1']['all']['pct_events']['isLocalIP'], 2)}% event; "
           f"location {p(V12['V1']['all']['pct_events']['location'], 2)}% event",
           'User login: IP/lokasi hanya di login_failed (0,26% event) → fitur tanpa IP/lokasi'),
    'V2': ('Selesai', f"Σ event tabel user-hari = {V12['V2']['summary_event_sum']:,}; selisih {V12['V2']['difference']}".replace(',', '.'),
           'Lulus'),
    'V3': ('Selesai', f"{V3['differing_user_days']} user-hari berbeda dari {V3['user_days_ref']:,} (seed 1)".replace(',', '.'),
           'Lulus; pembanding = swap per event (port generate_test_file.py)'),
    'V4': ('Selesai', 'TP 13, FP 3.337, TN 69.112, FN 7, Train 5.289', 'Identik dengan README Landauer; evaluator baru juga 0 selisih per hari'),
    'V5': ('Selesai' if R['V5']['pass'] else 'Gagal',
           f"{R['V5']['feature_values_changed_on_or_before_t']} nilai fitur berubah di hari ≤ t (100 user)",
           f"Kontrol: {R['V5']['feature_values_changed_after_t (sanity, should be >0)']:,} nilai berubah setelah t".replace(',', '.')),
    'V6': ('Selesai' if R['V6']['pass'] else 'Gagal', f"{R['V6']['n_columns_C_D']} kolom (49 tipe event + F1–F8); kolom terlarang: tidak ada",
           'role, loginName, IP, lokasi, label tidak masuk input'),
    'V7': ('Selesai' if R['V7']['identical'] else 'Gagal', 'Selisih metrik = 0' if R['V7']['identical'] else 'Ada selisih',
           'D dua kali di seed 3, random_state = 0'),
    'E1': ('Selesai', 'θ=0,6: ' + line(A), f"Sweep θ 0,30–0,90 tersimpan di results.json. {SEED}"),
    'E2': ('Selesai', vline('E2_B') + f"; PR-AUC {p(var['E2_B']['test_seed1']['PR_AUC'], 4)}", 'Jauh lebih buruk dari A'),
    'E3': ('Selesai', vline('E3_B+RS') + f"; PR-AUC {p(var['E3_B+RS']['test_seed1']['PR_AUC'], 4)}", 'Risk scoring menurunkan FPR B dari 0,227 ke 0,108'),
    'E4': ('Selesai', vline('E4_C') + f"; PR-AUC {p(var['E4_C']['test_seed1']['PR_AUC'], 4)}",
           'RQ1: FPR C < FPR B, tetapi TPR 0,55 < 0,65 → syarat RQ1 tidak terpenuhi penuh'),
    'E5': ('Selesai', vline('E5_D') + f"; PR-AUC {p(var['E5_D']['test_seed1']['PR_AUC'], 4)}",
           'H1 TIDAK didukung: FPR D > A dan TPR < 0,65. Pada FPR setara (θ=0,65; FPR 0,045) TP D 10 vs A 13'),
    'E6': ('Selesai',
           f"McNemar (exact): D vs A p<1e-300, ΔFPR +{p(e6['E5_D vs A']['dFPR'], 3)} [{p(e6['E5_D vs A']['ci95'][0], 3)}; {p(e6['E5_D vs A']['ci95'][1], 3)}]; "
           f"D vs C p={e6['E5_D vs E4_C']['p_value_exact']:.1e}, ΔFPR {p(e6['E5_D vs E4_C']['dFPR'], 3)}; "
           f"C vs B ΔFPR {p(e6['E4_C vs E2_B']['dFPR'], 3)}",
           'Wilcoxon diganti McNemar per user-hari + bootstrap CI 95% (run tunggal). D signifikan LEBIH BURUK dari A'),
    'R1': ('Selesai', vline('R1_D-h', D, 'D'), 'Tanpa h: FPR naik, TPR_adj 0,80'),
    'R2': ('Selesai', vline('R2_D-c', D, 'D'), 'Tanpa konteks: FPR naik mendekati C (sesuai dugaan)'),
    'R3': ('Selesai', vline('R3_D-a', D, 'D'), 'Aturan saja: FPR terendah (0,027) tetapi TPR hanya 0,35'),
    'R4': ('Selesai', vline('R4_D-gate', D, 'D'), 'Varian terbaik: TPR 0,65 dengan FPR 0,079 (masih > A)'),
    'F1': ('Selesai', vline('F1_C-Vol', C, 'C'), 'Volume berkontribusi'),
    'F2': ('Selesai', vline('F2_C-Mix', C, 'C'), 'Campuran aktivitas berkontribusi'),
    'F3': ('Selesai', vline('F3_C-Time', C, 'C'), 'Kontribusi terkecil'),
    'F4': ('Selesai', vline('F4_C-Res', C, 'C'), 'Folder berkontribusi, juga untuk TPR'),
    'F5': ('Selesai', vline('F5_C-Sens', C, 'C'), 'Aksi sensitif berkontribusi'),
    'F6': ('Selesai', vline('F6_C-Burst', C, 'C'), 'Kontribusi terbesar'),
    'F7': ('Selesai', '; '.join(f"−{x}: FPR {p(var[f'F7_D-{x}']['test_seed1']['FPR'], 4)}, TPR {p(var[f'F7_D-{x}']['test_seed1']['TPR'], 2)}"
                                for x in ('k1', 'k2', 'k3', 'k4', 'k5')),
           'Membuang k1 menurunkan FPR (0,099→0,078): k1 merugikan (z-score jenuh ±10 saat MAD≈0)'),
    'D1': ('Selesai', vline('D1_D-static', D, 'D'), 'Model statis lebih buruk → retrain tetap perlu'),
    'D2': ('Selesai', vline('D2_D-daily', D, 'D'), 'FPR setara D; waktu ~7× (≈495 dtk vs 70 dtk per seed) → retrain 7 hari cukup'),
    'D3': ('Selesai', vline('D3_D-noUserMode', D, 'D'), 'FPR sedikit turun, R≈0, TPR_adj turun ke 0,60'),
    'P1': ('Selesai', '; '.join(f"W={w}: FPR {p(m['FPR'], 4)}, TPR {p(m['TPR'], 3)}" for w, m in R['P1']['candidates'].items()),
           f"Dikunci W = {R['P1']['locked_W']} (seed 0+2)"),
    'P2': ('Selesai', '; '.join(f"{w}: FPR {p(m['FPR'], 4)}" for w, m in R['P2']['candidates'].items()),
           f"Dikunci bobot {tuple(R['P2']['locked_weights'])} (seed 0+2)"),
    'P3': ('Selesai', '; '.join(f"λ={l}: FPR {p(m['FPR'], 4)}" for l, m in R['P3']['candidates'].items()),
           f"Dikunci λ = {R['P3']['locked_lambda']} (seed 0+2)"),
    'P4': ('Selesai', f"θ_R = {p(R['P4']['locked_theta_R'], 2)} (validasi: TPR 0,65, FPR {p(R['P4']['validation_curve'][str(R['P4']['locked_theta_R'])]['FPR'], 4)})",
           'Kurva uji seed 1 di results.json: θ 0,65 → FPR 0,045, TPR 0,50'),
    'P5': ('Selesai', f"FPR {p(min(p5f), 4)}–{p(max(p5f), 4)}, TPR {p(min(p5t), 3)}–{p(max(p5t), 3)} (14 variasi ±20%)",
           'Tidak berubah drastis'),
    'K1': ('Selesai', f"n={k['K1_easy(sim<0.3)']['n']}: TPR A {p(k['K1_easy(sim<0.3)']['A_TPR'], 2)}, TPR D {p(k['K1_easy(sim<0.3)']['D_TPR'], 2)}",
           'Dari pasangan seed 1 (tanpa injeksi tambahan); FPR = FPR keseluruhan'),
    'K2': ('Selesai', f"n={k['K2_hard(sim>=0.3)']['n']}: TPR A {p(k['K2_hard(sim>=0.3)']['A_TPR'], 2)}, TPR D {p(k['K2_hard(sim>=0.3)']['D_TPR'], 2)}",
           'D tidak unggul dari A pada kasus sulit'),
    'K3': ('Belum', '', 'Tidak dijalankan (keputusan peneliti, 22-10-2026)'),
    'H1': ('Selesai', 'Tabel 20 korban (similarity, rasio volume, deteksi A/D) di results.json → H1',
           f"A mendeteksi {A['TP']}, D {D['TP']} (TP_adj D {D['TP_adj']})"),
    'H2': ('Selesai', 'Sampel 50 FP: ' + ', '.join(f"{c} ({n})" for c, n in h2['sample50_causes'][:5]),
           f"Semua FP memicu ≥1 flag; total FP D = {h2['n_false_alarms_D']:,}".replace(',', '.') + ' (seed 1, bukan seed 3)'),
    'H3': ('Selesai', f"Alert A {A['alerts']:,}".replace(',', '.') + f" ({p(A['alerts_per_day'], 2)}/hari, prec {p(A['Prec'], 4)}); "
                      + f"D {D['alerts']:,}".replace(',', '.') + f" ({p(D['alerts_per_day'], 2)}/hari, prec {p(D['Prec'], 4)})",
           'Alert D ~1,9× A dengan TP lebih sedikit'),
}

wb = openpyxl.load_workbook(src)
ws = wb['Skenario Uji']
done = set()
for r in range(4, 42):
    sid = ws.cell(r, 1).value
    if sid in rows:
        st, hasil, cat = rows[sid]
        ws.cell(r, 10).value = st
        ws.cell(r, 11).value = hasil
        ws.cell(r, 12).value = cat
        done.add(sid)
missing = set(rows) - done
assert not missing, missing
wb.save(dst)
print('filled', len(done), 'rows ->', dst)
