# UEBA Nextcloud — reproduksi pengujian skripsi

Kode untuk mereproduksi semua pengujian pada CLUE-LDS (log Nextcloud, 5.389 user):
baseline Landauer et al. (2022) vs Isolation Forest + fitur konteks + risk scoring,
plus pengujian injeksi pembajakan sesi (K3) harian dan per time window.

Semua hasil yang dilaporkan di skripsi ada di `expected_results/`. Menjalankan
`run_all.sh` membuat ulang file yang sama di `results/`.

## Kebutuhan

- Linux (diuji di Ubuntu 24.04), Python 3.12.3, 2 CPU, RAM 8 GB, disk kosong ±15 GB
- Waktu total ±4–5 jam

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Versi paket di `requirements.txt` adalah versi persis yang dipakai untuk hasil di skripsi.
Isolation Forest dan generator acak bergantung pada versi numpy/scikit-learn, jadi versi
lain bisa memberi angka yang sedikit berbeda.

## Data

CLUE-LDS dari Zenodo (record 7119953), `clue.zip`, 635 MB, MD5 `9e318370f96b68077667e9cdc05f26a5`.
`run_all.sh` mengunduhnya sendiri ke `data/` jika belum ada. Lokasi bisa diganti lewat
variabel lingkungan `DATA_DIR`, `WORK_DIR`, `RESULTS_DIR` (lihat `src/config.py`).

## Menjalankan

```bash
PY=.venv/bin/python ./run_all.sh
```

Mulai dari langkah tertentu (misalnya setelah langkah 1–6 selesai): `PY=.venv/bin/python ./run_all.sh 7`.

| Langkah | Script | Isi | Output |
|---|---|---|---|
| 0 | (wget + md5sum) | unduh & cek dataset | `data/clue.zip` |
| 1 | `src/to_parquet.py` | zip → Parquet, kolom `rn` = urutan baris asli | `work/events.parquet` |
| 2 | `src/user_stats.py` | statistik per user untuk pemilihan pasangan swap | `work/user_stats.json` |
| 3 | `src/v1_v2.py`, `src/ip_geo_coverage.py` | validasi data V1–V2, cakupan IP/geolokasi | `results/v1_v2.json`, `results/ip_geo_coverage.json` |
| 4 | `src/base_daily.py` | tabel per user-hari untuk model | `work/base/` |
| 5 | `src/duck_seed.py`, `src/summarize_seeds.py` | baseline A (Landauer), seed 1 + seed 0, 2–9 | `work/A/` |
| 6 | `src/check_evaluator.py` | V4: evaluator kami pada keputusan A harus = TP 13 / FP 3337 | stdout (assert) |
| 7 | `src/run_all.py` | Pengujian 1 (uid swap): semua skenario `Skenario_Pengujian.xlsx` | `results/results.json`, `run_all.log` |
| 8 | `src/hybrid.py`, `src/explore_rel.py`, `src/bench_detectors.py` | eksplorasi post-hoc: hybrid, X4 (fitur relatif + kalibrasi), benchmark detektor | `results/hybrid.json`, `explore_rel.json`, `bench_detectors.json` |
| 9 | `src/k3_day.py` | Pengujian 2: injeksi K3 harian, intensitas 2×/5×/10× | `results/k3_day.json` |
| 10 | `src/k3_window.py` | Pengujian 3: K3 per window 15/30 menit | `results/k3_window.json` |
| 11 | `src/k3_window2.py` | Pengujian 3b: window yang diperbaiki (akumulasi, EWMA, supresi), 30/60 menit | `results/k3_window2.json` |

## Peta pengujian → hasil

| Pengujian di skripsi | Kunci di file hasil | Angka utama (seed 1) |
|---|---|---|
| Baseline A (Landauer, t=0,6) | `results.json` → `P1`, `E1_A` | TP 13/20, FP 3.337 |
| Skenario B, B+RS, C, D, P1–P5, H1–H3, K1–K2, E6 | `results.json` | lihat `run_all.log` |
| Validasi V5–V7 | `results.json` → `V5`, `V6`, `V7` | |
| Hybrid, X4, benchmark | `hybrid.json`, `explore_rel.json`, `bench_detectors.json` | X4: TPR 0,75 |
| K3 harian | `k3_day.json` | D 46/60 vs A 9/60 |
| K3 window 15 menit | `k3_window.json` | X4-win 35/60 vs A-win 1/60 |
| K3 window diperbaiki (X5) | `k3_window2.json` | 30 menit: 49/60 (seed 1) |

Rancangan tiap pengujian ditulis sebelum dijalankan: `docs/SPEC_model_v2.md` (Pengujian 1),
`docs/SPEC_K3_window.md` (Pengujian 3), `docs/SPEC_K3_window_v2.md` (Pengujian 3b).

## Catatan seed

- **Seed 1** = run bawaan Landauer; semua angka utama dilaporkan pada seed ini, untuk semua 5.389 user.
- **Seed 0 dan 2** dipakai untuk tuning (ambang, bobot). Tidak ada parameter yang di-tuning pada seed 1.
- **Seed 3–9** hanya dipakai di Pengujian 3b sebagai data yang belum pernah dilihat, untuk
  memastikan perbaikan window tidak hanya cocok dengan seed 1 (hasil: 309/420 pada window 30 menit).
- Seed menentukan pasangan user yang ditukar (uid swap) dan korban/waktu injeksi K3;
  dengan seed yang sama hasilnya identik setiap kali dijalankan.

## Folder

- `src/` — pipeline yang dipakai untuk semua hasil (DuckDB + Parquet).
- `legacy/` — pipeline awal berbasis chunk yang langsung membaca `clue.zip`, beserta
  `run_equiv.sh` yang membuktikan pipeline chunk = script asli Landauer pada 3 juta event pertama.
  Butuh script asli dari [ait-aecid/clue-lds](https://github.com/ait-aecid/clue-lds)
  (`make_detect_chunk.py` membuat `detect_chunk.py` dari `detect.py` Landauer). Tidak diperlukan
  untuk mereproduksi hasil; `src/duck_seed.py` memberi TP/FP yang sama.
- `src/v3.py` — validasi V3 (DuckDB = pipeline chunk); butuh output `legacy/run_seed.sh`.
- `tools/` — membuat tabel hasil (`results_table.py`) dan mengisi workbook skenario (`fill_workbook.py`).
- `expected_results/` — hasil asli dari VPS, untuk dibandingkan.

## Referensi

Landauer, M., Skopik, F., Höld, G., & Wurzenberger, M. (2022). A User and Entity Behavior
Analytics Log Data Set for Anomaly Detection in Cloud Computing. IEEE BigData 2022.
