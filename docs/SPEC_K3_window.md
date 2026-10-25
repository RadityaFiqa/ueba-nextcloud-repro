# K3 + deteksi per window 15/30 menit — rancangan (DRAFT, perlu persetujuan)

Status: **draft**. Semua parameter di bawah dikunci SEBELUM ada hasil. Dasar angka: statistik
perilaku normal 174 user yang memenuhi syarat korban (belum ada hasil model apa pun).

## 1. Pertanyaan
- **K3-a (harian):** pada pembajakan dengan aksi jahat, apakah D / X4 / hybrid mendeteksi lebih
  banyak dari baseline A, dengan threshold yang sudah dikunci (tanpa tuning ulang)?
- **K3-b (per window):** dengan window 15 dan 30 menit, pada anggaran alert yang sama dengan A
  (≤ 1,75 alert/hari), model mana yang mendeteksi lebih banyak sesi dan **lebih cepat**?

## 2. Model serangan (M7 – sesi pembajakan)
Statistik normal (174 user): median 22 event/hari; median 4 event per window 30 menit
(p90 per user: 8); 14% user-hari berisi share/hapus; jam UTC hampir rata (3–6% per jam).

| Parameter | Nilai (usulan) |
|---|---|
| Korban per seed | 20, dipilih acak dari 174 user (syarat Landauer: > 27 hari, > 100 event, > 4 tipe) |
| Data dasar | Data bersih (tanpa uid swap), supaya efek K3 terpisah dari uid swap |
| Hari serangan | Acak di antara hari aktif ke-29 sampai ke-(n−1) korban (punya histori, bisa dinilai) |
| Mulai sesi | Acak 00:00–21:00 UTC pada hari itu |
| Durasi | Acak 1–3 jam (sesuai workbook) |
| Login | 1 `login_attempt` + 1 `login_successful` di awal sesi |
| Akses massal | `file_accessed`, laju = k × median event per 30 menit milik korban (minimal 4 per 30 menit) |
| Exfiltrasi | `shared_link` sebanyak s |
| Perusakan | `file_deleted` sebanyak d |
| Folder | 50% dari folder yang pernah dipakai korban, 50% dari folder user lain yang belum pernah disentuh korban |
| Waktu tiap event | Acak merata dalam sesi, bercampur dengan aktivitas korban sendiri |

**Tiga intensitas** (relatif terhadap korban, supaya user besar tidak otomatis "tersamar"):

| Intensitas | k (× laju normal) | shared_link s | file_deleted d |
|---|---|---|---|
| Rendah | 2 | 1 | 0 |
| Sedang | 5 | 3 | 0 |
| Tinggi | 10 | 5 | 5 |

Seed injeksi K3 terpisah dari seed uid swap (`random.Random(1000 + seed)`). Satu run = satu seed
× satu intensitas, 20 serangan. Tuning hanya di seed 0 dan 2; dilaporkan di seed 1.

## 3. K3-a: evaluasi harian (protokol Landauer)
- Model A, D, X4, hybrid **tanpa tuning ulang** (θ dan bobot seperti hasil utama/eksploratif).
- Positif = user-hari serangan. Hari lain = negatif (FPR praktis sama dengan run tanpa serangan).
- Metrik: TPR per intensitas, TPR gabungan (60 serangan), FPR, alert/hari; McNemar pada 60 positif
  (model vs A).

## 4. K3-b: evaluasi per window (15 dan 30 menit)
- Cakupan: 174 user yang memenuhi syarat (≈ 99% event); user lain tidak dinilai di level window.
- **A-win:** detektor Landauer pada count vector per window (antrean 30 window terakhir user).
- **X4-win:** Set A relatif per window (vs median 30 window aktif terakhir user) + 4 fitur konteks
  window (volume z, porsi folder baru, kelebihan aksi sensitif, di luar jam biasa user);
  IF global (jendela 60 hari, retrain 7 hari); kalibrasi per user (30 window terakhir);
  R = 0,5a + 0,4c + 0,1h.
- Threshold tiap model dipilih di seed 0+2 sehingga **false alarm ≤ 1,75 per hari** (= A harian),
  lalu dibandingkan pada anggaran alert yang sama.
- Metrik: **sesi terdeteksi** (≥ 1 alert di window yang beririsan dengan sesi), **latensi**
  (menit dari mulai sesi sampai akhir window pertama yang alert; median dan p90), false alarm/hari.
- Tanpa mode retraining di level window (setiap window dinilai).

## 5. Kriteria (ditetapkan sekarang)
- K3-a: model usulan lebih baik dari A jika TPR gabungan lebih tinggi dan McNemar p < 0,05.
- K3-b: X4-win lebih baik dari A-win jika, pada anggaran ≤ 1,75 alert/hari, sesi terdeteksi lebih
  banyak DAN median latensi lebih rendah.
- Hasil apa pun dilaporkan; parameter tidak diubah setelah melihat hasil.

## 6. Keputusan terbuka
1. Kebijakan folder penyerang (50/50) — lebih realistis, tetapi menguntungkan fitur F6.
2. Tiga intensitas di atas.
3. K3-b hanya pada 174 user yang memenuhi syarat.
4. Pembanding pada anggaran alert yang sama (≤ 1,75/hari).
5. Jalankan 15 dan 30 menit sekaligus.
