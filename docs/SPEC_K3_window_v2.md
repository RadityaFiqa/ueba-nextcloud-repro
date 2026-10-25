# K3-b v2 — perbaikan deteksi per window (X5-win)

Ditetapkan 2026-10-24, SEBELUM dijalankan. Diminta pengguna: gabungan (1) akumulasi bukti antar
window, (2) baseline EWMA + profil jam, (3) penekanan alarm berulang per user; window 30 dan 60 menit;
pembanding X4-win (versi K3-b) dan A-win.

## Data dan serangan
Sama dengan `SPEC_K3_window_DRAFT.md` (174 user, generator K3, tiga intensitas, 20 korban per seed).
- Tuning: seed 0 dan 2 (120 sesi).
- **Uji utama: seed 3–9 (420 sesi)** — seed yang belum pernah dilihat. Seed 1 dilaporkan terpisah
  sebagai pembanding dengan K3-b sebelumnya (sudah dilihat, tidak dipakai untuk klaim).
- Anggaran: ≤ 1,75 false alarm per hari (rata-rata seed validasi), aturan pilih θ sama:
  sesi terdeteksi terbanyak, lalu median latensi terendah.

## Model
| Model | Isi |
|---|---|
| A-win | Landauer per window (-u -f), tidak berubah |
| X4-win | Versi K3-b, tidak berubah |
| X5-base | X4-win dengan baseline baru (2) |
| X5-acc | X5-base + akumulasi bukti (1) |
| **X5-full** | X5-base + akumulasi (1) + penekanan (3) |

**(2) Baseline EWMA + profil jam.**
- Set A relatif: `log1p(x_k) − log1p(m_k)`, dengan m_k = EWMA hitungan tipe k per window aktif user
  (α = 0,1; dimulai dari window pertama). Menggantikan median 30 window (yang sering 0).
- c1 volume: `log1p(total) − log1p(m_jam)`, dengan m_jam = EWMA total per jam-hari (24 slot, α = 0,2;
  slot belum terisi → EWMA total semua jam). Flag: c1 ≥ log(4) (≥ ~4× biasanya untuk jam itu).
- c2 folder baru, c3 aksi sensitif, c4 di luar jam: sama dengan X4-win.
- IF global, kalibrasi per user, R = 0,5a + 0,4c + 0,1h: sama dengan X4-win.

**(1) Akumulasi bukti.** Skor alarm S_t = jumlah R pada window kalender t, t−1, …, t−K+1 milik user
(window tanpa aktivitas = 0). K = 3 untuk 30 menit (1,5 jam), K = 2 untuk 60 menit (2 jam).

**(3) Penekanan.** Setelah alarm untuk user u, alarm berikutnya untuk u ditahan selama 4 jam
(8 window 30 menit / 4 window 60 menit). Hanya alarm yang dikeluarkan dihitung (false alarm dan deteksi).

## Metrik
Per seed uji dan gabungan seed 3–9: sesi terdeteksi, median dan p90 latensi (menit dari mulai sesi
sampai akhir window alarm), false alarm per hari. Sesi terdeteksi = ada alarm yang dikeluarkan di
window sesi, atau (untuk model dengan akumulasi) di K−1 window sesudahnya; alarm di window itu tidak
dihitung sebagai false alarm. Window tanpa skor R (belum ada model) tidak bisa memicu alarm. Uji: sign test berpasangan per sesi (X5-full vs X4-win,
X5-full vs A-win) pada 420 sesi.

## Kriteria (ditetapkan sekarang)
X5-full lebih baik dari X4-win bila, pada anggaran yang sama, sesi terdeteksi lebih banyak (sign test
p < 0,05) dan median latensi tidak lebih buruk dari X4-win + 1 window.
