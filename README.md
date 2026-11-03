# UEBA on Nextcloud logs — reproduction package

Code for the thesis and paper *Evaluasi Deteksi Anomali Aktivitas Pengguna pada Private Cloud Storage
(Nextcloud) Menggunakan Pendekatan User and Entity Behavior Analytics (UEBA)*, BINUS Online, 2026.

**Question.** Can an unsupervised detector that judges every user only against that user's own history find
misuse of a legitimate Nextcloud account? It is compared with the method of Landauer et al. (2022).

**Data.** CLUE-LDS, the public Nextcloud log of Landauer et al.: 50.5 M events, 5,389 users, 1,910 days.
No labels, no IP addresses and no geolocation are used as model inputs.

**Attacks.**
- **Account swap** (*pertukaran akun*): Landauer's own scenario. From a chosen day, one user's activity is
  recorded under another user's ID. There are 20 swaps per seed.
- **Session hijack** (*pembajakan sesi*): generated for this study. A 1–3 hour session is injected into a
  victim's day, with logins, file access at 2×, 5× or 10× the victim's normal rate, shares and deletes, and
  folders the victim never used. There are 20 victims × 3 intensities per seed.

**Running the models.** Decisions are made per user per day, in time order, with the evaluation protocol of
Landauer (adaptation days after an alarm are not scored).

## The three models

| Name in the thesis and paper | What it does | Alarm when |
|---|---|---|
| **Pembanding Landauer** | Landauer et al. (2022): the distance between today's 49 event-type counts and the nearest of the user's last 30 days | distance > 0.6 |
| **IF Personal** | One Isolation Forest on user-relative inputs: 49 counts as `log1p(today) − log1p(median of the user's previous 14 active days)` plus 8 context features F1–F8. The score is ranked against the user's own previous 30 days and gives `a` | `a ≥ θ` |
| **Model Utama** (main model) | IF Personal plus a risk score `R = 0.5·a + 0.4·c + 0.1·h`. `c` is the weight of raised warning flags / 10. `h = 0.3·h_prev + 0.7·R_prev` is the user's recent suspicion | `R ≥ θ` |

The context features compare today with the user's own previous 14 active days:

| | Feature | | Feature |
|---|---|---|---|
| F1 | activity volume (robust z) | F5 | length of the active period (robust z) |
| F2 | change in the mix of actions | F6 | share of activity in folders never used before |
| F3 | share of never-used action types | F7 | extra share of risky actions (share, delete, …) |
| F4 | share of activity outside usual hours | F8 | busiest 10 minutes (robust z) |

The warning flags of Model Utama are listed below (weight in brackets):

| Flag | Raised when | Flag | Raised when |
|---|---|---|---|
| k1 (2) | \|F1\| ≥ 3 | k4 (2) | F6 ≥ 0.5 |
| k2 (2) | F2 ≥ 0.5 or F3 ≥ 0.2 | k5 (1) | F7 ≥ 0.1 or F8 ≥ 3 |
| k3 (1) | F4 ≥ 0.3 | k6 (2) | order of activities changed: `trans_js` ≥ 0.5 |

`trans_js` is the Jensen–Shannon divergence between today's consecutive action pairs (a → b) and those of the
user's previous 14 active days.

**Choosing the threshold θ.** θ is chosen only on validation seeds 0 and 2, using both attacks. False alarms
must be at most 1.75 per day, which is Landauer's own rate. Within that limit, θ gives the highest mean
detection rate of the two attacks. Seed 1 (Landauer's default) is the main test. Seeds 3–9 check robustness on
the account swap.

**Metrics.** Every table reports TPR and FPR on scored days, as in Landauer. It also reports TPR_adj and
FPR_adj, which include adaptation days.

## Main results (seed 1)

| Model | θ | Account swap: detected / 20 | FPR | Session hijack: detected / 60 |
|---|---|---|---|---|
| Pembanding Landauer | 0.60 | 13 | 4.61% | 9 |
| IF Personal | 0.99 | 7 | 3.20% | 25 |
| **Model Utama** | 0.67 | **14** | **4.56%** | **24** |

On seeds 0–9, Model Utama detects 118 of 200 account swaps; Landauer detects 89. On test seeds 3–9 the
counts are 78 vs 54 (Wilcoxon one-sided p = 0.016). The ablations (RM1, RM2), threshold curves, TPR_adj,
FPR_adj and statistical tests are in [`expected_results/final.json`](expected_results/final.json).

## Requirements

- Linux (tested on Ubuntu 24.04), Python 3.12.3, 2 CPUs, 8 GB RAM, about 15 GB free disk
- Total run time: about 2 hours

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used. Isolation Forest and the random generators depend on the
numpy and scikit-learn versions, so other versions may give slightly different numbers.

## Running

```bash
PY=.venv/bin/python ./run_all.sh
```

This downloads CLUE-LDS from Zenodo (record 7119953, `clue.zip`, 635 MB, MD5
`9e318370f96b68077667e9cdc05f26a5`) to `data/` if it is missing. Results are written to `results/` and can
be compared with `expected_results/`. To start from a given step, pass it as an argument, for example
`./run_all.sh 12`. Locations can be changed with `DATA_DIR`, `WORK_DIR` and `RESULTS_DIR` (see
`src/config.py`).

| Step | Script | What it does | Output |
|---|---|---|---|
| 0 | (wget + md5sum) | download and verify the dataset | `data/clue.zip` |
| 1 | `src/step01_data_to_parquet.py` | zip → Parquet, keeping the original line order | `work/events.parquet` |
| 2 | `src/step02_user_stats.py` | per-user statistics for choosing the swapped pairs | `work/user_stats.json` |
| 3 | `src/step03_check_v1_v2_data.py`, `src/step03_check_ip_geo_coverage.py` | checks V1–V2: field coverage, aggregation, IP and geolocation coverage | `results/v1_v2.json`, `results/ip_geo_coverage.json` |
| 4 | `src/step04_daily_tables.py` | per user-day tables | `work/base/` |
| 5 | `src/step05_model_A_landauer.py`, `src/step05_summarize_A_seeds.py` | Pembanding Landauer with the original algorithm, seeds 0–9 | `work/A/` |
| 6 | `src/step06_check_evaluator_vs_A.py` | check V4: our evaluator on Landauer's decisions gives TP 13 / FP 3,337, as in Landauer's paper | stdout, ends with `EVALUATOR CHECK PASS` |
| 12 | `src/step12_final.sh` (`step12_final_prep.py`, `step12_final_eval.py`) | **all numbers of the thesis and the paper**: the three models, thresholds, seed 1 for both attacks, RM1 and RM2 ablations, threshold curves, seeds 0–9, statistical tests | `results/final.json`, `results/final_curve.csv` |
| 13 | `src/step13_check_V5_V7.py` | checks V5–V7 on the inputs of IF Personal and Model Utama | `results/checks_V5_V7.json` |

Steps 7–11 were used for earlier model versions and have been removed. The numbering is kept because the
thesis and paper refer to `step12_final`.

Where each table and figure comes from (`results/final.json`):

| Thesis / paper | Key |
|---|---|
| Tabel 4.2 thresholds | `E1_thresholds` |
| Tabel 4.3 RM1 ablation, Gambar 4.1 | `E3_RM1` |
| Tabel 4.4 RM2 ablation | `E4_RM2` |
| Tabel 4.5–4.6 seed 1, both attacks | `E2_seed1` |
| Tabel 4.7 McNemar and bootstrap | `E2_stats` |
| Gambar 4.2–4.3 threshold and operating curves | `final_curve.csv` |
| Tabel 4.8, Gambar 4.4 seeds 0–9 | `E6_10_putaran` |

## Data and process checks (V1–V7)

| ID | Check | Result | Script |
|---|---|---|---|
| V1 | Field coverage | type and time 100%; path 45.2%; IP 0.55%; geolocation 0.24% | step 3 |
| V2 | Aggregation | events in the user-day tables = 50,522,931; difference 0 | step 3 |
| V3 | Account-swap equivalence | swap on summary tables = swap per event; 0 of 83,147 user-days differ | `src/check_v3_swap_equivalence.py` (optional, needs `legacy/`) |
| V4 | Landauer reproduced | TP 13, FP 3,337, TN 69,112, FN 7, adaptation days 5,289, identical to Landauer's paper | steps 5–6 |
| V5 | No future data | changing a user's data after day t changes none of the inputs on days ≤ t | step 13 |
| V6 | No label leakage | 57 Isolation Forest inputs (49 relative counts + F1–F8); no role, login name, IP, location or label | step 13 |
| V7 | Repeatable | computing the inputs and scores twice gives identical values, equal to step 12 | step 13 |

## Code

| File | Contents |
|---|---|
| `src/lib_models.py` | base tables, account swap, features F1–F8, relative counts, Isolation Forest, per-user ranking, risk score, Landauer detector, evaluator |
| `src/lib_order_features.py` | order-of-activities feature `trans_js` (flag k6) |
| `src/lib_k3_attack.py` | session-hijack generator |
| `src/lib_uid_swap.py` | Landauer's account-swap selection (pairs and switch times) |
| `src/config.py` | data, work and result paths |

- `legacy/` holds the first pipeline, which reads `clue.zip` directly. Its `run_equiv.sh` shows that this
  pipeline equals Landauer's original scripts from [ait-aecid/clue-lds](https://github.com/ait-aecid/clue-lds).
  It is only needed for check V3.
- `expected_results/` holds the results of the thesis run, for comparison.

## Reference

Landauer, M., Skopik, F., Höld, G., & Wurzenberger, M. (2022). A User and Entity Behavior Analytics Log Data
Set for Anomaly Detection in Cloud Computing. *2022 IEEE International Conference on Big Data*, 4285–4294.
