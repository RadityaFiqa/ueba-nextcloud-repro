# UEBA on Nextcloud logs — reproduction package

Code to reproduce every experiment of the thesis on CLUE-LDS (Nextcloud logs,
5,389 users, 50.5 M events). It compares the baseline of Landauer et al. (2022)
with Isolation Forest + contextual features + risk scoring. Two kinds of attack are
tested: the dataset's own user-ID swap, and an injected session hijack (K3). The
hijack is evaluated per day and per 15/30/60-minute window.

All results reported in the thesis are in `expected_results/`. Running `run_all.sh`
recreates the same files in `results/`.

## Requirements

- Linux (tested on Ubuntu 24.04), Python 3.12.3, 2 CPUs, 8 GB RAM, ~15 GB free disk
- Total run time: about 4–5 hours

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the exact package versions used for the thesis results. Isolation
Forest and the random generators depend on the numpy/scikit-learn versions, so other
versions may give slightly different numbers.

## Data

CLUE-LDS from Zenodo (record 7119953): `clue.zip`, 635 MB, MD5 `9e318370f96b68077667e9cdc05f26a5`.
`run_all.sh` downloads it to `data/` if it is missing. Locations can be changed with the
environment variables `DATA_DIR`, `WORK_DIR` and `RESULTS_DIR` (see `src/config.py`).

## Running

```bash
PY=.venv/bin/python ./run_all.sh
```

To start from a given step (e.g. after steps 1–6 are done): `PY=.venv/bin/python ./run_all.sh 7`.

| Step | Script | What it does | Output |
|---|---|---|---|
| 0 | (wget + md5sum) | download and verify the dataset | `data/clue.zip` |
| 1 | `src/to_parquet.py` | zip → Parquet; column `rn` keeps the original line order | `work/events.parquet` |
| 2 | `src/user_stats.py` | per-user statistics used to pick the swap pairs | `work/user_stats.json` |
| 3 | `src/v1_v2.py`, `src/ip_geo_coverage.py` | data checks V1–V2, IP/geolocation coverage | `results/v1_v2.json`, `results/ip_geo_coverage.json` |
| 4 | `src/base_daily.py` | per user-day tables for the models | `work/base/` |
| 5 | `src/duck_seed.py`, `src/summarize_seeds.py` | baseline A (Landauer), seed 1 plus seeds 0, 2–9 | `work/A/` |
| 6 | `src/check_evaluator.py` | V4: our evaluator on A's decisions must give TP 13 / FP 3,337 | stdout (assert) |
| 7 | `src/run_all.py` | Test 1 (uid swap): all scenarios of `Skenario_Pengujian.xlsx` | `results/results.json`, `run_all.log` |
| 8 | `src/hybrid.py`, `src/explore_rel.py`, `src/bench_detectors.py` | post-hoc exploration: hybrid, X4 (relative features + calibration), detector benchmark | `results/hybrid.json`, `explore_rel.json`, `bench_detectors.json` |
| 9 | `src/k3_day.py` | Test 2: K3 session hijack, per day, intensities 2×/5×/10× | `results/k3_day.json` |
| 10 | `src/k3_window.py` | Test 3: K3 per 15/30-minute window | `results/k3_window.json` |
| 11 | `src/k3_window2.py` | Test 3b: improved window detector (accumulation, EWMA, suppression), 30/60 minutes | `results/k3_window2.json` |

Each test was specified before it was run: `docs/SPEC_model_v2.md` (Test 1),
`docs/SPEC_K3_window.md` (Test 3), `docs/SPEC_K3_window_v2.md` (Test 3b).

## Models

| Name | What it is |
|---|---|
| **A** | Baseline of Landauer et al. (2022): one model per user, L1 distance of daily event counts to the user's history, threshold 0.6 |
| **B** | One global Isolation Forest on basic daily features (event counts per type) |
| **B+RS** | B + risk scoring |
| **C** | Isolation Forest on basic + contextual features F1–F8 |
| **D** | Proposed model: C + risk score `R = 0.5·a + 0.4·c + 0.1·h` (a = IF score, c = context flags k1–k5, h = history `0.3·h_prev + 0.7·R_prev`) |
| **Hybrid** | D with the IF score *a* replaced by Landauer's per-user distance (post-hoc) |
| **X4** | D with features relative to the user's own last 14 days and a per-user score calibration over the last 30 days (post-hoc) |
| **A-win / X4-win** | A and X4 applied to 15/30/60-minute windows instead of days |
| **X5** | X4-win + accumulation over K windows + EWMA and hour-of-day baseline + 4-hour alert suppression |

## Seeds and evaluation

- **Seed 1** is Landauer's default run. All main numbers are reported on it, for all 5,389 users.
- **Seeds 0 and 2** are used for tuning (thresholds, weights). Nothing was tuned on seed 1.
- `k3_window2.json` also holds Test 3b results on seeds 3–9; they are not used in this report.
- A seed fixes which users are swapped and which victims/times get the K3 hijack. The same seed
  always gives the same result.
- **TPR** = share of attacks detected. **FPR** = share of normal user-days flagged.
  **Alerts/day** = all alerts divided by the number of days in the test period.
  **R** = share of user-days spent in retraining mode. **TPR_adj** = TPR counting a detection
  even if it falls on a retraining day.
- **Latency** = time from the start of a hijack session to the end of the first window that
  raises an alarm, assuming the detector runs once each time a window closes.

## Results

### Data checks (V1–V7)

| ID | Check | Result |
|---|---|---|
| V1 | Field coverage | type and time 100%; path 45.2% of events; isLocalIP 0.55%; location 0.24% |
| V2 | Aggregation | sum of events in user-day tables = 50,522,931; difference **0** |
| V3 | uid-swap equivalence | swap on raw events = swap on summary tables; **0** of 83,147 user-days differ |
| V4 | Baseline reproduction | TP 13, FP 3,337, TN 69,112, FN 7, train 5,289 — **identical** to Landauer's README |
| V5 | Causal features | changing data after day t on 100 users changes **0** feature values on days ≤ t |
| V6 | No label leakage | 57 input columns (49 event types + F1–F8); no role, login name, IP, location or label |
| V7 | Reproducibility | D run twice on seed 3 with random_state 0: **identical** metrics |
| – | Evaluator check | new evaluator on A's daily decisions: **0** differences over 77,758 user-days |

### Parameter tuning (P1–P5, seeds 0 + 2)

| Scenario | Candidates | Chosen | Note |
|---|---|---|---|
| P1 history window W | 14 / 30 / 60 days | **14** | FPR 0.112 / 0.144 / 0.141 |
| P2 weights (a, c, h) | 4 combinations | **(0.5, 0.4, 0.1)** | FPR 0.099 vs 0.112 for (0.6, 0.3, 0.1) |
| P3 history decay λ | 0.3 / 0.5 / 0.7 | **0.3** | FPR 0.0988–0.1001 |
| P4 threshold θ_R | 0.50–0.90 | **0.55** | TPR 0.65, FPR 0.0988 |
| P5 flag thresholds ±20% | 14 variations | unchanged | FPR 0.096–0.103, TPR 0.65–0.675 |

### Test 1 — uid swap, seed 1 (20 attacked user-days)

| Model | θ | TP/20 | TPR | TPR_adj | FPR | R | Alerts/day |
|---|---|---|---|---|---|---|---|
| **A – Landauer** | 0.60 | **13** | **0.65** | 0.65 | **0.046** | **0.068** | **1.75** |
| B – IF, basic features | 0.52 | 5 | 0.25 | 0.35 | 0.227 | 0.477 | 4.83 |
| B+RS | 0.50 | 6 | 0.30 | 0.40 | 0.108 | 0.192 | 3.57 |
| C – IF, contextual features | 0.80 | 11 | 0.55 | 0.75 | 0.128 | 0.212 | 4.10 |
| D – proposed | 0.55 | 11 | 0.55 | 0.75 | 0.099 | 0.155 | 3.40 |

Threshold sweep of D (θ_R locked at 0.55 from tuning):

| θ_R | TP/20 | TPR | FPR | Alerts/day |
|---|---|---|---|---|
| 0.55 (locked) | 11 | 0.55 | 0.099 | 3.40 |
| 0.60 | 10 | 0.50 | 0.068 | 2.48 |
| 0.65 | 10 | 0.50 | 0.045 | 1.71 |
| 0.70 | 7 | 0.35 | 0.028 | 1.07 |

Risk-score components (R1–R4):

| Variant | θ | TP/20 | TPR | FPR | Alerts/day |
|---|---|---|---|---|---|
| D (reference) | 0.55 | 11 | 0.55 | 0.099 | 3.40 |
| R1 D without history h | 0.55 | 12 | 0.60 | 0.117 | 3.96 |
| R2 D without context c | 0.75 | 10 | 0.50 | 0.120 | 3.76 |
| R3 D without IF score a | 0.50 | 7 | 0.35 | 0.027 | 1.03 |
| R4 gate (a ≥ 0.81 and c ≥ 0.25) | – | 13 | 0.65 | 0.079 | 2.85 |

Feature groups (removed from C; C has FPR 0.128) and single flags (removed from D; D has FPR 0.099):

| Removed from C | TPR | FPR | | Removed from D | TPR | FPR |
|---|---|---|---|---|---|---|
| F8 burst | 0.45 | 0.175 | | k1 volume spike | 0.60 | **0.078** |
| F1 volume | 0.55 | 0.163 | | k2 behaviour shift | 0.50 | 0.105 |
| F7 sensitive actions | 0.60 | 0.160 | | k3 off-hours | 0.60 | 0.137 |
| F6 new resources | 0.45 | 0.155 | | k4 new resource | 0.45 | 0.133 |
| F2, F3 action mix | 0.50 | 0.147 | | k5 sensitive/burst | 0.55 | 0.127 |
| F4, F5 time | 0.55 | 0.134 | | | | |

Adaptation (D1–D3):

| Variant | TPR | FPR | R |
|---|---|---|---|
| D1 trained once, no retraining | 0.60 | 0.123 | 0.218 |
| D2 retrained daily (instead of weekly) | 0.60 | 0.099 | 0.156 |
| D3 no per-user retraining mode (r = 0) | 0.60 | 0.089 | 0.003 |

Statistical tests (E6, McNemar on false positives per user-day, bootstrap 95% CI):

| Comparison | FP only model 1 | FP only model 2 | p | ΔFPR [95% CI] |
|---|---|---|---|---|
| D vs A | 4,782 | 1,345 | < 10⁻³⁰⁰ | +0.053 [+0.043, +0.062] |
| D vs C | 925 | 2,755 | 9×10⁻²⁰⁹ | −0.029 [−0.036, −0.022] |
| C vs B | 870 | 5,189 | < 10⁻³⁰⁰ | −0.099 [−0.119, −0.079] |

Pair difficulty (K1–K2):

| Group | n | TPR A | TPR D | TPR_adj D |
|---|---|---|---|---|
| K1 easy (similarity < 0.3) | 6 | 0.67 | 0.50 | 0.83 |
| K2 hard (similarity ≥ 0.3) | 14 | 0.64 | 0.57 | 0.71 |

Operational load (H3): A raises 3,350 alerts (1.75/day, 13 TP); D raises 6,501 (3.40/day, 11 TP).

**Outcome of Test 1:** H1 is not supported. On the uid swap, D has about twice the FPR of A
and a lower TPR; the difference is significant. Contextual features and risk scoring do
lower the FPR of the Isolation Forest (C < B, D < C).

### Post-hoc exploration on the uid swap, seed 1

Designed after Test 1 results were known, so these do not replace H1.

| Model | θ | TP/20 | TPR | FPR | Alerts/day | ΔFPR vs A [95% CI] |
|---|---|---|---|---|---|---|
| A – Landauer | 0.60 | 13 | 0.65 | 0.046 | 1.75 | – |
| D – proposed | 0.55 | 11 | 0.55 | 0.099 | 3.40 | +0.053 [+0.043, +0.062] |
| Hybrid (w 0.5/0.3/0.2) | 0.40 | 12 | 0.60 | 0.061 | 2.27 | +0.015 [+0.012, +0.018] |
| Hybrid gate (a > 0.48 and c ≥ 0.25) | – | 11 | 0.55 | 0.056 | 2.09 | +0.010 [+0.007, +0.012] |
| X1 C, relative features | 0.87 | 11 | 0.55 | 0.100 | 3.47 | +0.053 [+0.041, +0.067] |
| X2 C, relative + calibration | 0.83 | 14 | 0.70 | 0.157 | 5.05 | +0.111 [+0.104, +0.117] |
| X3 D, relative features | 0.61 | 12 | 0.60 | 0.070 | 2.54 | +0.024 [+0.018, +0.030] |
| **X4 D, relative + calibration** | 0.60 | **15** | **0.75** | 0.078 | 2.83 | +0.032 [+0.027, +0.038] |

Detector benchmark on the X4 setup:

| Detector | θ | TP/20 | TPR | FPR | Alerts/day | PR-AUC | Seconds/seed |
|---|---|---|---|---|---|---|---|
| **Isolation Forest** | 0.60 | **15** | **0.75** | 0.078 | 2.83 | 0.0046 | 69 |
| ECOD | 0.58 | 14 | 0.70 | 0.084 | 3.00 | 0.0051 | 21 |
| COPOD | 0.57 | 12 | 0.60 | 0.093 | 3.28 | 0.0055 | 21 |
| HBOS | 0.62 | 10 | 0.50 | 0.064 | 2.36 | 0.0047 | 3 |
| LOF | 0.51 | 8 | 0.40 | 0.131 | 4.33 | 0.0083 | 25 |

### Test 2 — K3 session hijack, per day, seed 1 (60 hijack sessions)

Thresholds are taken from Test 1 without re-tuning. 20 sessions per intensity
(2×, 5× and 10× the victim's normal activity, plus share and delete actions). FPR and
alerts/day are averaged over the three intensity runs.

| Model | 2× | 5× | 10× | Total (60) | FPR | Alerts/day | McNemar vs A |
|---|---|---|---|---|---|---|---|
| A – Landauer | 3 | 2 | 4 | 9 (15%) | 0.046 | 1.74 | – |
| Hybrid | 6 | 8 | 11 | 25 (42%) | 0.061 | 2.26 | p = 3×10⁻⁵ |
| **D – proposed** | **14** | **15** | **17** | **46 (77%)** | 0.099 | 3.41 | **p = 1.5×10⁻¹¹** |
| X4 | 11 | 14 | 17 | 42 (70%) | 0.078 | 2.82 | p = 2×10⁻¹⁰ |

### Test 3 — K3 per window, seed 1 (174 users, 60 sessions)

Every model is held to the same alert budget: at most 1.75 false alarms per day (A's daily
rate), with θ chosen on seeds 0 + 2.

| Window | Model | θ | Sessions detected (60) | Median latency | p90 latency | False alarms/day |
|---|---|---|---|---|---|---|
| 30 min | A-win | 0.93 | 5 (8%) | 19 min | 24 min | 1.72 |
| 30 min | **X4-win** | 0.73 | **32 (53%)** | 48 min | 109 min | 1.53 |
| 15 min | A-win | 0.999 | 1 (2%) | 59 min | – | 1.50 |
| 15 min | **X4-win** | 0.74 | **35 (58%)** | 33 min | 103 min | 1.56 |

### Test 3b — improved window detector (X5), seed 1 (60 sessions)

Same budget (≤ 1.75 false alarms/day), θ chosen on seeds 0 + 2.
K = 3 windows of accumulation at 30 min, K = 2 at 60 min.

| Window | Model | θ | Sessions detected (60) | Median latency | p90 latency | False alarms/day |
|---|---|---|---|---|---|---|
| 30 min | A-win | 0.93 | 5 (8%) | 19 min | 24 min | 1.72 |
| 30 min | X4-win | 0.73 | 32 (53%) | 48 min | 109 min | 1.53 |
| 30 min | X5-base (EWMA baselines) | 0.74 | 36 (60%) | 57 min | 148 min | 1.55 |
| 30 min | X5-acc (+ accumulation) | 1.92 | 42 (70%) | 79 min | 110 min | 1.62 |
| 30 min | **X5-full (+ suppression)** | 1.80 | **49 (82%)** | 79 min | 108 min | 1.65 |
| 60 min | A-win | 0.89 | 10 (17%) | 31 min | 48 min | 1.74 |
| 60 min | X4-win | 0.72 | 27 (45%) | 68 min | 153 min | 1.41 |
| 60 min | X5-base | 0.73 | 36 (60%) | 79 min | 130 min | 1.56 |
| 60 min | X5-acc | 1.295 | 42 (70%) | 87 min | 131 min | 1.74 |
| 60 min | **X5-full** | 1.257 | **45 (75%)** | 84 min | 131 min | 1.74 |

### Summary

| Test | Attack | A (Landauer) | Proposed |
|---|---|---|---|
| 1. uid swap, per day | 20 user-days | **13/20**, FPR 0.046 | D 11/20, FPR 0.099 (X4 post-hoc: 15/20, FPR 0.078) |
| 2. Session hijack, per day | 60 sessions | 9/60 (15%) | **D 46/60 (77%)** |
| 3. Session hijack, 15-min window | 60 sessions | 1/60 (2%) | **X4-win 35/60 (58%)**, median 33 min |
| 3b. Session hijack, 30-min window | 60 sessions | 5/60 (8%) | **X5-full 49/60 (82%)**, median 79 min |
| 3b. Session hijack, 60-min window | 60 sessions | 10/60 (17%) | **X5-full 45/60 (75%)**, median 84 min |

The baseline is better at detecting a full identity swap, but the proposed model detects
far more hijack sessions at the same false-alarm budget.

## Folders

- `src/` — the pipeline used for all results (DuckDB + Parquet).
- `legacy/` — the first chunked pipeline that reads `clue.zip` directly, plus `run_equiv.sh`,
  which shows the chunked pipeline equals Landauer's original scripts on the first 3 M events.
  It needs the original scripts from [ait-aecid/clue-lds](https://github.com/ait-aecid/clue-lds)
  (`make_detect_chunk.py` builds `detect_chunk.py` from Landauer's `detect.py`). It is not needed
  to reproduce the results; `src/duck_seed.py` gives the same TP/FP.
- `src/v3.py` — check V3 (DuckDB = chunked pipeline); needs the output of `legacy/run_seed.sh`.
- `tools/` — builds result tables (`results_table.py`) and fills the scenario workbook (`fill_workbook.py`).
- `expected_results/` — the original results from the thesis runs, for comparison.

## Reference

Landauer, M., Skopik, F., Höld, G., & Wurzenberger, M. (2022). A User and Entity Behavior
Analytics Log Data Set for Anomaly Detection in Cloud Computing. IEEE BigData 2022.
