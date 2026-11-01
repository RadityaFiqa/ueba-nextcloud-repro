# Exploratory scripts

Post-hoc experiments run while designing the final model. They are kept for transparency and are
**not** needed to reproduce the numbers of the thesis or the paper (use `src/step12_final.sh`).
Each script states in its header what it tested, which seeds it used and where it writes its result.
Most of them import `x4_refine_core.py` from this folder and expect `work/transitions.parquet`
(built by `src/step12_final_prep.py T`).

| Script | Question |
|---|---|
| `feature_ablation.py`, `feature_screen_swap.py` | which context features matter on uid swaps / hijacks |
| `fpr_reduction.py`, `fpr_reduction2.py`, `x4_dgate.py` | ways to lower X4's false alarms (incl. a Landauer-distance gate, rejected) |
| `x4_seeds.py`, `landauer_extended.py` | X4 and Landauer on seeds 0–9 |
| `x4_extra_features.py`, `x4_extra_ablation.py`, `x4k6_*.py` | order-of-activities features and flag k6; ablation of extra features |
| `x4_window.py` | history length W = 7 / 14 / 25 days |
| `peruser_models.py` | one Isolation Forest per user (stopped; not used) |
| `x4_refine*.py`, `x5_*.py` | flag weights / burst flag tuning and operating curve |
| `landauer_curve.py`, `threshold_curves.py`, `uniform_threshold.py` | detectors at several thresholds |
| `hijack_flags_v5.py`, `trace_uidswap.py`, `x4k6_trace.py`, `x4k6_example.py` | traces of single attacks |
| `bench_x4k6.py` | compute-cost benchmark (not run) |
