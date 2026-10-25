# Model v2 spec (DRAFT for review)

Status: **draft, not approved**. Nothing in E2–E5, R*, F*, D*, P*, K* runs until the open decisions (§7) are settled.

Sources: hints in `Outcome_and_ATM.xlsx` (ATM sheets) and `Skenario_Pengujian.xlsx` (Parameter Default). Everything marked **(proposal)** is my choice and needs your OK.

Evaluation follows the decision of 2026-10-22: **single run, seed 1, all 5,389 users**, Landauer's protocol (user-day unit, TP = first active day after the swap, retraining days not counted, r = 1).

---

## 1. Unit and history

- Unit: one **user-day** (uid × UTC calendar day), the same as Landauer.
- History H(u, t): the user's previous **W = 30 active days** before day t (strictly before, never day t itself → causal, checked by V5).
- Minimum history **M = 5 active days** for the robust z-scores; with 1–4 days of history the z-score features are set to 0 (neutral) **(proposal)**. With 0 days of history (the user's first day) the day is a training day, exactly as in Landauer.

## 2. Set A: basic features (49 columns)

Count of each of the 49 event types on day t (`file_accessed`, `login_attempt`, …). Used as model input in B and B+RS. The columns are not user-relative.

## 3. F1–F8: user-relative contextual features

| ID | Group | Definition **(proposal where noted)** | Range |
|---|---|---|---|
| F1 | Vol | Robust z of `log(1 + events_t)` vs H: `(x − median_H) / (1.4826·MAD_H + 1e-6)`, clipped to [−10, 10]. Median/MAD instead of mean/std (ATM: sync bursts) | ℝ |
| F2 | Mix | Jensen–Shannon divergence (base 2) between today's event-type distribution and the pooled distribution over H (ATM: JS instead of KL) | [0, 1] |
| F3 | Mix | Share of today's events whose type never occurs in H **(proposal)** | [0, 1] |
| F4 | Time | Out-of-hours fraction: share of today's events whose hour-of-day (fractional, UTC) falls outside the user's [P2.5, P97.5] range over H (Al-Smadi) | [0, 1] |
| F5 | Time | Robust z of the daily active span (last − first event, hours) vs H **(proposal)** | ℝ |
| F6 | Res | Path novelty: share of today's path-bearing events whose 2-level folder prefix (`/a/b`) never occurs in H. 0 when no path event today (path present on 45% of events / 66% of user-days, V1) | [0, 1] |
| F7 | Sens | Excess sensitive-action ratio: `max(0, r_t − mean_H(r))`, where r = share of events in the sensitive set (below) **(proposal: set and "excess")** | [0, 1] |
| F8 | Burst | Robust z of the day's peak 10-minute event count vs H **(proposal: 10 min window)** | ℝ |

Sensitive set (F7) **(proposal)**: deletion (`file_deleted`, `deleted_from_trashbin`, `version_deleted`), sharing (`shared_user`, `shared_link`, `shared_group`, `shared_email`, `public_share_update_permission`, `public_share_password_changed`, `public_share_expiration_date_changed`, `permission_changed`), account/admin (`password_changed`, `email_changed`, `user_created`, `user_deleted`, `user_disabled`, `user_added_to_group`, `user_became_groupadmin`, `admin_settings_changed`), and `login_failed`.

Excluded from every model input (V6): `role`, `loginName`, `isLocalIP`, `location`, IP-type uids' metadata, and labels. Per V1, IP/location exist on only 0.26% of named-user events (failed logins only), so leaving them out costs nothing.

**Model inputs:** B = Set A (49 columns); C and D = Set A + F1–F8 (57 columns). This matches the "57 kolom" in V6.

## 4. Isolation Forest (component a)

- **Global** model over all users, `n_estimators = 200`, `max_samples = 256`, `random_state = 0`.
- Training window **T = 60 calendar days** before the retrain date; **retrain every Δ = 7 days**. For the first 60 days of the dataset the window expands from day 1, with the first model at day 7 **(proposal)**. Days before the first model are training days for B/C/D.
- Training uses every user-day in the window. No labels are used, so attacked days can be in training data, which is realistic.
- Score → **a ∈ [0, 1]**: the empirical percentile of the day's anomaly score among the training scores (`a = share of training days less anomalous`) **(proposal)**. This makes a comparable across retrains.

## 5. Flags k1–k5 and context score c

Binary flags computed from F1–F8 **(proposal: thresholds and weights)**, total weight 8:

| Flag | Condition | Weight |
|---|---|---|
| k1 volume spike | F1 ≥ 3 | 2 |
| k2 behaviour shift | F2 ≥ 0.5 **or** F3 ≥ 0.2 | 2 |
| k3 off-hours | F4 ≥ 0.3 | 1 |
| k4 new resources | F6 ≥ 0.5 | 2 |
| k5 sensitive / burst | F7 ≥ 0.1 **or** F8 ≥ 3 | 1 |

`c_t = Σ w_k · k_k / 8 ∈ [0, 1]`.

## 6. History h, risk R and alert

- `h_t = λ · h_{t−1} + (1 − λ) · R_{t−1}`, with h = 0 on the user's first scored day and **λ = 0.5**, decayed per active day of the user **(proposal)**.
- `R_t = 0.6·a_t + 0.3·c_t + 0.1·h_t`.
- Alert if `R_t ≥ θ_R`. θ_R comes from P4 (sweep 0.50–0.90): the operating point with the lowest FPR at TPR ≥ 65%.
- Splunk tiers for reporting only: Low < 0.31, Medium 0.31–0.60, High 0.61–0.90, Critical > 0.90.
- Model B / C alert rule (no risk scoring): alert if `a_t ≥ θ_a`, with θ_a swept in the same way.
- Per-user retraining mode **r = 1**, as in Landauer: after an alert the user's next active day is a training day (not counted). An alert during training extends it.

## 7. Open decisions (please answer)

1. **Single-run consequences.** With only seed 1, P1–P5 tune on the same run that E1–E6 report. Your rule "tidak ada tuning di seed uji" can't hold. Option: tune on seeds 0–2 (validation only, never reported as results) and report only seed 1. This keeps the rule and still gives a single reported run.
2. **E6 statistics.** Wilcoxon over 7 seeds isn't possible with one run. Proposal: **McNemar** on paired user-day decisions (D vs A on normal user-days for FPR), plus a bootstrap 95% CI over users for ΔFPR.
3. **Minimum history.** Neutral z-scores for days with 1–4 days of history (above) keep the evaluated user-days and R comparable with Landauer. Making those days training days would raise R from about 0.068 to roughly 0.14 and fail the "R ≤ baseline" outcome.
4. All the **(proposal)** items above: F3, F5, F7 (set and excess), F8 window, a-calibration, the first-model rule, flag thresholds and weights, and h.
5. **K3 (M7 session injection):** I need session length, actions, counts and the three intensities, fixed before any results are seen.
