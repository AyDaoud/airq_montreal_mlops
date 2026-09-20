# Spec B — Honest Evaluation and the Alert Product

- **Date:** 2026-09-18
- **Status:** Approved (pending spec review)
- **Branch:** `spec-b/honest-evaluation`, based on `spec-a/data-foundation`
- **Predecessor:** Spec A — Data Foundation (complete, 76 tests, CI green)
- **Successors:** Spec C (observability), Spec D (automation and cloud)

---

## 1. Context

Spec A made the pipeline real and deliberately claimed no model accuracy. Measurements taken at the end of Spec A explain why, and they reframe what Spec B is for.

### 1.1 The regression loses to doing nothing

Honest time split (train before 2025-03-25, test after), predicting tomorrow's IQA, reproducible via `python -m scripts.compare_baselines`:

| Approach | MAE | RMSE |
|---|---|---|
| **persistence — "tomorrow = today"** | **7.380** | 14.737 |
| RF as built (today's IQA excluded) | 8.324 | 16.844 |
| RF + today's IQA restored | 8.069 | 16.807 |
| RF on a Δ-from-today target | 8.511 | — |
| seasonal-naive — same weekday last week | 11.238 | 22.211 |

The trained model is roughly **9% worse than the trivial baseline**. Two corrections follow:

- The `mae_va ≈ 4.6` the training code reports is inflated by about **43%**, because `_time_split` produces a station holdout rather than a time holdout.
- A **Δ-from-today target does not help** (8.511, the worst of the three). This was proposed as the likely fix before it was measured; it is recorded here so the spec does not build on it.

### 1.1b A model family that DOES beat persistence — measured across 5 folds

A bake-off across model families, then validated on 5 rolling-origin folds (90-day test windows, expanding):

| fold | cut | persistence MAE | **Huber on Δ** | Ridge on Δ | RF on level |
|---|---|---|---|---|---|
| 1 | 2024-10-24 | 6.301 | **0.954** | 1.037 | 0.913 |
| 2 | 2025-01-22 | 5.445 | **0.904** | 0.998 | 0.901 |
| 3 | 2025-04-22 | 8.239 | **0.892** | 0.948 | 1.068 |
| 4 | 2025-07-21 | 10.270 | **0.867** | 0.905 | 1.066 |
| 5 | 2025-10-19 | 4.843 | **0.877** | 0.962 | 0.930 |
| | **mean MASE** | — | **0.899** | 0.970 | 0.976 |
| | **folds won** | — | **5/5** | 4/5 | 3/5 |

**Huber regression on a Δ-from-today target beats persistence by ~10%, in every fold.**

Why the tree ensembles failed, and why this works:

- Tree ensembles average over leaves, shrinking predictions toward the training mean. On a strongly persistent, near-random-walk series that is exactly the wrong inductive bias.
- A **linear** model on the *difference* has no such shrinkage: it starts from today's value and learns only the correction.
- **Huber loss** is robust to the heavy-tailed exceedance days that dominate squared error (recall RMSE 14.7 against MAE 7.4 for persistence — a 2× ratio).

**CORRECTION (measured after this section was written).** A 2×2 backtest isolating loss from framing shows the Δ-target contributes **exactly nothing**:

| | level | delta | effect of framing |
|---|---|---|---|
| **huber** | 0.8990 | 0.8990 | **+0.0000** |
| **ridge** | 0.9701 | 0.9701 | **+0.0000** |
| *effect of loss* | −0.0711 | −0.0711 | |

**The entire win is the Huber loss, not the reframe.** Because `iqa` is itself a feature, a *linear* model spans the same hypothesis space either way — `y = iqa + w·x` is reachable from both parametrisations, differing only through the L2 penalty (~3e-05 in MASE). Huber wins because squared error is dominated by the heavy-tailed exceedance days, and Huber loss is linear beyond its threshold so those days stop dragging the fit.

The framing is *not* a no-op for trees, which cannot represent "start from today" internally — there it actively hurts (8.511 vs 8.324). `huber_level` is kept in the registry so the scoreboard demonstrates the equivalence rather than asserting it.

**The model family was the problem, not the features.** §1.1 recorded that a Δ-target made things worse — that was true *with a RandomForest* (8.511). The same reframe with a linear robust regressor wins. Target framing and model family had to change together; testing either alone was misleading.

**A methodological warning this produced.** RF scored MASE 1.093 on the single split in §1.1 but 0.976 averaged over 5 folds, winning 3 of them. A single split mislead by ~12%. This is the direct justification for the rolling-origin protocol in §4 — no conclusion in this spec may rest on one split.

### 1.2 The exceedance problem is where signal exists

Same split, predicting whether IQA exceeds 50 tomorrow:

```
test base rate:            4.95%   (160 of 3,232)
persistence classifier:    precision 0.531   recall 0.531
RF, lags only:             PR-AUC 0.3798   ->  7.7x lift over random
RF, lags + weather:        PR-AUC 0.2587   ->  5.2x lift
```

There is real signal here — 7.7× lift over the base rate — but persistence is *also* a strong classifier baseline and must be plotted alongside.

### 1.3 Weather may be actively harmful — contradicting Spec A's premise

Adding weather features drops classifier PR-AUC from 0.3798 to 0.2587. Spec A was designed partly on the claim that weather would be the project's "biggest modelling unlock". That claim is **not supported** by this measurement.

Probable cause: roughly 20 features against only 299 training positives, with trees splitting spuriously on high-cardinality continuous weather. This is a single split with no regularisation, so it is **preliminary**. Spec B must test it across all folds before accepting or rejecting it, and must publish the answer either way.

---

## 2. Goals and non-goals

### Goals
1. An evaluation protocol whose numbers cannot be quietly misread.
2. Fix the three defects Spec A pinned as `strict=True` xfails.
3. Make the exceedance alert the project's primary deliverable.
4. Calibrated uncertainty on the regression.
5. A published, regenerable scoreboard.

### Also in scope (expanded after the §1.1b finding)
6. **Model family selection is a first-class deliverable.** Huber-on-Δ is the incumbent to beat; the backtest evaluates linear, robust-linear, boosted and tree families on equal terms and the scoreboard picks the winner on mean MASE across folds.

### Non-goals (deferred to a bounded follow-up)
- Exhaustive hyperparameter search (a small, declared grid is in scope; a large search is not)
- Per-station retraining of Prophet/LSTM beyond one scoreboard run
- Anything in Spec C (dashboards, Postgres, Streamlit) or Spec D (Terraform, Cloud Run)

### An explicitly permitted outcome

**Spec B may conclude that daily IQA at h=24 is not predictable beyond persistence.** (The §1.1b result suggests it is, by roughly 10% — but that is a preliminary bake-off, not the spec's own backtest, and the spec must be free to reach a different answer.) That is a valid, publishable result. No acceptance criterion requires a model to win. A spec that can only succeed by producing a winner builds in a reason to fool ourselves, which is the failure this project has been correcting.

---

## 3. Architecture

```
src/evaluation/
  splits.py      # date-based train/test split; rolling-origin folds   <- fixes xfail 1
  baselines.py   # persistence, seasonal-naive(7), climatology
  metrics.py     # MAE, RMSE, MASE; PR-AUC, precision/recall; coverage
  backtest.py    # runs any estimator across folds x horizons
  conformal.py   # split-conformal intervals
src/models/
  series.py      # per-station series accessor                         <- fixes xfail 2
  classifier.py  # exceedance classifier + cost-based threshold
scripts/
  run_backtest.py  # regenerates docs/RESULTS.md, logs to MLflow
```

Each module has one responsibility, is tested in isolation, and does not reach into another's internals. `backtest.py` depends on `splits`, `metrics` and `baselines` through their public functions only.

---

## 4. Evaluation protocol

**Rolling origin, expanding window, 5 folds.** Fold *k* trains on all data before cutoff *k* and tests the following 90 days. **Cutoffs are dates, never row indices** — slicing by row on a frame sorted by `[station_id, date_local]` is precisely the defect being fixed.

**Horizons: 24h, 48h, 72h**, reported as a curve. Persistence decays with horizon; if a model ever wins it will win at h≥48, and reporting only h24 would hide that.

**Headline metric is MASE, scaled to persistence:**

```
MASE = MAE_model / MAE_persistence      1.00 ties, <1.00 beats, >1.00 loses
```

Today's model scores about **1.09**. Scaling to persistence makes every cell instantly interpretable and makes the comparison impossible to drop silently.

**Baselines evaluated in every fold at every horizon:** persistence, seasonal-naive(7), climatology (station × calendar-month mean).

**Gap handling.** The gold table contains a 239-day hole between the frozen historical dump (ends 2026-01-18) and the isolated realtime days. Folds are built on the contiguous block only; `splits.py` takes an explicit `max_date` and the scoreboard records it.

---

## 5. The alert product

Primary deliverable: **P(IQA > 50 tomorrow)**, per station.

Reported as:
- **PR curve and PR-AUC** against the base rate. Never accuracy — a model that always predicts "no" scores about 97%.
- **Persistence plotted as a single point** on the same axes (currently precision 0.531 / recall 0.531). A classifier that cannot beat that point has not earned its complexity.
- **Operating threshold from a stated cost ratio.** A missed exceedance costs **5× a false alarm** — public-health framing, where failing to warn is worse than warning unnecessarily. The ratio lives in config, and the README states it in prose; it is never an unexplained default.
- **A calibration curve.** A probability that is not calibrated is not a decision aid.
- **The weather question settled**: the classifier is fitted with and without weather features across all folds, with regularisation, and the result is published either way.

---

## 6. Conformal intervals

Split-conformal on the regression: reserve a calibration slice inside each training fold, take the empirical quantile of absolute residuals at nominal 90%, emit `ŷ ± q`.

**Empirical coverage is a reported metric, not a footnote.** The purpose of the interval is honesty; an interval covering 70% of outcomes while claiming 90% is a defect. Acceptance requires coverage within ±3 percentage points of nominal.

---

## 7. The three pinned defects

| xfail test | Defect | Fix |
|---|---|---|
| `test_time_split_holdout_starts_after_train_ends` | "time split" is a station split | `splits.py` splits on dates |
| `test_series_builder_yields_one_series_per_station` | Prophet/LSTM flatten 11 series into one | `series.py` accessor |
| `test_current_day_iqa_is_available_as_a_feature` | today's IQA excluded from features | add `iqa` to the feature set |

Each `strict=True` marker is **removed in the same commit as its fix**. That is what strict mode exists for: a stale marker becomes a failure.

Note from §1.1: fixing the third defect improves MAE only 8.324 → 8.069. It is correct to fix, but it is not what limits the model.

---

## 8. Decisions

| # | Decision | Rationale |
|---|---|---|
| B1 | MASE against persistence is the headline metric | Makes every number interpretable and the comparison undroppable |
| B2 | 5 rolling-origin folds, 90-day test windows, expanding | Enough folds for a spread, few enough to stay fast |
| B3 | Horizons 24/48/72h reported as a curve | A single horizon hides where persistence decays |
| B4 | Cost ratio 5:1 (missed exceedance : false alarm) | Public-health framing; stated in the README, not buried |
| B5 | Keep the per-station accessor; run Prophet/LSTM **once** for the scoreboard | Fixes the xfail honestly; results decide whether they stay in the repo |
| B6 | Conformal coverage is an acceptance criterion | An uncovered interval is a bug |
| B7 | Folds built on the contiguous block only, `max_date` explicit | The 239-day gap would otherwise corrupt fold boundaries |
| B8 | Publish the weather result either way | Spec A's premise is under test; retract in writing if wrong |

---

## 9. Testing

- `splits.py` — folds never overlap; test always follows train in time; a frame sorted by `[station_id, date_local]` still yields a time split (the regression test for the original defect)
- `baselines.py` — persistence equals `iqa.shift(1)` per station and never crosses a station boundary; climatology handles an unseen month
- `metrics.py` — MASE of persistence against itself is exactly 1.0; PR-AUC of a random scorer approximates the base rate
- `conformal.py` — coverage on synthetic data with known noise lands within tolerance
- `classifier.py` — the cost-based threshold moves in the expected direction as the ratio changes
- `backtest.py` — an estimator that returns a constant produces the expected MASE

All tests remain offline and fixture-based, consistent with Spec A.

---

## 10. Acceptance criteria

1. `python -m scripts.run_backtest` regenerates `docs/RESULTS.md` end to end.
2. Every model reports MASE against persistence at h24/h48/h72 across all 5 folds.
3. The classifier reports PR-AUC, a calibration curve, and precision/recall at the 5:1 threshold, with persistence plotted alongside.
4. Conformal coverage is within ±3pp of nominal 90%.
5. All three xfail markers are removed and the suite passes without them.
6. The README Results section is replaced with the real scoreboard.
7. **If nothing beats persistence, the README summary line says so plainly.**
8. The weather question is answered across all folds and published.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Nothing beats persistence | An explicitly permitted outcome (§2). The deliverable is the evidence, not a win. |
| 299 training positives is thin for the classifier | Report fold-level spread, not just a mean; treat a single-fold result as provisional |
| Rolling backtest is slow with Prophet/LSTM | They run once for the scoreboard (B5); the RF and baselines run every fold |
| Conformal coverage fails | That is a finding, not a blocker: report it and investigate rather than tuning until it passes |
| Scope creep into tuning | Hyperparameter search is an explicit non-goal; the harness comes first so tuning has a scoreboard to aim at |
