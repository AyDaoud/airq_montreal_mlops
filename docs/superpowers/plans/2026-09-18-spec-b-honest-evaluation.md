# Spec B — Honest Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation harness whose numbers cannot be misread, use it to select a model that genuinely beats persistence, and ship an exceedance alert with a stated cost trade-off.

**Architecture:** A small `src/evaluation/` package — date-based splits, baselines, metrics, a backtest driver, conformal intervals — that any estimator can be run through. A model registry declares the candidates. `scripts/run_backtest.py` regenerates `docs/RESULTS.md` from scratch, so every published number is reproducible by one command.

**Tech Stack:** Python 3.12, pandas 2.2, scikit-learn 1.7 (HuberRegressor, Ridge, RandomForest, HistGradientBoosting), MLflow, pytest.

**Reference spec:** `docs/superpowers/specs/2026-09-18-spec-b-honest-evaluation-design.md`

---

## Environment notes

- Branch `spec-b/honest-evaluation`, based on `spec-a/data-foundation`. Spec A is **not merged to main**; do not branch from main.
- Always run python as `.venv/bin/python`. Baseline suite: **76 passed, 3 xfailed**.
- The gold table `data/gold/daily_station_iqa.parquet` exists locally (gitignored): 16,195 rows, 29 columns.
- Tests must stay **offline**. `data/` is not available in CI.
- The contiguous data block ends **2026-01-18**; later rows are isolated realtime days behind a 239-day gap. All evaluation restricts to the contiguous block.

## Known-good numbers to reproduce

These were measured before the plan was written. Tasks reference them as sanity gates.

| Thing | Value |
|---|---|
| persistence MAE, single split at 2025-03-25 | 7.380 |
| Huber-on-Δ mean MASE over 5 folds | **0.899**, wins 5/5 |
| Ridge-on-Δ mean MASE | 0.970, wins 4/5 |
| RF-on-level mean MASE | 0.976, wins 3/5 |
| exceedance base rate (test) | 4.95% |
| classifier PR-AUC, lags only | 0.3798 |
| classifier PR-AUC, lags + weather | 0.2587 |

## File structure

| File | Responsibility |
|---|---|
| `src/evaluation/splits.py` | Date-based split and rolling-origin folds. No model knowledge. |
| `src/evaluation/baselines.py` | persistence, seasonal-naive(7), climatology. Pure functions over a frame. |
| `src/evaluation/metrics.py` | MAE/RMSE/MASE, PR-AUC, precision/recall, interval coverage. |
| `src/evaluation/registry.py` | Declares candidate estimators and their target framing. |
| `src/evaluation/backtest.py` | Runs the registry across folds × horizons. Depends only on the above. |
| `src/evaluation/conformal.py` | Split-conformal intervals and empirical coverage. |
| `src/models/series.py` | Per-station series accessor. Fixes xfail 2. |
| `src/models/classifier.py` | Exceedance classifier and cost-based threshold selection. |
| `scripts/run_backtest.py` | Regenerates `docs/RESULTS.md`; logs to MLflow. |

---

## Task 1: Date-based splits and rolling-origin folds

Fixes xfail 1. This is the defect that inflated the reported `mae_va` by ~43%.

**Files:**
- Create: `src/evaluation/__init__.py`, `src/evaluation/splits.py`
- Test: `tests/evaluation/__init__.py`, `tests/evaluation/test_splits.py`

- [ ] **Step 1: Create package directories**

```bash
mkdir -p src/evaluation tests/evaluation
touch src/evaluation/__init__.py tests/evaluation/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/evaluation/test_splits.py`:

```python
import pandas as pd
import pytest

from src.evaluation.splits import rolling_origin_folds, time_split


def _frame(n_days=400, n_stations=3):
    """Deliberately sorted by [station_id, date_local] - the shape that broke
    the original row-index split."""
    rows = []
    for station_id in range(n_stations):
        for day in range(n_days):
            rows.append(
                {
                    "station_id": station_id,
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": 20 + (day % 13),
                }
            )
    return pd.DataFrame(rows).sort_values(["station_id", "date_local"])


def test_time_split_holdout_starts_after_train_ends():
    train, test = time_split(_frame(), test_fraction=0.2)
    assert train["date_local"].max() < test["date_local"].min()


def test_time_split_keeps_every_station_on_both_sides():
    """The original defect produced a station holdout, not a time holdout."""
    train, test = time_split(_frame(), test_fraction=0.2)
    assert set(train["station_id"]) == set(test["station_id"])


def test_time_split_loses_no_rows():
    frame = _frame()
    train, test = time_split(frame, test_fraction=0.2)
    assert len(train) + len(test) == len(frame)


def test_rolling_origin_yields_requested_number_of_folds():
    folds = list(rolling_origin_folds(_frame(), n_folds=4, test_days=30))
    assert len(folds) == 4


def test_rolling_origin_train_always_precedes_test():
    for train, test in rolling_origin_folds(_frame(), n_folds=4, test_days=30):
        assert train["date_local"].max() < test["date_local"].min()


def test_rolling_origin_training_window_expands():
    sizes = [len(tr) for tr, _ in rolling_origin_folds(_frame(), n_folds=4, test_days=30)]
    assert sizes == sorted(sizes), f"training window shrank: {sizes}"


def test_rolling_origin_test_windows_do_not_overlap():
    windows = [
        (te["date_local"].min(), te["date_local"].max())
        for _, te in rolling_origin_folds(_frame(), n_folds=4, test_days=30)
    ]
    for earlier, later in zip(windows, windows[1:]):
        assert earlier[1] < later[0], f"overlapping test windows: {earlier} {later}"


def test_max_date_is_respected():
    """The gold table has a 239-day gap after 2026-01-18; folds must not span it."""
    frame = _frame(n_days=400)
    cap = pd.Timestamp("2024-06-01")
    _, test = time_split(frame, test_fraction=0.2, max_date=cap)
    assert test["date_local"].max() <= cap


def test_rejects_a_frame_without_the_date_column():
    with pytest.raises(ValueError, match="date_local"):
        time_split(pd.DataFrame({"iqa": [1, 2, 3]}), test_fraction=0.2)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/evaluation/test_splits.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.splits'`

- [ ] **Step 4: Implement `src/evaluation/splits.py`**

```python
"""Time-respecting splits.

The original ``_time_split`` sliced ``iloc[:k]`` on a frame sorted by
``[station_id, date_local]``, which yields a *station* holdout, not a time
one. Every split here is computed from dates, never from row positions.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

DATE_COLUMN = "date_local"


def _require_dates(df: pd.DataFrame) -> None:
    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Frame has no {DATE_COLUMN!r} column; splits are computed from "
            "dates, never from row positions."
        )


def _capped(df: pd.DataFrame, max_date: pd.Timestamp | str | None) -> pd.DataFrame:
    if max_date is None:
        return df
    return df[df[DATE_COLUMN] <= pd.Timestamp(max_date)]


def time_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    max_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on a date boundary so the holdout strictly follows training."""
    _require_dates(df)
    frame = _capped(df, max_date)
    cutoff = frame[DATE_COLUMN].quantile(1 - test_fraction)
    train = frame[frame[DATE_COLUMN] < cutoff]
    test = frame[frame[DATE_COLUMN] >= cutoff]
    return train, test


def rolling_origin_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_days: int = 90,
    max_date: pd.Timestamp | str | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``n_folds`` expanding-window folds, oldest first.

    Fold *k* trains on everything before its cutoff and tests the following
    ``test_days`` days. Test windows are contiguous and non-overlapping.
    """
    _require_dates(df)
    frame = _capped(df, max_date)
    end = frame[DATE_COLUMN].max()

    cutoffs = [
        end - pd.Timedelta(days=test_days * (k + 1)) for k in range(n_folds)
    ][::-1]

    for cutoff in cutoffs:
        train = frame[frame[DATE_COLUMN] < cutoff]
        test = frame[
            (frame[DATE_COLUMN] >= cutoff)
            & (frame[DATE_COLUMN] < cutoff + pd.Timedelta(days=test_days))
        ]
        if train.empty or test.empty:
            continue
        yield train, test
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/evaluation/test_splits.py -v
```
Expected: 9 passed

- [ ] **Step 6: Remove the now-obsolete xfail marker**

`tests/test_training_split.py::test_time_split_holdout_starts_after_train_ends` is marked `strict=True` xfail against the OLD `_time_split`. That function still exists and is still broken, so the marker is still correct there. **Leave it alone in this task** — Task 10 rewires `training_daily` to the new splits and removes it then.

Verify the old defect is still pinned:
```bash
.venv/bin/python -m pytest tests/test_training_split.py -v
```
Expected: 3 passed, 2 xfailed — unchanged.

- [ ] **Step 7: Commit**

```bash
.venv/bin/flake8 src/evaluation tests/evaluation
.venv/bin/black --check src/evaluation tests/evaluation
.venv/bin/python -m pytest
git add src/evaluation/__init__.py src/evaluation/splits.py tests/evaluation/
git commit -m "feat(eval): add date-based splits and rolling-origin folds

The original _time_split sliced by row index on a frame sorted by
[station_id, date_local], producing a station holdout rather than a time
one and inflating reported validation scores by ~43%."
```

---

## Task 2: Baselines

**Files:**
- Create: `src/evaluation/baselines.py`
- Test: `tests/evaluation/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_baselines.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import climatology, persistence, seasonal_naive


def _frame():
    rows = []
    for station_id in (3, 6):
        for day in range(40):
            rows.append(
                {
                    "station_id": station_id,
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": 10 + day + (100 if station_id == 6 else 0),
                }
            )
    return pd.DataFrame(rows)


def test_persistence_is_todays_value():
    frame = _frame()
    assert (persistence(frame) == frame["iqa"]).all()


def test_seasonal_naive_looks_back_seven_days():
    out = seasonal_naive(_frame(), period=7)
    station = _frame().query("station_id == 3").reset_index(drop=True)
    got = out[: len(station)].reset_index(drop=True)
    assert pd.isna(got[:6]).all(), "first 6 rows have no 7-day history"
    assert got.iloc[10] == station["iqa"].iloc[4]


def test_seasonal_naive_never_crosses_a_station_boundary():
    frame = _frame()
    out = seasonal_naive(frame, period=7)
    first_rows_of_station_6 = out[frame["station_id"].values == 6][:6]
    assert pd.isna(first_rows_of_station_6).all()


def test_climatology_is_the_station_month_mean_of_training_data():
    train = _frame()
    test = _frame()
    out = climatology(train, test)
    expected = train.query("station_id == 3")["iqa"].mean()
    assert out.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_climatology_falls_back_for_an_unseen_station_month():
    train = _frame()
    test = _frame().assign(
        date_local=lambda d: d["date_local"] + pd.DateOffset(months=6)
    )
    out = climatology(train, test)
    assert out.notna().all(), "unseen station-month produced NaN"
    assert out.iloc[0] == pytest.approx(train["iqa"].mean(), rel=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/evaluation/test_baselines.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.baselines'`

- [ ] **Step 3: Implement `src/evaluation/baselines.py`**

```python
"""Do-nothing baselines every model must be measured against.

Persistence is the one that matters: on daily air quality it is strong, and
until Spec B it had never been computed. Every baseline is grouped by
station so no value ever leaks across a station boundary.
"""

from __future__ import annotations

import pandas as pd

VALUE_COLUMN = "iqa"
GROUP_COLUMN = "station_id"
DATE_COLUMN = "date_local"


def persistence(df: pd.DataFrame) -> pd.Series:
    """Tomorrow equals today."""
    return df[VALUE_COLUMN].copy()


def seasonal_naive(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """Tomorrow equals the value ``period`` days before today.

    With ``period=7`` this is "the same weekday last week": shifting by
    ``period - 1`` from today lands on that day for a next-day target.
    """
    return df.groupby(GROUP_COLUMN)[VALUE_COLUMN].shift(period - 1)


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """The training mean for this station and calendar month.

    Falls back to the station mean, then the global mean, so an unseen
    station-month never yields NaN.
    """
    train = train.copy()
    test = test.copy()
    train["_month"] = train[DATE_COLUMN].dt.month
    test["_month"] = test[DATE_COLUMN].dt.month

    by_station_month = train.groupby([GROUP_COLUMN, "_month"])[VALUE_COLUMN].mean()
    by_station = train.groupby(GROUP_COLUMN)[VALUE_COLUMN].mean()
    overall = train[VALUE_COLUMN].mean()

    keys = pd.MultiIndex.from_arrays([test[GROUP_COLUMN], test["_month"]])
    values = by_station_month.reindex(keys).to_numpy()

    out = pd.Series(values, index=test.index)
    out = out.fillna(test[GROUP_COLUMN].map(by_station))
    return out.fillna(overall)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/evaluation/test_baselines.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
.venv/bin/flake8 src/evaluation tests/evaluation
.venv/bin/black --check src/evaluation tests/evaluation
git add src/evaluation/baselines.py tests/evaluation/test_baselines.py
git commit -m "feat(eval): add persistence, seasonal-naive and climatology baselines"
```

---

## Task 3: Metrics, with MASE as the headline

MASE makes every number self-interpreting: `1.00` ties persistence, `<1.00` beats it. It is what stops the comparison being quietly dropped.

**Files:**
- Create: `src/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_metrics.py`:

```python
import numpy as np
import pytest

from src.evaluation.metrics import (
    classification_report_at_threshold,
    interval_coverage,
    mase,
    regression_metrics,
)


def test_regression_metrics_are_correct():
    out = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert out["mae"] == pytest.approx(2.0 / 3.0)
    assert out["rmse"] == pytest.approx(np.sqrt(4.0 / 3.0))


def test_mase_of_persistence_against_itself_is_exactly_one():
    y = np.array([5.0, 7.0, 9.0, 4.0])
    baseline = np.array([4.0, 5.0, 7.0, 9.0])
    assert mase(y, baseline, baseline) == pytest.approx(1.0)


def test_mase_below_one_means_the_model_beat_the_baseline():
    y = np.array([10.0, 10.0, 10.0])
    baseline = np.array([8.0, 8.0, 8.0])   # MAE 2
    model = np.array([9.0, 9.0, 9.0])      # MAE 1
    assert mase(y, model, baseline) == pytest.approx(0.5)


def test_mase_raises_when_the_baseline_is_perfect():
    """A zero-error baseline makes the ratio undefined; fail loudly."""
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="baseline"):
        mase(y, np.array([1.5, 2.5]), y)


def test_classification_report_at_threshold():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.6, 0.9]
    out = classification_report_at_threshold(y, proba, threshold=0.5)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["n_positive"] == 2
    assert out["n_flagged"] == 2


def test_pr_auc_of_a_random_scorer_approximates_the_base_rate():
    rng = np.random.RandomState(0)
    y = (rng.rand(4000) < 0.05).astype(int)
    out = classification_report_at_threshold(y, rng.rand(4000), threshold=0.5)
    assert out["pr_auc"] == pytest.approx(0.05, abs=0.02)


def test_interval_coverage_counts_containment():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    lower = np.array([0.0, 1.0, 2.0, 0.0])
    upper = np.array([2.0, 3.0, 4.0, 5.0])
    assert interval_coverage(y, lower, upper) == pytest.approx(0.75)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/evaluation/test_metrics.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.metrics'`

- [ ] **Step 3: Implement `src/evaluation/metrics.py`**

```python
"""Metrics, with MASE as the headline.

MASE is the mean absolute error scaled by the baseline's: 1.00 ties the
baseline, below 1.00 beats it. Reporting raw MAE lets a comparison be
dropped silently; reporting MASE does not.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def mase(y_true, y_pred, y_baseline) -> float:
    """Mean absolute scaled error against an explicit baseline."""
    baseline_mae = mean_absolute_error(y_true, y_baseline)
    if baseline_mae == 0:
        raise ValueError(
            "baseline has zero error; MASE is undefined. Check the baseline "
            "is not the target itself."
        )
    return float(mean_absolute_error(y_true, y_pred) / baseline_mae)


def classification_report_at_threshold(
    y_true, y_proba, threshold: float
) -> dict[str, float]:
    """Precision/recall at an operating point, plus threshold-free PR-AUC.

    Accuracy is deliberately absent: at a 3% base rate, predicting "never"
    scores 97% and means nothing.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    flagged = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "threshold": float(threshold),
        "n_positive": int(y_true.sum()),
        "n_flagged": int(flagged.sum()),
        "base_rate": float(y_true.mean()),
    }


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of outcomes inside the predicted interval."""
    y_true = np.asarray(y_true)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(inside.mean())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/evaluation/test_metrics.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
.venv/bin/flake8 src/evaluation tests/evaluation
.venv/bin/black --check src/evaluation tests/evaluation
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat(eval): add metrics with MASE scaled to a baseline

MASE makes every number self-interpreting: 1.00 ties persistence, below
1.00 beats it. Accuracy is deliberately absent from the classification
report - at a 3% base rate, predicting 'never' scores 97%."
```

---

## Task 4: Model registry

Declares the candidates and, critically, whether each predicts the **level** or the **change from today**. The §1.1b finding was that target framing and model family have to move together.

**Files:**
- Create: `src/evaluation/registry.py`
- Test: `tests/evaluation/test_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_registry.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.evaluation.registry import CANDIDATES, Candidate, fit_predict


def test_registry_declares_the_measured_winner():
    """Huber on a delta target scored mean MASE 0.899, winning 5/5 folds."""
    assert "huber_delta" in CANDIDATES
    assert CANDIDATES["huber_delta"].target == "delta"


def test_registry_spans_more_than_one_model_family():
    families = {type(c.build()).__name__ for c in CANDIDATES.values()}
    assert len(families) >= 3, f"only {families}"


def test_every_candidate_declares_a_valid_target_framing():
    for name, candidate in CANDIDATES.items():
        assert candidate.target in ("level", "delta"), name


def test_fit_predict_on_a_delta_candidate_recovers_the_level():
    """A delta model must have today's value added back before scoring."""
    rng = np.random.RandomState(0)
    train = pd.DataFrame(
        {"iqa": rng.rand(200) * 10, "lag_1": rng.rand(200) * 10}
    )
    train["target"] = train["iqa"] + 1.0
    test = train.iloc[:20].copy()

    candidate = Candidate(
        name="const_delta",
        target="delta",
        build=lambda: _AlwaysOne(),
    )
    preds = fit_predict(candidate, train, test, ["iqa", "lag_1"])
    assert np.allclose(preds, test["iqa"] + 1.0)


def test_fit_predict_on_a_level_candidate_returns_the_raw_prediction():
    train = pd.DataFrame({"iqa": [1.0, 2.0], "lag_1": [1.0, 2.0], "target": [5.0, 5.0]})
    candidate = Candidate(name="const_level", target="level", build=lambda: _AlwaysOne())
    preds = fit_predict(candidate, train, train, ["iqa", "lag_1"])
    assert np.allclose(preds, 1.0)


class _AlwaysOne:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/evaluation/test_registry.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.registry'`

- [ ] **Step 3: Implement `src/evaluation/registry.py`**

```python
"""Candidate models, each declaring its target framing.

Target framing and model family interact. A delta target with a
RandomForest is worse than the level (MAE 8.511 vs 8.324); the same delta
target with a robust linear model is the best thing measured (mean MASE
0.899). Neither choice can be evaluated alone, so both live here together.

Why tree ensembles lose on this problem: they average over leaves and
shrink toward the training mean, which is the wrong inductive bias for a
near-random-walk series. A linear model on the difference starts from
today's value and learns only the correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VALUE_COLUMN = "iqa"
TARGET_COLUMN = "target"


@dataclass(frozen=True)
class Candidate:
    """One model under test.

    ``target`` is ``"level"`` (predict tomorrow directly) or ``"delta"``
    (predict tomorrow minus today, then add today back).
    """

    name: str
    target: str
    build: Callable[[], object]


CANDIDATES: dict[str, Candidate] = {
    "huber_delta": Candidate(
        name="huber_delta",
        target="delta",
        build=lambda: make_pipeline(
            StandardScaler(), HuberRegressor(max_iter=800, epsilon=1.35)
        ),
    ),
    "ridge_delta": Candidate(
        name="ridge_delta",
        target="delta",
        build=lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    ),
    "ridge_level": Candidate(
        name="ridge_level",
        target="level",
        build=lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    ),
    "hgb_delta": Candidate(
        name="hgb_delta",
        target="delta",
        build=lambda: HistGradientBoostingRegressor(random_state=42, max_iter=300),
    ),
    "rf_level": Candidate(
        name="rf_level",
        target="level",
        build=lambda: RandomForestRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1, random_state=42
        ),
    ),
}


def fit_predict(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """Fit on ``train`` and predict the LEVEL for ``test``.

    Delta candidates are trained on ``target - iqa`` and have today's value
    added back, so every candidate returns a comparable level prediction.
    """
    model = candidate.build()
    if candidate.target == "delta":
        model.fit(train[features], train[TARGET_COLUMN] - train[VALUE_COLUMN])
        return np.asarray(test[VALUE_COLUMN] + model.predict(test[features]))
    model.fit(train[features], train[TARGET_COLUMN])
    return np.asarray(model.predict(test[features]))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/evaluation/test_registry.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
.venv/bin/flake8 src/evaluation tests/evaluation
.venv/bin/black --check src/evaluation tests/evaluation
git add src/evaluation/registry.py tests/evaluation/test_registry.py
git commit -m "feat(eval): add a model registry declaring target framing per candidate

Target framing and model family interact: a delta target hurts a
RandomForest but is the best thing measured with a robust linear model.
Declaring them together is what makes the comparison fair."
```

---

## Task 5: The backtest driver

**Files:**
- Create: `src/evaluation/backtest.py`
- Test: `tests/evaluation/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_backtest.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import add_horizon_target, run_backtest
from src.evaluation.registry import Candidate


def _frame(n_days=500, n_stations=2):
    rng = np.random.RandomState(0)
    rows = []
    for station_id in range(n_stations):
        value = 25.0
        for day in range(n_days):
            value = max(1.0, value + rng.randn() * 3)
            rows.append(
                {
                    "station_id": station_id,
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": value,
                    "lag_1": value,
                }
            )
    return pd.DataFrame(rows)


def test_add_horizon_target_shifts_by_the_requested_days():
    frame = add_horizon_target(_frame(n_days=10, n_stations=1), horizon=2)
    station = frame.sort_values("date_local").reset_index(drop=True)
    assert station["target"].iloc[0] == pytest.approx(station["iqa"].iloc[2])


def test_add_horizon_target_never_crosses_a_station_boundary():
    frame = add_horizon_target(_frame(n_days=5, n_stations=2), horizon=1)
    last_row_of_station_0 = frame[frame["station_id"] == 0].sort_values("date_local").iloc[-1]
    assert pd.isna(last_row_of_station_0["target"])


def test_backtest_returns_one_row_per_fold_candidate_horizon():
    candidates = {
        "always_today": Candidate("always_today", "delta", lambda: _AlwaysZero()),
    }
    out = run_backtest(
        _frame(), features=["lag_1"], candidates=candidates, n_folds=3,
        test_days=60, horizons=(1, 2),
    )
    assert len(out) == 3 * 1 * 2
    assert set(out.columns) >= {"fold", "candidate", "horizon", "mae", "mase"}


def test_a_model_that_predicts_today_ties_persistence():
    """A delta model predicting zero change IS persistence, so MASE == 1."""
    candidates = {"always_today": Candidate("always_today", "delta", lambda: _AlwaysZero())}
    out = run_backtest(
        _frame(), features=["lag_1"], candidates=candidates, n_folds=2,
        test_days=60, horizons=(1,),
    )
    assert out["mase"].round(6).eq(1.0).all(), out[["fold", "mase"]]


def test_backtest_includes_the_baselines():
    out = run_backtest(
        _frame(), features=["lag_1"], candidates={}, n_folds=2, test_days=60,
        horizons=(1,),
    )
    assert set(out["candidate"]) >= {"persistence", "seasonal_naive", "climatology"}


class _AlwaysZero:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/evaluation/test_backtest.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.backtest'`

- [ ] **Step 3: Implement `src/evaluation/backtest.py`**

```python
"""Run every candidate and baseline across folds and horizons.

Produces one tidy row per (fold, candidate, horizon) so the scoreboard can
be grouped any way without re-running anything.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.baselines import climatology, persistence, seasonal_naive
from src.evaluation.metrics import mase, regression_metrics
from src.evaluation.registry import CANDIDATES, Candidate, fit_predict
from src.evaluation.splits import rolling_origin_folds

VALUE_COLUMN = "iqa"
GROUP_COLUMN = "station_id"
DATE_COLUMN = "date_local"


def add_horizon_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach ``target`` = the value ``horizon`` days ahead, per station."""
    out = df.sort_values([GROUP_COLUMN, DATE_COLUMN]).copy()
    out["target"] = out.groupby(GROUP_COLUMN)[VALUE_COLUMN].shift(-horizon)
    return out


def run_backtest(
    gold: pd.DataFrame,
    features: list[str],
    candidates: dict[str, Candidate] | None = None,
    n_folds: int = 5,
    test_days: int = 90,
    horizons: tuple[int, ...] = (1, 2, 3),
    max_date: str | None = None,
) -> pd.DataFrame:
    """Evaluate candidates and baselines; return one row per combination."""
    candidates = CANDIDATES if candidates is None else candidates
    records: list[dict] = []

    for horizon in horizons:
        framed = add_horizon_target(gold, horizon).dropna(subset=["target"])
        needed = [c for c in features if c in framed.columns]
        framed = framed.dropna(subset=needed)

        folds = rolling_origin_folds(
            framed, n_folds=n_folds, test_days=test_days, max_date=max_date
        )
        for fold_number, (train, test) in enumerate(folds, start=1):
            baseline = persistence(test)
            common = {
                "fold": fold_number,
                "horizon": horizon,
                "n_train": len(train),
                "n_test": len(test),
                "cutoff": test[DATE_COLUMN].min().date().isoformat(),
            }

            predictions = {
                "persistence": baseline,
                "seasonal_naive": seasonal_naive(test),
                "climatology": climatology(train, test),
            }
            for name, candidate in candidates.items():
                predictions[name] = fit_predict(candidate, train, test, needed)

            for name, predicted in predictions.items():
                mask = pd.Series(predicted, index=test.index).notna()
                if not mask.any():
                    continue
                y_true = test["target"][mask]
                y_pred = pd.Series(predicted, index=test.index)[mask]
                scores = regression_metrics(y_true, y_pred)
                records.append(
                    {
                        **common,
                        "candidate": name,
                        **scores,
                        "mase": mase(y_true, y_pred, baseline[mask]),
                    }
                )

    return pd.DataFrame(records)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/evaluation/test_backtest.py -v
```
Expected: 5 passed

- [ ] **Step 5: Run it against the REAL gold table and check the known numbers**

```bash
cd /home/ayman/airq_montreal_mlops && .venv/bin/python - <<'EOF' 2>&1 | grep -viE "warning|converg"
import pandas as pd
from src.evaluation.backtest import run_backtest
from src.features.build_features import build_features_daily_iqa

gold = pd.read_parquet("data/gold/daily_station_iqa.parquet")
feat, features = build_features_daily_iqa(gold)
out = run_backtest(feat, features + ["iqa"], n_folds=5, test_days=90,
                   horizons=(1,), max_date="2026-01-18")
summary = out.groupby("candidate")["mase"].agg(["mean", "min", "max", "count"])
print(summary.sort_values("mean").round(3).to_string())
EOF
```

**Sanity gate.** `persistence` must be exactly 1.000. `huber_delta` should land near **0.899** (the pre-measured value; ±0.03 is fine — the feature set differs slightly). If `persistence` is not 1.000, the MASE scaling is wrong: stop and report `STATUS: BLOCKED`.

- [ ] **Step 6: Commit**

```bash
.venv/bin/flake8 src/evaluation tests/evaluation
.venv/bin/black --check src/evaluation tests/evaluation
.venv/bin/python -m pytest
git add src/evaluation/backtest.py tests/evaluation/test_backtest.py
git commit -m "feat(eval): add the rolling-origin backtest driver

One tidy row per (fold, candidate, horizon). Baselines are evaluated in
the same loop as candidates so they cannot drift apart."
```

---

## Task 6: Split-conformal prediction intervals

**Files:**
- Create: `src/evaluation/conformal.py`
- Test: `tests/evaluation/test_conformal.py`

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_conformal.py`:

```python
import numpy as np
import pytest

from src.evaluation.conformal import conformal_quantile, make_interval
from src.evaluation.metrics import interval_coverage


def test_quantile_is_the_empirical_residual_quantile():
    residuals = np.arange(1, 101, dtype=float)  # 1..100
    q = conformal_quantile(residuals, confidence=0.90)
    assert 89.0 <= q <= 92.0, q


def test_interval_is_symmetric_around_the_prediction():
    lower, upper = make_interval(np.array([10.0, 20.0]), q=3.0)
    assert np.allclose(lower, [7.0, 17.0])
    assert np.allclose(upper, [13.0, 23.0])


def test_empirical_coverage_matches_nominal_on_synthetic_data():
    """The whole point of conformal: the interval must mean what it says."""
    rng = np.random.RandomState(0)
    calibration_residuals = np.abs(rng.randn(2000) * 5)
    q = conformal_quantile(calibration_residuals, confidence=0.90)

    truth = rng.randn(4000) * 5 + 50
    predicted = np.full(4000, 50.0)
    lower, upper = make_interval(predicted, q)
    coverage = interval_coverage(truth, lower, upper)
    assert 0.87 <= coverage <= 0.93, coverage


def test_higher_confidence_gives_a_wider_interval():
    residuals = np.abs(np.random.RandomState(1).randn(1000) * 4)
    assert conformal_quantile(residuals, 0.95) > conformal_quantile(residuals, 0.80)


def test_rejects_an_impossible_confidence_level():
    with pytest.raises(ValueError, match="confidence"):
        conformal_quantile(np.array([1.0, 2.0]), confidence=1.5)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/evaluation/test_conformal.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.conformal'`

- [ ] **Step 3: Implement `src/evaluation/conformal.py`**

```python
"""Split-conformal prediction intervals.

Distribution-free: hold out a calibration slice, take the empirical
quantile of absolute residuals, and emit a symmetric interval. Coverage is
guaranteed under exchangeability, which is why empirical coverage is
reported as an acceptance criterion rather than assumed.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(residuals, confidence: float = 0.90) -> float:
    """The conformal width for a given confidence level.

    Uses the finite-sample corrected rank ceil((n+1)*confidence)/n, which is
    what gives the coverage guarantee rather than the plain quantile.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    residuals = np.abs(np.asarray(residuals, dtype=float))
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * confidence) / n)
    return float(np.quantile(residuals, level, method="higher"))


def make_interval(predictions, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval of half-width ``q`` around each prediction."""
    predictions = np.asarray(predictions, dtype=float)
    return predictions - q, predictions + q
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/evaluation/test_conformal.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
.venv/bin/flake8 src/evaluation tests/evaluation
.venv/bin/black --check src/evaluation tests/evaluation
git add src/evaluation/conformal.py tests/evaluation/test_conformal.py
git commit -m "feat(eval): add split-conformal prediction intervals

Uses the finite-sample corrected rank so coverage is guaranteed under
exchangeability rather than approximated."
```

---

## Task 7: Per-station series accessor — removes xfail 2

**Files:**
- Create: `src/models/series.py`
- Modify: `src/models/training_daily.py` (re-export)
- Modify: `tests/test_training_split.py` (remove the marker)
- Test: `tests/test_series.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_series.py`:

```python
import pandas as pd
import pytest

from src.models.series import station_series, station_series_map


def _gold():
    rows = []
    for station_id in (3, 6, 17):
        for day in range(20):
            rows.append(
                {
                    "station_id": station_id,
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": 10 + day + station_id,
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=0)  # deliberately shuffled


def test_station_series_returns_one_contiguous_sorted_series():
    out = station_series(_gold(), station_id=6)
    assert out["station_id"].nunique() == 1
    assert out["date_local"].is_monotonic_increasing
    assert len(out) == 20


def test_station_series_map_covers_every_station():
    out = station_series_map(_gold())
    assert set(out) == {3, 6, 17}
    assert all(len(v) == 20 for v in out.values())


def test_no_series_contains_another_stations_rows():
    """Prophet saw 11 y values per ds and the LSTM slid windows across
    station boundaries because the frame was never split."""
    for station_id, series in station_series_map(_gold()).items():
        assert (series["station_id"] == station_id).all()


def test_unknown_station_raises():
    with pytest.raises(KeyError, match="999"):
        station_series(_gold(), station_id=999)


def test_values_within_a_series_are_in_date_order():
    series = station_series(_gold(), station_id=3)
    assert series["iqa"].tolist() == sorted(series["iqa"].tolist())
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_series.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.series'`

- [ ] **Step 3: Implement `src/models/series.py`**

```python
"""Per-station series access.

Prophet and the LSTM previously consumed a flat frame containing all 11
stations interleaved by date. Prophet then saw 11 ``y`` values per ``ds``,
and the LSTM slid a 30-day window across station boundaries, producing
physically meaningless sequences.
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMN = "station_id"
DATE_COLUMN = "date_local"


def station_series(df: pd.DataFrame, station_id: int) -> pd.DataFrame:
    """One station's rows, sorted by date."""
    subset = df[df[GROUP_COLUMN] == station_id]
    if subset.empty:
        known = sorted(df[GROUP_COLUMN].unique().tolist())
        raise KeyError(f"station {station_id} not present; known stations: {known}")
    return subset.sort_values(DATE_COLUMN).reset_index(drop=True)


def station_series_map(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Every station's series, keyed by station id."""
    return {
        int(station_id): station_series(df, int(station_id))
        for station_id in sorted(df[GROUP_COLUMN].unique())
    }
```

- [ ] **Step 4: Re-export from `training_daily` so the pinned test resolves**

The xfail in `tests/test_training_split.py` asserts `hasattr(td, "_station_series")`. Add to `src/models/training_daily.py`, after the existing imports:

```python
from src.models.series import station_series as _station_series  # noqa: F401
from src.models.series import station_series_map as _station_series_map  # noqa: F401
```

- [ ] **Step 5: Remove the now-satisfied xfail marker**

In `tests/test_training_split.py`, delete the entire `@pytest.mark.xfail(...)` decorator above `test_series_builder_yields_one_series_per_station`, leaving the test itself. It must now pass on its own.

- [ ] **Step 6: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_series.py tests/test_training_split.py -v
```
Expected: `tests/test_series.py` 5 passed; `test_training_split.py` 4 passed, 1 xfailed (only the split defect remains pinned).

If the marker is left in place it will now report **XPASS**, which under `strict=True` is a failure — that is the mechanism working as intended.

- [ ] **Step 7: Commit**

```bash
.venv/bin/flake8 src/models tests/test_series.py tests/test_training_split.py
.venv/bin/black --check src/models tests/test_series.py tests/test_training_split.py
.venv/bin/python -m pytest
git add src/models/series.py src/models/training_daily.py tests/test_series.py tests/test_training_split.py
git commit -m "feat(models): add per-station series accessor, unpin xfail 2

Prophet saw 11 y values per ds and the LSTM slid a 30-day window across
station boundaries. Both now have a way to get one coherent series."
```

---

## Task 8: Put today's IQA back in the feature set — removes xfail 3

**Files:**
- Modify: `src/features/build_features.py`
- Modify: `tests/test_build_features.py` (remove the marker)

- [ ] **Step 1: Remove `iqa` from the non-feature set**

In `src/features/build_features.py`, `_NON_FEATURE_COLUMNS` currently contains `"iqa"`. Remove that one entry. Leave every other exclusion — `station_id`, `date_local`, the `target_*` columns, `latitude`, `longitude`, the `sub_` prefix rule — exactly as they are.

Add a comment above the set:

```python
# 'iqa' is the current day's value and IS a legitimate feature for a
# next-day target - excluding it made this a two-step-ahead model reported
# as one-step. Latitude/longitude stay excluded: they are constant per
# station and let a tree memorise station identity.
```

- [ ] **Step 2: Remove the xfail marker**

In `tests/test_build_features.py`, delete the `@pytest.mark.xfail(...)` decorator above `test_current_day_iqa_is_available_as_a_feature`, keeping the test.

- [ ] **Step 3: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_build_features.py -v
```
Expected: 9 passed, 0 xfailed.

- [ ] **Step 4: Confirm the real feature count went from 19 to 20**

```bash
cd /home/ayman/airq_montreal_mlops && .venv/bin/python -c "
import pandas as pd
from src.features.build_features import build_features_daily_iqa
gold = pd.read_parquet('data/gold/daily_station_iqa.parquet')
frame, features = build_features_daily_iqa(gold)
print('features:', len(features))
print('iqa present:', 'iqa' in features)
print('lat/lon absent:', not {'latitude','longitude'} & set(features))
print('retention: %.1f%%' % (100*len(frame)/len(gold)))
"
```
Expected: `features: 20`, `iqa present: True`, `lat/lon absent: True`, retention ~99%.

- [ ] **Step 5: Re-bake the serving model, since the feature contract changed**

This matters: `artifacts/rf/feature_names.json` currently lists 19 features and the API rejects a request missing any of them. A stale artifact would make `/predict` reject valid requests.

```bash
.venv/bin/python -m scripts.bake_serving_model
.venv/bin/python -c "
import json; print('baked features:', len(json.load(open('artifacts/rf/feature_names.json'))))"
```
Expected: 20.

- [ ] **Step 6: Commit**

```bash
.venv/bin/flake8 src/features tests/test_build_features.py
.venv/bin/black --check src/features tests/test_build_features.py
.venv/bin/python -m pytest
git add src/features/build_features.py tests/test_build_features.py
git commit -m "fix(features): include today's IQA, unpin xfail 3

Excluding the current day's value while targeting tomorrow made this a
two-step-ahead model reported as one-step. Measured effect is small
(MAE 8.324 -> 8.069) but the framing was wrong."
```

---

## Task 9: Exceedance classifier with a cost-based threshold

**Files:**
- Create: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_classifier.py`:

```python
import numpy as np
import pytest

from src.models.classifier import (
    COST_RATIO,
    build_classifier,
    choose_threshold,
    expected_cost,
)


def test_cost_ratio_is_five_to_one():
    """A missed exceedance costs 5x a false alarm - a public-health call,
    stated rather than buried in a default."""
    assert COST_RATIO == 5.0


def test_expected_cost_weights_misses_more_heavily_than_false_alarms():
    y = np.array([0, 0, 1, 1])
    one_miss = np.array([0, 0, 0, 1])
    one_false_alarm = np.array([1, 0, 1, 1])
    assert expected_cost(y, one_miss) > expected_cost(y, one_false_alarm)


def test_choose_threshold_returns_a_probability():
    rng = np.random.RandomState(0)
    y = (rng.rand(500) < 0.1).astype(int)
    proba = np.clip(y * 0.5 + rng.rand(500) * 0.5, 0, 1)
    threshold = choose_threshold(y, proba)
    assert 0.0 < threshold < 1.0


def test_a_higher_cost_ratio_lowers_the_threshold():
    """Penalising misses harder should make the alarm easier to trip."""
    rng = np.random.RandomState(1)
    y = (rng.rand(800) < 0.1).astype(int)
    proba = np.clip(y * 0.4 + rng.rand(800) * 0.6, 0, 1)
    assert choose_threshold(y, proba, cost_ratio=20.0) <= choose_threshold(
        y, proba, cost_ratio=2.0
    )


def test_build_classifier_is_balanced_for_the_rare_positive_class():
    model = build_classifier()
    assert model.get_params()["class_weight"] == "balanced"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_classifier.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.classifier'`

- [ ] **Step 3: Implement `src/models/classifier.py`**

```python
"""Exceedance classifier: will IQA exceed 50 tomorrow?

Base rate is about 3% of station-days, so accuracy is meaningless - a model
predicting "never" scores 97%. Everything here is framed around precision,
recall and an explicit cost trade-off instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# A missed exceedance costs this many times a false alarm. Public-health
# framing: failing to warn a sensitive resident is worse than warning
# unnecessarily. Stated here and in the README, never left implicit.
COST_RATIO = 5.0


def build_classifier(**kwargs) -> RandomForestClassifier:
    """The exceedance classifier, balanced for a rare positive class."""
    params = {
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


def expected_cost(y_true, y_pred, cost_ratio: float = COST_RATIO) -> float:
    """Total cost where a miss costs ``cost_ratio`` and a false alarm 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    misses = int(((y_true == 1) & (y_pred == 0)).sum())
    false_alarms = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(cost_ratio * misses + false_alarms)


def choose_threshold(
    y_true, y_proba, cost_ratio: float = COST_RATIO, n_steps: int = 200
) -> float:
    """The probability threshold minimising expected cost on this data."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    grid = np.linspace(0.01, 0.99, n_steps)
    costs = [
        expected_cost(y_true, (y_proba >= t).astype(int), cost_ratio) for t in grid
    ]
    return float(grid[int(np.argmin(costs))])
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_classifier.py -v
```
Expected: 5 passed

- [ ] **Step 5: Settle the weather question across folds**

A single split suggested weather HURTS the classifier (PR-AUC 0.3798 lags-only vs 0.2587 with weather), contradicting Spec A's premise. Test it properly:

```bash
cd /home/ayman/airq_montreal_mlops && .venv/bin/python - <<'EOF' 2>&1 | grep -viE "warning"
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score
from src.evaluation.splits import rolling_origin_folds
from src.features.build_features import build_features_daily_iqa
from src.models.classifier import build_classifier

gold = pd.read_parquet("data/gold/daily_station_iqa.parquet")
frame, features = build_features_daily_iqa(gold)
frame = frame.dropna(subset=["target_exceed_h24"])
lags = [c for c in features if c.startswith(("lag_", "roll_")) or c == "iqa"]
rows = []
for k, (tr, te) in enumerate(rolling_origin_folds(frame, n_folds=5, test_days=90,
                                                  max_date="2026-01-18"), 1):
    ytr = tr["target_exceed_h24"].astype(int); yte = te["target_exceed_h24"].astype(int)
    if yte.sum() < 5: continue
    row = {"fold": k, "base_rate": round(yte.mean(), 4)}
    for label, cols in (("lags_only", lags), ("lags_plus_weather", features)):
        m = build_classifier().fit(tr[cols], ytr)
        row[label] = round(average_precision_score(yte, m.predict_proba(te[cols])[:, 1]), 4)
    rows.append(row)
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print("\nmean PR-AUC  lags_only: %.4f   lags+weather: %.4f" %
      (out.lags_only.mean(), out.lags_plus_weather.mean()))
print("weather helps in %d of %d folds" % ((out.lags_plus_weather > out.lags_only).sum(), len(out)))
EOF
```

Report the table verbatim. **Whatever it says is the answer** — if weather helps, Spec A's premise stands; if it hurts, it must be retracted in the README. Do not tune until it agrees with either position.

- [ ] **Step 6: Commit**

```bash
.venv/bin/flake8 src/models tests/test_classifier.py
.venv/bin/black --check src/models tests/test_classifier.py
.venv/bin/python -m pytest
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat(models): add exceedance classifier with a 5:1 cost threshold

A missed exceedance costs 5x a false alarm. The ratio is a named constant
and appears in the README, rather than being an unexplained default."
```

---

## Task 10: Rewire training to the new splits — removes xfail 1

**Files:**
- Modify: `src/models/training_daily.py`
- Modify: `tests/test_training_split.py`

- [ ] **Step 1: Replace `_time_split` with a delegation to the new module**

In `src/models/training_daily.py`, replace the existing `_time_split`:

```python
def _time_split(df, ratio=0.2):
    n = len(df)
    k = int(n * (1 - ratio))
    return df.iloc[:k], df.iloc[k:]
```

with:

```python
def _time_split(df, ratio=0.2):
    """Split on a date boundary.

    The previous implementation sliced by row index on a frame sorted by
    [station_id, date_local], which produced a station holdout and inflated
    reported validation scores by roughly 43%.
    """
    from src.evaluation.splits import time_split

    return time_split(df, test_fraction=ratio)
```

- [ ] **Step 2: Remove the last xfail marker**

In `tests/test_training_split.py`, delete the `@pytest.mark.xfail(...)` decorator above `test_time_split_holdout_starts_after_train_ends`, keeping the test.

- [ ] **Step 3: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_training_split.py -v
```
Expected: 5 passed, 0 xfailed. **All three pinned defects are now fixed.**

- [ ] **Step 4: Confirm the split really is a time split now**

```bash
cd /home/ayman/airq_montreal_mlops && .venv/bin/python - <<'EOF'
from src.models.training_daily import _daily_df, _time_split
df = _daily_df()
tr, va = _time_split(df, ratio=0.2)
print("train dates :", tr.date_local.min().date(), "->", tr.date_local.max().date())
print("valid dates :", va.date_local.min().date(), "->", va.date_local.max().date())
print("train stations:", len(set(tr.station_id)), "| valid stations:", len(set(va.station_id)))
print("OVERLAP in time?", tr.date_local.max() >= va.date_local.min())
EOF
```
Expected: train dates strictly before valid dates, **both sides holding all 11 stations**, and `OVERLAP in time? False`. Before this task, both folds spanned the identical full date range while the station sets barely overlapped.

- [ ] **Step 5: Re-run `train_rf` and record the honest number**

```bash
cd /home/ayman/airq_montreal_mlops && timeout 900 .venv/bin/python -c "
from src.models.training_daily import train_rf
_, feats, mets = train_rf()
print('features:', len(feats))
print({k: round(v, 3) for k, v in mets.items()})
"
```

Report the metrics. Expect `mae_va` to get **worse** than the previously reported 4.579 — that number was inflated by the station split. A jump to roughly 7–8 is the correct, honest result, not a regression.

- [ ] **Step 6: Commit**

```bash
.venv/bin/flake8 src/models tests/test_training_split.py
.venv/bin/black --check src/models tests/test_training_split.py
.venv/bin/python -m pytest
git add src/models/training_daily.py tests/test_training_split.py
git commit -m "fix(models): split on dates, unpin xfail 1 - all three defects fixed

_time_split sliced by row index on a frame sorted by [station_id,
date_local], so train and validation spanned the identical date range
while the station sets barely overlapped. The reported mae_va will now
rise to its honest value."
```

---

## Task 11: The scoreboard — `scripts/run_backtest.py`

**Files:**
- Create: `scripts/run_backtest.py`
- Create: `docs/RESULTS.md` (generated)
- Test: `tests/test_run_backtest_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_run_backtest_smoke.py`:

```python
import subprocess
import sys


def test_script_exposes_help():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_backtest", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--out" in result.stdout
    assert "--folds" in result.stdout


def test_render_markdown_produces_a_table():
    import pandas as pd

    from scripts.run_backtest import render_markdown

    frame = pd.DataFrame(
        {
            "candidate": ["persistence", "huber_delta"],
            "horizon": [1, 1],
            "mase": [1.0, 0.9],
            "mae": [7.0, 6.3],
            "fold": [1, 1],
        }
    )
    out = render_markdown(frame, cost_ratio=5.0, classifier_rows=None)
    assert "huber_delta" in out
    assert "MASE" in out
    assert "|" in out
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_run_backtest_smoke.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_backtest'`

- [ ] **Step 3: Implement `scripts/run_backtest.py`**

```python
"""Regenerate docs/RESULTS.md from scratch.

Every number published in the README comes from this script, so a reader
can reproduce the scoreboard with one command.

    .venv/bin/python -m scripts.run_backtest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import run_backtest
from src.features.build_features import build_features_daily_iqa
from src.models.classifier import COST_RATIO
from src.models.training_daily import GOLD_PATH

CONTIGUOUS_END = "2026-01-18"
DEFAULT_OUT = Path("docs/RESULTS.md")


def render_markdown(
    results: pd.DataFrame, cost_ratio: float, classifier_rows: pd.DataFrame | None
) -> str:
    """Render the scoreboard as Markdown."""
    lines = [
        "# Results",
        "",
        "Generated by `python -m scripts.run_backtest`. Do not edit by hand.",
        "",
        "MASE is scaled to persistence: **1.00 ties it, below 1.00 beats it**.",
        "",
        "## Regression — mean MASE across folds",
        "",
    ]
    pivot = (
        results.groupby(["candidate", "horizon"])["mase"].mean().unstack("horizon")
    )
    pivot = pivot.sort_values(pivot.columns[0])
    header = " | ".join(f"h{int(c)*24}" for c in pivot.columns)
    lines += [f"| model | {header} |", "|---|" + "---|" * len(pivot.columns)]
    for name, row in pivot.iterrows():
        cells = " | ".join(f"{v:.3f}" for v in row)
        lines.append(f"| `{name}` | {cells} |")

    wins = results[results.horizon == results.horizon.min()]
    lines += ["", "## Folds won against persistence (h24)", ""]
    lines += ["| model | folds beaten | of |", "|---|---|---|"]
    for name, group in wins.groupby("candidate"):
        lines.append(f"| `{name}` | {int((group.mase < 1).sum())} | {len(group)} |")

    if classifier_rows is not None and not classifier_rows.empty:
        lines += [
            "",
            "## Exceedance alert (IQA > 50 tomorrow)",
            "",
            f"Operating threshold minimises expected cost at a **{cost_ratio:.0f}:1** "
            "ratio (a missed exceedance costs five times a false alarm).",
            "",
            classifier_rows.to_markdown(index=False),
        ]

    lines += ["", "---", "", f"Data through {CONTIGUOUS_END}.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser("regenerate the results scoreboard")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=90)
    args = parser.parse_args()

    gold = pd.read_parquet(GOLD_PATH)
    frame, features = build_features_daily_iqa(gold)

    results = run_backtest(
        frame,
        features,
        n_folds=args.folds,
        test_days=args.test_days,
        horizons=(1, 2, 3),
        max_date=CONTIGUOUS_END,
    )

    summary = results.groupby("candidate")["mase"].mean().sort_values()
    print(summary.round(3).to_string())
    best = summary.index[0]
    print(f"\nbest: {best} (mean MASE {summary.iloc[0]:.3f})")
    if summary.iloc[0] >= 1.0:
        print("NOTE: nothing beat persistence. Say so plainly in the README.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(results, COST_RATIO, None))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests, then generate the scoreboard**

```bash
.venv/bin/python -m pytest tests/test_run_backtest_smoke.py -v
.venv/bin/python -m scripts.run_backtest
cat docs/RESULTS.md
```

Report `docs/RESULTS.md` verbatim.

**Sanity gate:** `persistence` must show MASE exactly 1.000 at every horizon. If not, the scaling is broken — report `STATUS: BLOCKED`.

- [ ] **Step 5: Commit**

```bash
.venv/bin/flake8 scripts/run_backtest.py tests/test_run_backtest_smoke.py
.venv/bin/black --check scripts/run_backtest.py tests/test_run_backtest_smoke.py
.venv/bin/python -m pytest
git add scripts/run_backtest.py tests/test_run_backtest_smoke.py docs/RESULTS.md
git commit -m "feat(eval): add the scoreboard generator and publish docs/RESULTS.md

Every published number is reproducible with one command."
```

---

## Task 12: README results and the weather retraction

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the Results section**

Replace the whole "Results" section (which currently states no accuracy is claimed and shows only the persistence comparison) with:

- a one-line verdict naming the best model and its mean MASE
- the h24/h48/h72 MASE table from `docs/RESULTS.md`
- how many folds each model beat persistence in
- the exceedance section: PR-AUC, the 5:1 cost ratio stated in prose, precision/recall at the chosen threshold, and persistence's classifier score for comparison
- conformal coverage against nominal 90%
- a link to `docs/RESULTS.md` and the command that regenerates it

**Use the real numbers produced in Tasks 9 and 11.** Do not copy the illustrative figures from this plan.

- [ ] **Step 2: Record the weather finding honestly**

Add a short subsection under Results titled **"Does weather help?"**, reporting the Task 9 Step 5 table.

If weather does NOT help, the README must say so explicitly and note that it **contradicts Spec A's stated premise** that weather would be the project's biggest modelling unlock. A project that publishes a result contradicting its own design premise is more credible than one that quietly drops it.

If weather DOES help, say that too, with the fold-level numbers.

- [ ] **Step 3: Update the Roadmap**

Mark Spec B complete. Note that hyperparameter search and per-station Prophet/LSTM retraining remain deferred.

- [ ] **Step 4: Verify no stale claims remain**

```bash
cd /home/ayman/airq_montreal_mlops
grep -n "4.579\|no accuracy claims\|makes no accuracy" README.md || echo "clean: no stale accuracy claims"
grep -n "xfail" README.md || echo "clean: no stale xfail references"
```
Both should report clean, since all three defects are now fixed and real numbers exist.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/python -m pytest
git add README.md
git commit -m "docs: publish Spec B results

Replaces the no-accuracy-claimed placeholder with the real scoreboard,
and records the weather result whichever way it fell."
git push
```

---

## Final verification

- [ ] **All three xfails are gone**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
```
Expected: all passed, **0 xfailed**.

- [ ] **The scoreboard regenerates from scratch**

```bash
rm docs/RESULTS.md && .venv/bin/python -m scripts.run_backtest && head -20 docs/RESULTS.md
```

- [ ] **Tests still run with no data**

```bash
mv data /tmp/_parked && .venv/bin/python -m pytest -q 2>&1 | tail -3 ; mv /tmp/_parked data
```
Expected: all pass. Restore `data/` even if the run fails.

- [ ] **CI is green**

```bash
git push
```
Then check https://github.com/AyDaoud/airq_montreal_mlops/actions

---

## Plan self-review

Checked against the spec on 2026-09-18:

| Spec section | Covered by |
|---|---|
| §1.1 persistence loses | Tasks 2, 5 (baselines + backtest) |
| §1.1b Huber-on-Δ wins | Task 4 (registry), Task 5 step 5 (sanity gate) |
| §1.2 exceedance signal | Task 9 |
| §1.3 weather may hurt | Task 9 step 5, Task 12 step 2 |
| §3 architecture | Tasks 1–6, 9, 11 |
| §4 evaluation protocol | Tasks 1, 3, 5 |
| §5 alert product | Task 9, Task 12 |
| §6 conformal | Task 6 |
| §7 three xfails | Tasks 7, 8, 10 |
| §8 decisions B1–B8 | B1 Task 3; B2/B3 Task 5; B4 Task 9; B5 Task 7; B6 Task 6; B7 Tasks 1/5; B8 Task 9 |
| §10 acceptance 1–8 | Final verification |

**Known gap, accepted:** conformal intervals are implemented and unit-tested (Task 6) but are not wired into the scoreboard, because `run_backtest` reports point metrics only. Acceptance criterion 4 (coverage within ±3pp) is therefore verified by the Task 6 unit test on synthetic data rather than on the real folds. Wiring conformal into the backtest is the first item of any Spec B follow-up.

**Ordering note:** Task 8 changes the feature contract from 19 to 20 columns, so Task 8 step 5 re-bakes the serving artifact. Any task after 8 that touches serving must use the 20-feature artifact.
