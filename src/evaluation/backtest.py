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
                series = pd.Series(predicted, index=test.index)
                mask = series.notna()
                if not mask.any():
                    continue
                y_true = test["target"][mask]
                y_pred = series[mask]
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
