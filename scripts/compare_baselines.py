"""Compare the trained model against do-nothing baselines on an honest time split.

Spec A makes no accuracy claims. This script is the evidence for that position:
it shows the current model does not beat "tomorrow = today".

Run:  .venv/bin/python -m scripts.compare_baselines
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.build_features import build_features_daily_iqa
from src.models.training_daily import GOLD_PATH

# The realtime feed contributes isolated recent days separated from the
# historical block by a long gap; restrict to the contiguous period.
CONTIGUOUS_END = "2026-01-18"
TEST_FRACTION = 0.2


def _score(name: str, y_true, y_pred) -> float:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"  {name:38s} MAE {mae:6.3f}   RMSE {rmse:6.3f}")
    return mae


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    gold = gold[gold["date_local"] <= CONTIGUOUS_END].copy()
    gold = gold.sort_values(["station_id", "date_local"])

    # A TIME split: the cut is a date, not a row index. Slicing by row on a
    # frame sorted by [station_id, date_local] yields a station split instead.
    cut = gold["date_local"].quantile(1 - TEST_FRACTION)
    print(f"Time split at {cut.date()}  (train < cut, test >= cut)\n")

    gold["persistence"] = gold.groupby("station_id")["iqa"].shift(0)
    gold["seasonal_naive"] = gold.groupby("station_id")["iqa"].shift(6)
    test = gold[(gold["date_local"] >= cut) & gold["target_iqa_h24"].notna()]

    print("=== Baselines ===")
    persistence_mae = _score(
        "persistence (tomorrow = today)",
        test["target_iqa_h24"],
        test["persistence"],
    )
    seasonal = test[test["seasonal_naive"].notna()]
    _score(
        "seasonal-naive (same weekday)",
        seasonal["target_iqa_h24"],
        seasonal["seasonal_naive"],
    )

    frame, features = build_features_daily_iqa(gold)
    train = frame[frame["date_local"] < cut]
    holdout = frame[frame["date_local"] >= cut]

    print("\n=== Models (same split) ===")
    results = {}
    for label, cols in (
        ("RF as built (no today's IQA)", features),
        ("RF + today's IQA", features + ["iqa"]),
    ):
        model = RandomForestRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1, random_state=42
        )
        model.fit(train[cols], train["target"])
        results[label] = _score(label, holdout["target"], model.predict(holdout[cols]))

    print("\n=== Verdict ===")
    for label, mae in results.items():
        verb = "beats" if mae < persistence_mae else "LOSES TO"
        print(
            f"  {label:32s} {mae:6.3f}  {verb} persistence "
            f"({persistence_mae:.3f}) by {abs(persistence_mae - mae):.3f}"
        )


if __name__ == "__main__":
    main()
