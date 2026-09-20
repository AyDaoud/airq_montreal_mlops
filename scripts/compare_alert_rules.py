"""Compare alerting rules against the trivial persistence rule.

Spec B's regression result is a win: Huber beats persistence by 10-24%.
The alert task is the opposite, and this script is the evidence. Every
method tried loses to "today was already above 50" at every horizon.

Run:  .venv/bin/python -m scripts.compare_alert_rules
"""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.backtest import add_horizon_target
from src.evaluation.splits import rolling_origin_folds
from src.features.build_features import build_features_daily_iqa
from src.models.classifier import (
    COST_RATIO,
    build_classifier,
    choose_threshold,
    expected_cost,
)
from src.models.training_daily import GOLD_PATH

CONTIGUOUS_END = "2026-01-18"
THRESHOLD = 50
MIN_POSITIVES = 20


def _non_weather(features: list[str]) -> list[str]:
    """Weather is excluded: it does not help this task (see README)."""
    return [
        c
        for c in features
        if not c.startswith(("temp_", "humidity_", "precip_", "wind_", "pressure_"))
    ]


def evaluate(
    horizon: int, frame: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """Total expected cost per rule, summed over folds, at COST_RATIO."""
    cols = _non_weather(features)
    framed = add_horizon_target(frame, horizon)
    framed["exceed"] = framed["target"] > THRESHOLD
    framed = framed.dropna(subset=["target"] + cols)

    totals = {
        "persistence": 0.0,
        "classifier_fixed_threshold": 0.0,
        "classifier_calibrated": 0.0,
        "huber_regression_thresholded": 0.0,
    }
    folds = 0
    for train, test in rolling_origin_folds(
        framed, n_folds=5, test_days=90, max_date=CONTIGUOUS_END
    ):
        y_true = test["exceed"].astype(int).values
        if y_true.sum() < MIN_POSITIVES:
            continue
        folds += 1
        y_train = train["exceed"].astype(int)

        totals["persistence"] += expected_cost(
            y_true, (test["iqa"] > THRESHOLD).astype(int).values, COST_RATIO
        )

        clf = build_classifier().fit(train[cols], y_train)
        totals["classifier_fixed_threshold"] += expected_cost(
            y_true, (clf.predict_proba(test[cols])[:, 1] >= 0.5).astype(int), COST_RATIO
        )

        # Hold out the last fifth of training BY DATE to pick the threshold,
        # since a threshold fitted on training predictions does not transfer:
        # class_weight="balanced" inflates them and the cut ends up far too
        # conservative at test time.
        cut = train["date_local"].quantile(0.8)
        fit, calib = train[train["date_local"] < cut], train[train["date_local"] >= cut]
        base = build_classifier().fit(fit[cols], fit["exceed"].astype(int))
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit").fit(
            calib[cols], calib["exceed"].astype(int)
        )
        chosen = choose_threshold(
            calib["exceed"].astype(int).values,
            calibrated.predict_proba(calib[cols])[:, 1],
            COST_RATIO,
        )
        totals["classifier_calibrated"] += expected_cost(
            y_true,
            (calibrated.predict_proba(test[cols])[:, 1] >= chosen).astype(int),
            COST_RATIO,
        )

        # Derive the alert from the regression that DOES beat persistence.
        huber = make_pipeline(StandardScaler(), HuberRegressor(max_iter=800)).fit(
            train[cols], train["target"]
        )
        totals["huber_regression_thresholded"] += expected_cost(
            y_true, (huber.predict(test[cols]) > THRESHOLD).astype(int), COST_RATIO
        )

    totals["folds"] = folds
    return totals


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    frame, features = build_features_daily_iqa(gold)
    frame = frame.dropna(subset=["target_exceed_h24"])

    print(f"Total expected cost at {COST_RATIO:.0f}:1 (missed alert : false alarm).")
    print("Lower is better.\n")
    for horizon in (1, 2, 3):
        totals = evaluate(horizon, frame, features)
        folds = totals.pop("folds")
        best = min(totals, key=totals.get)
        cells = "   ".join(f"{k}={v:.0f}" for k, v in totals.items())
        print(f"h{horizon * 24} ({folds} folds)  {cells}")
        print(f"      best: {best}\n")


if __name__ == "__main__":
    main()
