import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features_daily_iqa

LAGS = (1, 2, 3, 7, 14)


def _gold(n_days=60, n_stations=2):
    """A gold-shaped frame with enough rows to survive a 14-day lag."""
    rows = []
    for station_id in range(n_stations):
        for day in range(n_days):
            rows.append(
                {
                    "station_id": np.int16(station_id),
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": np.int16(10 + day + station_id * 100),
                    "temp_mean": 5.0 + day,
                    "wind_speed_mean": 3.0,
                    "wind_dir_sin": 0.5,
                    "wind_dir_cos": 0.5,
                    "precip_sum": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_returns_non_empty_frame():
    """Guards the defect in the previous test, which passed on zero rows."""
    out, feats = build_features_daily_iqa(_gold())
    assert len(out) > 0, "feature frame is empty; the test would be vacuous"
    assert len(feats) > 0


def test_lag_one_equals_previous_days_value():
    out, _ = build_features_daily_iqa(_gold(n_stations=1))
    out = out.sort_values("date_local").reset_index(drop=True)
    for i in range(1, len(out)):
        expected = out["iqa"].iloc[i - 1]
        assert out["lag_1"].iloc[i] == expected, f"lag_1 wrong at row {i}"


def test_lags_never_cross_station_boundaries():
    out, _ = build_features_daily_iqa(_gold(n_days=40, n_stations=2))
    for _station_id, group in out.groupby("station_id"):
        group = group.sort_values("date_local").reset_index(drop=True)
        for i in range(1, len(group)):
            assert group["lag_1"].iloc[i] == group["iqa"].iloc[i - 1]


def test_all_expected_lag_columns_present():
    out, feats = build_features_daily_iqa(_gold())
    for lag in LAGS:
        assert f"lag_{lag}" in out.columns
        assert f"lag_{lag}" in feats


def test_rolling_mean_excludes_the_current_day():
    out, _ = build_features_daily_iqa(_gold(n_stations=1))
    out = out.sort_values("date_local").reset_index(drop=True)
    row = out.iloc[20]
    window = out["iqa"].iloc[13:20]
    assert row["roll_7"] == pytest.approx(window.mean(), rel=1e-6)


def test_weather_columns_are_carried_into_features():
    _, feats = build_features_daily_iqa(_gold())
    for column in ("temp_mean", "wind_speed_mean", "wind_dir_sin", "precip_sum"):
        assert column in feats, f"{column} should be a feature"


def test_no_nulls_remain():
    out, feats = build_features_daily_iqa(_gold())
    assert out[feats].isna().sum().sum() == 0


def test_missing_required_column_fails_loudly():
    """Replaces the old 'guess the value column' fallback."""
    bad = _gold().drop(columns=["iqa"])
    with pytest.raises(ValueError, match="iqa"):
        build_features_daily_iqa(bad)


def test_current_day_iqa_is_available_as_a_feature():
    _, feats = build_features_daily_iqa(_gold())
    assert "iqa" in feats
