import pandas as pd
from src.features.build_features import build_features


def test_build_features_basic():
    # Minimal fake RSQA + weather data
    pr = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=10, freq="H", tz="UTC"),
            "station_id": [1] * 10,
            "pollutant": ["NO2"] * 10,
            "value": range(10),
        }
    )

    pw = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=10, freq="H", tz="UTC"),
            "temp": [0.0] * 10,
        }
    )

    df, feats = build_features(pr, pw, pollutant="NO2")

    # basic sanity checks
    assert "target" in df.columns
    for lag in (1, 2, 3, 6, 12, 24):
        assert f"lag_{lag}" in df.columns
    assert all(c in df.columns for c in feats)
    assert df.isna().sum().sum() == 0
