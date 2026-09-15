import numpy as np
import pandas as pd
import pytest

from src.data.aggregate import add_targets, daily_weather, hourly_to_daily


def _hourly(rows):
    """rows: list of (station_id, 'YYYY-MM-DD HH:MM' local, pollutant, value)."""
    local = pd.to_datetime([r[1] for r in rows]).tz_localize(
        "America/Montreal", ambiguous=True, nonexistent="shift_forward"
    )
    return pd.DataFrame(
        {
            "station_id": pd.array([r[0] for r in rows], dtype="int16"),
            "ts_utc": local.tz_convert("UTC"),
            "ts_local": local,
            "pollutant": [r[2] for r in rows],
            "value": pd.array([r[3] for r in rows], dtype="int16"),
            "source": "historical",
        }
    )


def test_daily_iqa_is_the_max_across_pollutants_and_hours():
    df = _hourly(
        [
            (3, "2024-06-01 01:00", "PM", 10),
            (3, "2024-06-01 01:00", "O3", 30),
            (3, "2024-06-01 14:00", "PM", 55),
            (3, "2024-06-01 14:00", "O3", 20),
        ]
    )
    out = hourly_to_daily(df)
    assert len(out) == 1
    assert out["iqa"].iloc[0] == 55
    assert out["driving_pollutant"].iloc[0] == "PM"
    assert out["n_hours_observed"].iloc[0] == 2


def test_per_pollutant_subindices_are_daily_maxima():
    df = _hourly(
        [
            (3, "2024-06-01 01:00", "PM", 10),
            (3, "2024-06-01 14:00", "PM", 40),
            (3, "2024-06-01 14:00", "NO2", 7),
        ]
    )
    out = hourly_to_daily(df)
    assert out["sub_PM"].iloc[0] == 40
    assert out["sub_NO2"].iloc[0] == 7
    assert pd.isna(out["sub_O3"].iloc[0])


def test_targets_shift_by_one_and_two_days():
    df = pd.DataFrame(
        {
            "station_id": pd.array([3, 3, 3], dtype="int16"),
            "date_local": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
            "iqa": pd.array([10, 20, 60], dtype="int16"),
        }
    )
    out = add_targets(df)
    assert out["target_iqa_h24"].tolist()[:2] == [20, 60]
    assert out["target_iqa_h48"].iloc[0] == 60
    assert out["target_exceed_h24"].tolist()[:2] == [False, True]


def test_targets_never_cross_a_station_boundary():
    df = pd.DataFrame(
        {
            "station_id": pd.array([3, 6], dtype="int16"),
            "date_local": pd.to_datetime(["2024-06-01", "2024-06-02"]),
            "iqa": pd.array([10, 99], dtype="int16"),
        }
    )
    out = add_targets(df)
    station_3 = out[out["station_id"] == 3]
    assert pd.isna(
        station_3["target_iqa_h24"].iloc[0]
    ), "station 3 borrowed station 6's value"


def test_targets_respect_calendar_gaps():
    """A missing day must not make 'tomorrow' mean three days later."""
    df = pd.DataFrame(
        {
            "station_id": pd.array([3, 3], dtype="int16"),
            "date_local": pd.to_datetime(["2024-06-01", "2024-06-04"]),
            "iqa": pd.array([10, 99], dtype="int16"),
        }
    )
    out = add_targets(df)
    assert pd.isna(out["target_iqa_h24"].iloc[0])


def test_daily_weather_encodes_wind_direction_as_unit_vector():
    # 04:00 UTC == 00:00 EDT, so these 24 hours are exactly one Montreal day.
    ts = pd.date_range("2024-06-01 04:00", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "cell_id": "45.5_-73.6",
            "ts_utc": ts,
            "temperature_2m": np.linspace(10, 20, 24),
            "relative_humidity_2m": 50.0,
            "precipitation": 0.5,
            "wind_speed_10m": 10.0,
            "wind_direction_10m": 90.0,  # due east
            "surface_pressure": 1000.0,
        }
    )
    out = daily_weather(weather)
    row = out.iloc[0]
    assert row["wind_dir_sin"] == pytest.approx(1.0, abs=1e-6)
    assert row["wind_dir_cos"] == pytest.approx(0.0, abs=1e-6)
    assert row["precip_sum"] == pytest.approx(12.0)
    assert row["wind_speed_mean"] == pytest.approx(10.0)


def test_weather_and_iqa_bucket_on_the_same_calendar_day():
    """Regression: weather was bucketed on the UTC date while IQA used the
    Montreal civil date, offsetting every weather feature by 4-5 hours."""
    ts = pd.date_range("2024-07-15 04:00", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "cell_id": "45.5_-73.6",
            "ts_utc": ts,
            "temperature_2m": 20.0,
            "relative_humidity_2m": 50.0,
            "precipitation": 1.0,
            "wind_speed_10m": 5.0,
            "wind_direction_10m": 180.0,
            "surface_pressure": 1000.0,
        }
    )
    out = daily_weather(weather)
    assert len(out) == 1, "24 hours of one Montreal day must collapse to one row"
    assert out["date_local"].iloc[0] == pd.Timestamp("2024-07-15")
    assert out["precip_sum"].iloc[0] == pytest.approx(24.0)

    iqa = _hourly([(3, f"2024-07-15 {h:02d}:00", "PM", 10) for h in range(24)])
    daily = hourly_to_daily(iqa)
    assert daily["date_local"].iloc[0] == out["date_local"].iloc[0]
