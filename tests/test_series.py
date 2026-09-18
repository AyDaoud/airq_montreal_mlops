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
    for station_id, series in station_series_map(_gold()).items():
        assert (series["station_id"] == station_id).all()


def test_unknown_station_raises():
    with pytest.raises(KeyError, match="999"):
        station_series(_gold(), station_id=999)


def test_values_within_a_series_are_in_date_order():
    series = station_series(_gold(), station_id=3)
    assert series["iqa"].tolist() == sorted(series["iqa"].tolist())
