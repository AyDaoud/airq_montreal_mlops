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
    # test.iloc[0] is station 3, 2024-01-01 (month=1). The fixture spans
    # 40 days from 2024-01-01, so it crosses into February for days 31-39;
    # the station-only mean therefore differs from the station-month mean,
    # and it is the latter climatology must return.
    expected = train.query("station_id == 3 and date_local.dt.month == 1")["iqa"].mean()
    assert out.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_climatology_falls_back_for_an_unseen_station_month():
    train = _frame()
    test = _frame().assign(
        date_local=lambda d: d["date_local"] + pd.DateOffset(months=6)
    )
    out = climatology(train, test)
    assert out.notna().all(), "unseen station-month produced NaN"
    # Station 3 is present in train (just not for this month), so the
    # correct fallback tier is the station mean, not the global mean -
    # the global mean is reserved for a station absent from train entirely.
    expected = train.query("station_id == 3")["iqa"].mean()
    assert out.iloc[0] == pytest.approx(expected, rel=1e-6)
