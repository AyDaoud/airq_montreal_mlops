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
    sizes = [
        len(tr) for tr, _ in rolling_origin_folds(_frame(), n_folds=4, test_days=30)
    ]
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
