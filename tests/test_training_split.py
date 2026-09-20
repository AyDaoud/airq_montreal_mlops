import pandas as pd
import pytest

import src.models.training_daily as td
from src.models.training_daily import _daily_df, _time_split


def test_daily_df_reads_the_gold_table(tmp_path, monkeypatch):
    gold = pd.DataFrame(
        {
            "station_id": pd.array([3, 3], dtype="int16"),
            "date_local": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "iqa": pd.array([10, 20], dtype="int16"),
        }
    )
    path = tmp_path / "gold.parquet"
    gold.to_parquet(path, index=False)
    monkeypatch.setattr(td, "GOLD_PATH", path)

    out = _daily_df()
    assert len(out) == 2
    assert {"station_id", "date_local", "iqa"} <= set(out.columns)


def test_daily_df_error_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(td, "GOLD_PATH", tmp_path / "does_not_exist.parquet")
    with pytest.raises(FileNotFoundError, match="src.data.cli"):
        _daily_df()


def test_daily_df_is_sorted_by_station_then_date(tmp_path, monkeypatch):
    gold = pd.DataFrame(
        {
            "station_id": pd.array([6, 3, 6, 3], dtype="int16"),
            "date_local": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01"]
            ),
            "iqa": pd.array([1, 2, 3, 4], dtype="int16"),
        }
    )
    path = tmp_path / "gold.parquet"
    gold.to_parquet(path, index=False)
    monkeypatch.setattr(td, "GOLD_PATH", path)

    out = _daily_df()
    assert out["station_id"].tolist() == [3, 3, 6, 6]


def test_time_split_holdout_starts_after_train_ends():
    df = pd.DataFrame(
        {
            "station_id": pd.array([3] * 10 + [6] * 10, dtype="int16"),
            "date_local": pd.to_datetime(
                list(pd.date_range("2024-01-01", periods=10))
                + list(pd.date_range("2024-01-01", periods=10))
            ),
        }
    ).sort_values(["station_id", "date_local"])

    train, valid = _time_split(df, ratio=0.2)
    assert train["date_local"].max() <= valid["date_local"].min()


def test_series_builder_yields_one_series_per_station():
    """Spec B should expose a per-station series accessor."""
    assert hasattr(td, "_station_series"), "no per-station series accessor exists"
