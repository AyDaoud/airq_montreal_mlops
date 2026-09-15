import pandas as pd

from src.data.aggregate import build_silver_iqa


def _frame(source, value, ts="2026-01-05 12:00"):
    return pd.DataFrame(
        {
            "station_id": pd.array([3], dtype="int16"),
            "ts_utc": pd.to_datetime([ts], utc=True),
            "ts_local": pd.to_datetime([ts], utc=True).tz_convert("America/Montreal"),
            "pollutant": ["PM"],
            "value": pd.array([value], dtype="int16"),
            "source": [source],
        }
    )


def test_historical_wins_over_realtime_on_conflict():
    """The annual dump is the corrected official record."""
    out = build_silver_iqa([_frame("realtime", 99), _frame("historical", 42)])
    assert len(out) == 1
    assert out["value"].iloc[0] == 42
    assert out["source"].iloc[0] == "historical"


def test_non_overlapping_rows_are_all_kept():
    out = build_silver_iqa(
        [
            _frame("historical", 10, "2026-01-05 12:00"),
            _frame("realtime", 20, "2026-01-06 12:00"),
        ]
    )
    assert len(out) == 2
    assert sorted(out["value"].tolist()) == [10, 20]


def test_output_is_sorted_and_unique():
    out = build_silver_iqa(
        [
            _frame("realtime", 20, "2026-01-06 12:00"),
            _frame("historical", 10, "2026-01-05 12:00"),
        ]
    )
    assert out["ts_utc"].is_monotonic_increasing
    assert not out.duplicated(subset=["station_id", "ts_utc", "pollutant"]).any()


def test_empty_input_returns_empty_frame():
    out = build_silver_iqa([])
    assert out.empty
    assert "station_id" in out.columns
