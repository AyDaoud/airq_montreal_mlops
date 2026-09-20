import pandas as pd
import pytest

from src.data.rsqa_ingest import normalize_iqa, to_bronze

FIXTURES = "tests/fixtures"


def _historical():
    return pd.read_csv(f"{FIXTURES}/iqa_historical_sample.csv")


def test_normalize_historical_produces_canonical_columns():
    out = normalize_iqa(_historical(), source="historical")
    assert list(out.columns) == [
        "station_id",
        "ts_utc",
        "ts_local",
        "pollutant",
        "value",
        "source",
    ]
    assert str(out["ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert (out["source"] == "historical").all()


def test_spring_forward_hour_two_survives_and_maps_to_edt():
    """02:00 EST on 2024-03-10 is a real observation and must not be dropped.

    Localizing directly to America/Montreal would raise NonExistentTimeError.
    02:00 EST == 07:00 UTC == 03:00 EDT.
    """
    out = normalize_iqa(_historical(), source="historical")
    row = out[
        (out["station_id"] == 3)
        & (out["pollutant"] == "O3")
        & (out["ts_utc"] == pd.Timestamp("2024-03-10 07:00", tz="UTC"))
    ]
    assert len(row) == 1
    assert row["ts_local"].dt.hour.iloc[0] == 3


def test_hour_before_transition_stays_in_est():
    out = normalize_iqa(_historical(), source="historical")
    row = out[
        (out["station_id"] == 3)
        & (out["pollutant"] == "O3")
        & (out["ts_utc"] == pd.Timestamp("2024-03-10 06:00", tz="UTC"))
    ]
    assert len(row) == 1
    assert row["ts_local"].dt.hour.iloc[0] == 1


def test_realtime_english_pollutant_column_is_accepted():
    rt = pd.read_csv(f"{FIXTURES}/iqa_realtime_sample.csv")
    out = normalize_iqa(rt, source="realtime")
    assert "pollutant" in out.columns
    assert (out["source"] == "realtime").all()
    assert len(out) == len(rt)


def test_normalize_rejects_frame_with_neither_spelling():
    df = pd.DataFrame(
        {"stationId": [3], "valeur": [1], "date": ["2024-01-01"], "heure": [0]}
    )
    with pytest.raises(ValueError, match="pollutant"):
        normalize_iqa(df, source="historical")


def test_to_bronze_is_idempotent(tmp_path):
    out = normalize_iqa(_historical(), source="historical")
    p1 = to_bronze(out, tmp_path, partition="year=2024")
    first = pd.read_parquet(p1)
    p2 = to_bronze(out, tmp_path, partition="year=2024")
    second = pd.read_parquet(p2)
    assert p1 == p2
    assert len(first) == len(second)
