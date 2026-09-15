import pandas as pd
import pytest

from src.data.contracts import (
    ContractError,
    RAW_IQA_HISTORICAL,
    RAW_STATIONS,
    validate,
)

FIXTURES = "tests/fixtures"


def test_historical_fixture_passes_contract():
    df = pd.read_csv(f"{FIXTURES}/iqa_historical_sample.csv")
    out = validate(df, RAW_IQA_HISTORICAL)
    assert len(out) == len(df)


def test_hour_out_of_range_is_rejected():
    df = pd.DataFrame(
        {
            "stationId": [3],
            "polluant": ["PM"],
            "valeur": [10],
            "date": ["2024-01-01"],
            "heure": [24],
        }
    )
    with pytest.raises(ContractError):
        validate(df, RAW_IQA_HISTORICAL)


def test_negative_value_is_rejected():
    df = pd.DataFrame(
        {
            "stationId": [3],
            "polluant": ["PM"],
            "valeur": [-1],
            "date": ["2024-01-01"],
            "heure": [5],
        }
    )
    with pytest.raises(ContractError):
        validate(df, RAW_IQA_HISTORICAL)


def test_stations_contract_rejects_corrupt_latitude():
    """Station 62 has latitude 4.504576e+07 in the real file."""
    df = pd.read_csv(f"{FIXTURES}/stations_sample.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["numero_station"])
    with pytest.raises(ContractError, match="latitude"):
        validate(df, RAW_STATIONS)


def test_stations_contract_passes_after_dropping_bad_rows():
    df = pd.read_csv(f"{FIXTURES}/stations_sample.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["numero_station"])
    df = df[
        (df["latitude"].between(45.2, 45.8)) & (df["longitude"].between(-74.1, -73.4))
    ]
    out = validate(df, RAW_STATIONS)
    assert len(out) > 10
