import pandas as pd

from src.data.sources import KNOWN_STATION_IDS
from src.data.stations import assign_cell, load_stations

FIXTURES = "tests/fixtures"


def _raw():
    return pd.read_csv(f"{FIXTURES}/stations_sample.csv", encoding="utf-8-sig")


def test_load_drops_all_nan_rows_and_corrupt_coordinates():
    out = load_stations(_raw())
    assert out["latitude"].between(45.2, 45.8).all()
    assert out["longitude"].between(-74.1, -73.4).all()
    assert 62 not in out["station_id"].tolist()


def test_every_known_iqa_station_is_present():
    out = load_stations(_raw())
    missing = set(KNOWN_STATION_IDS) - set(out["station_id"])
    assert missing == set(), f"missing geo for stations {sorted(missing)}"


def test_closed_stations_are_not_filtered_out():
    """statut is stale: 28, 50 and 66 report data yet are marked 'ferme'."""
    out = load_stations(_raw())
    for station_id in (28, 50, 66):
        assert station_id in out["station_id"].tolist()


def test_assign_cell_rounds_to_tenth_of_a_degree():
    assert assign_cell(45.5622, -73.5718) == "45.6_-73.6"
    assert assign_cell(45.4265, -73.9289) == "45.4_-73.9"


def test_known_stations_collapse_to_seven_cells():
    out = load_stations(_raw())
    known = out[out["station_id"].isin(KNOWN_STATION_IDS)]
    assert known["cell_id"].nunique() == 7
