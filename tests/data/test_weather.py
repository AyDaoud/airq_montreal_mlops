import json

from src.data.weather import HOURLY_VARS, parse_openmeteo

FIXTURES = "tests/fixtures"


def _payload():
    with open(f"{FIXTURES}/weather_archive_sample.json") as handle:
        return json.load(handle)


def test_boundary_layer_height_is_not_requested():
    """Archive returns null for it, so training cannot use it (decision D5)."""
    assert "boundary_layer_height" not in HOURLY_VARS


def test_parse_produces_tz_aware_utc_rows():
    out = parse_openmeteo(_payload(), cell_id="45.5_-73.6")
    assert len(out) == 48
    assert str(out["ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert (out["cell_id"] == "45.5_-73.6").all()


def test_parse_includes_every_requested_variable():
    out = parse_openmeteo(_payload(), cell_id="45.5_-73.6")
    for var in HOURLY_VARS:
        assert var in out.columns, f"{var} missing from parsed frame"


def test_wind_direction_within_range():
    out = parse_openmeteo(_payload(), cell_id="45.5_-73.6")
    direction = out["wind_direction_10m"].dropna()
    assert direction.between(0, 360).all()


def test_parse_is_empty_safe():
    out = parse_openmeteo({"hourly": {"time": []}}, cell_id="x")
    assert out.empty
    assert "ts_utc" in out.columns
