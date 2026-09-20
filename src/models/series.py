"""Per-station series access.

Prophet and the LSTM previously consumed a flat frame containing all 11
stations interleaved by date. Prophet then saw 11 ``y`` values per ``ds``,
and the LSTM slid a 30-day window across station boundaries, producing
physically meaningless sequences.
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMN = "station_id"
DATE_COLUMN = "date_local"


def station_series(df: pd.DataFrame, station_id: int) -> pd.DataFrame:
    """One station's rows, sorted by date."""
    subset = df[df[GROUP_COLUMN] == station_id]
    if subset.empty:
        known = sorted(df[GROUP_COLUMN].unique().tolist())
        raise KeyError(f"station {station_id} not present; known stations: {known}")
    return subset.sort_values(DATE_COLUMN).reset_index(drop=True)


def station_series_map(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Every station's series, keyed by station id."""
    return {
        int(station_id): station_series(df, int(station_id))
        for station_id in sorted(df[GROUP_COLUMN].unique())
    }
