"""Do-nothing baselines every model must be measured against.

Persistence is the one that matters: on daily air quality it is strong, and
until Spec B it had never been computed. Every baseline is grouped by
station so no value ever leaks across a station boundary.
"""

from __future__ import annotations

import pandas as pd

VALUE_COLUMN = "iqa"
GROUP_COLUMN = "station_id"
DATE_COLUMN = "date_local"


def persistence(df: pd.DataFrame) -> pd.Series:
    """Tomorrow equals today."""
    return df[VALUE_COLUMN].copy()


def seasonal_naive(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """Tomorrow equals the value ``period`` days before today.

    With ``period=7`` this is "the same weekday last week": shifting by
    ``period - 1`` from today lands on that day for a next-day target.
    """
    return df.groupby(GROUP_COLUMN)[VALUE_COLUMN].shift(period - 1)


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """The training mean for this station and calendar month.

    Falls back to the station mean, then the global mean, so an unseen
    station-month never yields NaN.
    """
    train = train.copy()
    test = test.copy()
    train["_month"] = train[DATE_COLUMN].dt.month
    test["_month"] = test[DATE_COLUMN].dt.month

    by_station_month = train.groupby([GROUP_COLUMN, "_month"])[VALUE_COLUMN].mean()
    by_station = train.groupby(GROUP_COLUMN)[VALUE_COLUMN].mean()
    overall = train[VALUE_COLUMN].mean()

    keys = pd.MultiIndex.from_arrays([test[GROUP_COLUMN], test["_month"]])
    values = by_station_month.reindex(keys).to_numpy()

    out = pd.Series(values, index=test.index)
    out = out.fillna(test[GROUP_COLUMN].map(by_station))
    return out.fillna(overall)
