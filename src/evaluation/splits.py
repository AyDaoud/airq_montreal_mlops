"""Time-respecting splits.

The original ``_time_split`` sliced ``iloc[:k]`` on a frame sorted by
``[station_id, date_local]``, which yields a *station* holdout, not a time
one. Every split here is computed from dates, never from row positions.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

DATE_COLUMN = "date_local"


def _require_dates(df: pd.DataFrame) -> None:
    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Frame has no {DATE_COLUMN!r} column; splits are computed from "
            "dates, never from row positions."
        )


def _capped(df: pd.DataFrame, max_date: pd.Timestamp | str | None) -> pd.DataFrame:
    if max_date is None:
        return df
    return df[df[DATE_COLUMN] <= pd.Timestamp(max_date)]


def time_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    max_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on a date boundary so the holdout strictly follows training."""
    _require_dates(df)
    frame = _capped(df, max_date)
    cutoff = frame[DATE_COLUMN].quantile(1 - test_fraction)
    train = frame[frame[DATE_COLUMN] < cutoff]
    test = frame[frame[DATE_COLUMN] >= cutoff]
    return train, test


def rolling_origin_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_days: int = 90,
    max_date: pd.Timestamp | str | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``n_folds`` expanding-window folds, oldest first.

    Fold *k* trains on everything before its cutoff and tests the following
    ``test_days`` days. Test windows are contiguous and non-overlapping.
    """
    _require_dates(df)
    frame = _capped(df, max_date)
    end = frame[DATE_COLUMN].max()

    cutoffs = [end - pd.Timedelta(days=test_days * (k + 1)) for k in range(n_folds)][
        ::-1
    ]

    for cutoff in cutoffs:
        train = frame[frame[DATE_COLUMN] < cutoff]
        test = frame[
            (frame[DATE_COLUMN] >= cutoff)
            & (frame[DATE_COLUMN] < cutoff + pd.Timedelta(days=test_days))
        ]
        if train.empty or test.empty:
            continue
        yield train, test
