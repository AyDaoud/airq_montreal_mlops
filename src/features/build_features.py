# src/features/build_features.py
from __future__ import annotations
import pandas as pd


def _choose_station_key(df: pd.DataFrame) -> str:
    candidates = [
        "station_name",
        "station",
        "nom_station",
        "Nom_station",
        "Station",
        "NomStation",
        "station_id",
        "code_station",
        "Code_Station",
        "no_station",
        "No_Station",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    df["_series_id"] = "ALL"
    return "_series_id"


def build_features(
    df_pollut: pd.DataFrame,
    df_weather: pd.DataFrame | None = None,
    pollutant: str | None = None,
    lags=(1, 2, 3, 6, 12, 24),
):
    x = df_pollut.copy()

    # normalize minimal columns
    if "datetime" not in x.columns:
        for alt in ["date_heure", "Date_Heure", "Date", "DATE", "date"]:
            if alt in x.columns:
                x = x.rename(columns={alt: "datetime"})
                break

    if "value" not in x.columns:
        for alt in ["valeur", "Valeur", "concentration", "Concentration"]:
            if alt in x.columns:
                x = x.rename(columns={alt: "value"})
                break

    keep = [
        c
        for c in ["datetime", "station_id", "station_name", "pollutant", "value"]
        if c in x.columns
    ]
    x = x[keep]
    x["datetime"] = pd.to_datetime(x["datetime"], utc=True, errors="coerce")
    x = x.dropna(subset=["datetime"])

    if pollutant and "pollutant" in x.columns:
        x = x.query("pollutant == @pollutant")

    if (
        df_weather is not None
        and len(df_weather) > 0
        and "datetime" in df_weather.columns
    ):
        w = df_weather.copy()
        w["datetime"] = pd.to_datetime(w["datetime"], utc=True, errors="coerce")
        x = x.merge(w, on="datetime", how="left")

    gkey = _choose_station_key(x)
    x = x.sort_values([gkey, "datetime"])

    for lag in lags:
        x[f"lag_{lag}"] = x.groupby(gkey)["value"].shift(lag)

    # hourly vs daily heuristic
    dt = pd.to_datetime(x["datetime"])
    median_step = dt.diff().dt.total_seconds().median() or 0
    if median_step >= 24 * 3600:
        x["roll_7"] = x.groupby(gkey)["value"].transform(
            lambda s: s.shift(1).rolling(7).mean()
        )
    else:
        x["roll_24"] = x.groupby(gkey)["value"].transform(
            lambda s: s.shift(1).rolling(24).mean()
        )

    x["hour"] = dt.dt.hour
    x["dow"] = dt.dt.dayofweek
    x["month"] = dt.dt.month
    x["target"] = x.groupby(gkey)["value"].shift(-1)

    x = x.dropna().reset_index(drop=True)
    features = [
        c
        for c in x.columns
        if c
        not in (
            "datetime",
            "station_id",
            "station_name",
            "pollutant",
            "value",
            "target",
        )
    ]
    return x, features


DAILY_LAGS = (1, 2, 3, 7, 14)

_NON_FEATURE_COLUMNS = {
    "station_id",
    "date_local",
    "iqa",
    "driving_pollutant",
    "name",
    "borough",
    "cell_id",
    "source",
    "latitude",
    "longitude",
    "target_iqa_h24",
    "target_iqa_h48",
    "target_exceed_h24",
    "target_exceed_h48",
    "target",
}

_NON_FEATURE_PREFIXES = ("sub_",)


def build_features_daily_iqa(df_daily: pd.DataFrame, lags=DAILY_LAGS):
    """Build daily features from the gold table.

    The gold table is contract-validated upstream, so this function does no
    column guessing: it requires ``station_id``, ``date_local`` and ``iqa``
    and raises if any is absent.
    """
    required = {"station_id", "date_local", "iqa"}
    missing = required - set(df_daily.columns)
    if missing:
        raise ValueError(
            f"Gold frame is missing required columns {sorted(missing)}. "
            "Run 'python -m src.data.cli build' first."
        )

    x = df_daily.copy().sort_values(["station_id", "date_local"])

    grouped = x.groupby("station_id")["iqa"]
    for lag in lags:
        x[f"lag_{lag}"] = grouped.shift(lag)
    x["roll_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).mean())

    x["dow"] = x["date_local"].dt.dayofweek
    x["month"] = x["date_local"].dt.month

    # Spec B replaces this with an explicit horizon argument.
    x["target"] = x.groupby("station_id")["iqa"].shift(-1)

    feature_columns = [
        c
        for c in x.columns
        if c not in _NON_FEATURE_COLUMNS
        and not c.startswith(_NON_FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(x[c])
    ]

    x = x.dropna(subset=feature_columns + ["target"]).reset_index(drop=True)
    return x, feature_columns
