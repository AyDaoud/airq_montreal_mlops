# src/features/build_features.py
from __future__ import annotations
import pandas as pd
import re


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


def build_features_daily_iqa(df_daily: pd.DataFrame, lags=(1, 2, 3, 7, 14)):
    """
    Robust daily IQA features:
    - auto-detects date & value columns
    - auto-chooses station key
    """
    x = df_daily.copy()

    # date -> datetime
    for cand in [
        "datetime",
        "date",
        "Date",
        "DATE",
        "date_heure",
        "Date_Heure",
        "jour",
        "Jour",
        "JOUR",
    ]:
        if cand in x.columns:
            x = x.rename(columns={cand: "datetime"})
            break
    x["datetime"] = pd.to_datetime(x["datetime"], utc=True, errors="coerce")
    x = x.dropna(subset=["datetime"])

    # value -> IQA
    value_candidates = [
        "value",
        "iqa",
        "IQA",
        "indice",
        "Indice",
        "iqa_value",
        "valeur_iqa",
        "Valeur",
        "valeur",
    ]
    val_col = next((c for c in value_candidates if c in x.columns), None)
    if val_col is None:
        fuzzy = [c for c in x.columns if re.search(r"(iqa|indice)", str(c), re.I)]
        if len(fuzzy) == 1:
            val_col = fuzzy[0]
    if val_col is None:
        numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
        if len(numeric_cols) >= 1:
            val_col = numeric_cols[0]
    if val_col is None:
        raise ValueError(f"No IQA/value-like column found. Columns: {list(x.columns)}")

    if val_col != "value":
        x = x.rename(columns={val_col: "value"})

    gkey = _choose_station_key(x)
    x = x.sort_values([gkey, "datetime"])

    for lag in lags:
        x[f"lag_{lag}"] = x.groupby(gkey)["value"].shift(lag)
    x["roll_7"] = x.groupby(gkey)["value"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )

    x["dow"] = x["datetime"].dt.dayofweek
    x["month"] = x["datetime"].dt.month
    x["target"] = x.groupby(gkey)["value"].shift(-1)
    x["value"] = pd.to_numeric(x["value"], errors="coerce")

    x = x.dropna().reset_index(drop=True)

    exclude = {"datetime", "value", "target"}
    numeric_cols = [
        c for c in x.columns if c not in exclude and pd.api.types.is_numeric_dtype(x[c])
    ]
    feats = numeric_cols
    return x, feats
