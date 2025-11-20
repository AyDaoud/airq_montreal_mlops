# src/models/forecast.py
from __future__ import annotations
import argparse, json, re
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import mlflow

# ---------- artifact locations ----------
ART_DIR         = Path("artifacts")
RF_MODEL_PATH   = ART_DIR / "rf"      / "model.pkl"
RF_FEATS_PATH   = ART_DIR / "rf"      / "feature_names.json"
PROPHET_PATH    = ART_DIR / "prophet" / "prophet.pkl"
LSTM_STATE_PATH = ART_DIR / "lstm"    / "lstm.pt"
LSTM_META_PATH  = ART_DIR / "lstm"    / "model_meta.json"

# ---------- small helpers ----------
def _load_rf():
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"RF model not found at {RF_MODEL_PATH.resolve()}")
    if not RF_FEATS_PATH.exists():
        raise FileNotFoundError(f"RF feature names not found at {RF_FEATS_PATH.resolve()}")
    return joblib.load(RF_MODEL_PATH), json.loads(RF_FEATS_PATH.read_text())


def _infer_lags(feature_names: List[str]) -> List[int]:
    return sorted({
        int(m.group(1))
        for c in feature_names
        for m in [re.fullmatch(r"lag_(\d+)", c)]
        if m
    })


def _cal_feats(ts: pd.Timestamp) -> Dict[str, float]:
    return {
        "hour":  float(ts.hour),
        "dow":   float(ts.dayofweek),
        "month": float(ts.month),
    }


def _initial_roll(vals: List[float], window: int) -> float:
    if len(vals) <= 1:
        return float("nan")
    prev = vals[:-1][-window:]
    return float(np.mean(prev)) if len(prev) else float("nan")


def _update_roll(history: deque, window: int) -> float:
    arr = list(history)
    if len(arr) <= 1:
        return float("nan")
    prev = arr[:-1][-window:]
    return float(np.mean(prev)) if len(prev) else float("nan")


def _find_col(df: pd.DataFrame,
              prefer: Optional[str],
              aliases: List[str],
              contains: List[str] = []) -> Optional[str]:
    cols = list(df.columns)
    if prefer and prefer in cols:
        return prefer
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    for c in cols:
        lc = c.lower()
        if any(tok in lc for tok in contains):
            return c
    return None


def _normalize_pollutant_name(x: str) -> str:
    s = str(x).strip()
    # typical RSQA weirdness: PM2,5 → PM2.5
    s = s.replace("PM2,5", "PM2.5").replace("pm2,5", "PM2.5")
    return s.upper()


# ---------- loaders ----------
def _load_daily(paths: List[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError("No daily IQA parquet found.")
    df = pd.concat(frames, ignore_index=True)

    dt = _find_col(df, None, ["datetime", "date", "timestamp"], contains=["date", "time"])
    if dt and dt != "datetime":
        df = df.rename(columns={dt: "datetime"})
    v = _find_col(df, None,
                  ["value", "iqa", "Indice", "aqi", "valeur"],
                  contains=["iqa", "indice", "value", "valeur", "aqi"])
    if v and v != "value":
        df = df.rename(columns={v: "value"})

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"]).sort_values("datetime")
    return df


def _detect_hourly_schema(df: pd.DataFrame) -> Tuple[str, Optional[str], List[str]]:
    """
    Return: datetime_col, station_col (or None), pollutant_value_cols (wide-format).
    If 'value' already exists, pollutant_value_cols = [] (already long).
    """
    dtcol = _find_col(df, None,
                      ["datetime", "date_time", "timestamp"],
                      contains=["date", "time"])
    stcol = _find_col(df, None,
                      ["station_name", "station", "nom_station",
                       "id_station", "station_id", "code_station"],
                      contains=["station"])
    if dtcol is None:
        raise KeyError("No datetime-like column found in hourly parquet.")

    if "value" in df.columns:
        return dtcol, stcol, []

    exclude = {dtcol}
    if stcol:
        exclude.add(stcol)

    num_like = []
    for c in df.columns:
        if c in exclude:
            continue
        cl = str(c).lower()
        if cl in ["no_poste", "poste", "site", "station", "nom_station",
                  "id_station", "station_id", "code_station"]:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            num_like.append(c)

    if not num_like:
        raise KeyError("Could not infer pollutant columns from hourly parquet (no numeric-like columns).")

    return dtcol, stcol, num_like


def _load_hourly(paths: List[Path], args) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError("No hourly RSQA parquet found.")
    df = pd.concat(frames, ignore_index=True)

    # Respect manual overrides first
    if args.dtcol and args.dtcol in df.columns:
        df = df.rename(columns={args.dtcol: "datetime"})
    if args.stcol and args.stcol in df.columns:
        df = df.rename(columns={args.stcol: "station_name"})
    if args.pocol and args.pocol in df.columns:
        df = df.rename(columns={args.pocol: "pollutant"})
    if args.vcol and args.vcol in df.columns:
        df = df.rename(columns={args.vcol: "value"})

    dtcol, stcol, pollutant_cols = _detect_hourly_schema(df)
    if dtcol != "datetime":
        df = df.rename(columns={dtcol: "datetime"})
    if stcol and stcol != "station_name":
        df = df.rename(columns={stcol: "station_name"})

    # Wide → long
    if "value" not in df.columns:
        df = df.melt(
            id_vars=[c for c in ["datetime", "station_name"] if c in df.columns],
            value_vars=pollutant_cols,
            var_name="pollutant",
            value_name="value"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if "pollutant" in df.columns:
        df["pollutant"] = df["pollutant"].astype(str).map(_normalize_pollutant_name)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"]).sort_values("datetime")
    return df


def _list_stations(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if "station" in c.lower()]
    if not cols:
        return []
    s = pd.Series(dtype=str)
    for c in cols:
        s = pd.concat([s, df[c].astype(str)], ignore_index=True)
    return sorted(s.dropna().unique().tolist())

# ---------- forecasting backends ----------
def _rf_forecast(history_df: pd.DataFrame, horizon: int, freq: str = "D") -> pd.DataFrame:
    model, feature_names = _load_rf()
    lags = _infer_lags(feature_names)
    if not lags:
        raise ValueError("RF model has no lag_* features; cannot forecast.")

    df = history_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"]).sort_values("datetime")

    max_lag = max(lags)
    if len(df) < max_lag:
        raise ValueError(f"Need >= {max_lag} history rows; got {len(df)}.")

    roll_feature, roll_window = None, None
    for name, win in [("roll_7", 7), ("roll_24", 24)]:
        if name in feature_names:
            roll_feature, roll_window = name, win
            break

    hist_vals = deque(df["value"].values[-max_lag:].tolist(), maxlen=max_lag)
    last_ts = df["datetime"].iloc[-1]
    if roll_feature:
        _ = _initial_roll(list(df["value"].values), roll_window)

    out = []
    step = pd.tseries.frequencies.to_offset(freq)
    for _ in range(horizon):
        next_ts = last_ts + step
        row = {f"lag_{L}": float(hist_vals[-L]) for L in lags}
        if roll_feature:
            row[roll_feature] = float(_update_roll(hist_vals, roll_window))
        row.update(_cal_feats(next_ts))
        for f in feature_names:
            if f not in row:
                row[f] = 0.0
        yhat = float(model.predict(pd.DataFrame([row])[feature_names])[0])
        out.append({"datetime": next_ts, "forecast": yhat})
        hist_vals.append(yhat)
        last_ts = next_ts

    return pd.DataFrame(out)


def _prophet_forecast(history_df: pd.DataFrame, horizon: int, freq: str = "D") -> pd.DataFrame:
    """
    Forecast using the *saved* Prophet model in artifacts/prophet/prophet.pkl.
    No Stan optimization here – just predict.
    """
    from prophet import Prophet  # noqa: F401  (ensures dependency present)

    if not PROPHET_PATH.exists():
        raise FileNotFoundError(
            f"Prophet artifact not found at {PROPHET_PATH}. "
            "Run `python -m scripts.train_daily_iqa --model prophet` first."
        )

    m: "Prophet" = joblib.load(PROPHET_PATH)

    df = history_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"]).sort_values("datetime")

    last_ts = df["datetime"].iloc[-1].tz_convert("UTC")
    step = "H" if freq.upper() == "H" else "D"

    future_index = pd.date_range(
        last_ts + pd.tseries.frequencies.to_offset(step),
        periods=horizon,
        freq=step
    ).tz_convert(None)

    future = pd.DataFrame({"ds": future_index})
    fc = m.predict(future)[["ds", "yhat"]]
    fc = fc.rename(columns={"ds": "datetime", "yhat": "forecast"})
    fc["datetime"] = pd.to_datetime(fc["datetime"], utc=False).dt.tz_localize("UTC")
    return fc


def _lstm_forecast(history_df: pd.DataFrame, horizon: int, freq: str = "D") -> pd.DataFrame:
    import torch, torch.nn as nn

    if not LSTM_STATE_PATH.exists() or not LSTM_META_PATH.exists():
        raise FileNotFoundError(
            f"LSTM artifacts not found. Expected weights at {LSTM_STATE_PATH} "
            f"and meta at {LSTM_META_PATH}."
        )

    meta = json.loads(LSTM_META_PATH.read_text())

    # --- handle both meta formats ---
    if "seq_len" not in meta:
        raise KeyError(f"LSTM meta.json missing 'seq_len'. Got keys: {list(meta.keys())}")

    seq_len = int(meta["seq_len"])

    # Newer format: mean/std at top level
    if "mean" in meta and "std" in meta:
        mu = float(meta["mean"])
        sigma = float(meta["std"])
    # Existing format from model_factory: nested under "scaler"
    elif "scaler" in meta and isinstance(meta["scaler"], dict) \
            and "mean" in meta["scaler"] and "std" in meta["scaler"]:
        mu = float(meta["scaler"]["mean"])
        sigma = float(meta["scaler"]["std"])
    else:
        raise KeyError(
            f"LSTM meta.json missing mean/std. Got {meta}. "
            "Expected either top-level 'mean'/'std' or 'scaler':{'mean','std'}."
        )

    class LSTMReg(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)          # (B, T, H)
            return self.fc(out[:, -1, :])  # (B, 1)

    # infer hidden size from the saved state
    state = torch.load(LSTM_STATE_PATH, map_location="cpu")
    hidden = state["lstm.weight_ih_l0"].shape[0] // 4
    net = LSTMReg(hidden)
    net.load_state_dict(state)
    net.eval()

    df = history_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"]).sort_values("datetime")

    series = df["value"].to_numpy(dtype=np.float32)
    if len(series) < seq_len:
        raise ValueError(f"Need >= {seq_len} history rows for LSTM; got {len(series)}.")

    # normalize with same stats used in training
    z = (series - mu) / (sigma + 1e-8)

    step = pd.tseries.frequencies.to_offset(freq)
    last_ts = df["datetime"].iloc[-1]
    hist = deque(z[-seq_len:].tolist(), maxlen=seq_len)

    out = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.from_numpy(
                np.array(hist, dtype=np.float32).reshape(1, seq_len, 1)
            )
            yhat_z = net(x).squeeze(1).item()
            yhat = yhat_z * sigma + mu
            last_ts = last_ts + step
            out.append({"datetime": last_ts, "forecast": float(yhat)})
            hist.append(yhat_z)

    return pd.DataFrame(out)


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser("AirQ | Forecast (Daily/Hourly) with RF / Prophet / LSTM + MLflow")
    p.add_argument("--model", choices=["rf", "prophet", "lstm"], default="rf")
    p.add_argument("--freq", choices=["D", "H"], default="D")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--history-window", type=int, default=30)
    p.add_argument("--station-filter", type=str, default=None)
    p.add_argument("--pollutant", type=str, default=None)
    p.add_argument("--input-parquet", nargs="*", default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--mlflow-uri", type=str, default=None)
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument("--list-stations", action="store_true")
    p.add_argument("--print-schema", dest="print_schema", action="store_true")
    # manual overrides (hourly)
    p.add_argument("--datetime-col", dest="dtcol", type=str, default=None)
    p.add_argument("--station-col", dest="stcol", type=str, default=None)
    p.add_argument("--pollutant-col", dest="pocol", type=str, default=None)
    p.add_argument("--value-col", dest="vcol", type=str, default=None)
    args = p.parse_args()

    mlflow_uri = args.mlflow_uri or f"sqlite:///{(Path.cwd() / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(mlflow_uri)
    default_exp = {
        "rf": "AirQ-Forecast-RF",
        "prophet": "AirQ-Forecast-PROPHET",
        "lstm": "AirQ-Forecast-LSTM",
    }[args.model]
    mlflow.set_experiment(args.experiment or default_exp)

    # Choose inputs
    if args.input_parquet:
        paths = [Path(p) for p in args.input_parquet]
    else:
        paths = (
            [Path("data/interim/rsqa_2022.parquet"),
             Path("data/interim/rsqa_2023.parquet"),
             Path("data/interim/rsqa_2024.parquet")]
            if args.freq == "H"
            else [Path("data/interim/iqa_daily_2022_2024.parquet"),
                  Path("data/interim/iqa_daily_2025_2027.parquet")]
        )

    df = _load_hourly(paths, args) if args.freq == "H" else _load_daily(paths)

    if args.list_stations:
        stations = _list_stations(df)
        print("\nAvailable station names (unique):")
        for s in stations:
            print(" -", s)
        print(f"\nTotal: {len(stations)}")
        return

    if args.print_schema:
        print("\nColumns:", list(df.columns))
        print(df.head(3))
        return

    # Filters
    if args.station_filter:
        stc = _find_col(df, args.stcol,
                        ["station_name", "station", "nom_station",
                         "id_station", "station_id", "code_station"],
                        contains=["station"])
        if stc:
            df = df[df[stc].astype(str).str.contains(args.station_filter,
                                                     case=False, na=False)]

    if args.freq == "H" and args.pollutant:
        if "pollutant" not in df.columns:
            raise KeyError("Hourly table has no 'pollutant'. Provide --pollutant-col if your data is long format.")
        want = _normalize_pollutant_name(args.pollutant)
        df = df[df["pollutant"].astype(str).map(_normalize_pollutant_name) == want]

    # Standardize datetime/value
    dt = _find_col(df, args.dtcol,
                   ["datetime", "date_time", "timestamp"],
                   contains=["date", "time"])
    if dt and dt != "datetime":
        df = df.rename(columns={dt: "datetime"})
    if "datetime" not in df.columns:
        raise KeyError("No datetime column found after loading.")

    if "value" not in df.columns:
        vcol = _find_col(df, args.vcol,
                         ["value", "valeur", "concentration", "result", "measurement"],
                         contains=["value", "val", "conc", "result", "measure"])
        if not vcol:
            raise KeyError("No 'value' column could be derived; pass --value-col explicitly.")
        df = df.rename(columns={vcol: "value"})

    df = df.sort_values("datetime")

    # Build history window
    hist_len_needed = args.history_window
    if args.model == "rf":
        _, feats = _load_rf()
        lags = _infer_lags(feats)
        if lags:
            hist_len_needed = max(hist_len_needed, max(lags))

    hist = df[["datetime", "value"]].tail(hist_len_needed).copy()
    hist["datetime"] = pd.to_datetime(hist["datetime"], utc=True, errors="coerce")
    hist["value"] = pd.to_numeric(hist["value"], errors="coerce")
    hist = hist.dropna(subset=["datetime", "value"])

    # Forecast
    if args.model == "rf":
        fc = _rf_forecast(hist, args.horizon, args.freq)
        out_default = "data/predictions/rf_forecast.parquet"
    elif args.model == "prophet":
        fc = _prophet_forecast(hist, args.horizon, args.freq)
        out_default = "data/predictions/prophet_forecast.parquet"
    else:
        fc = _lstm_forecast(hist, args.horizon, args.freq)
        out_default = "data/predictions/lstm_forecast.parquet"

    out_path = Path(args.output or out_default)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out_path, index=False)

    # MLflow logging
    with mlflow.start_run(run_name=f"forecast_{args.model}_{args.freq}_{args.horizon}"):
        mlflow.log_param("model", args.model)
        mlflow.log_param("horizon", args.horizon)
        mlflow.log_param("freq", args.freq)
        mlflow.log_param("history_window", args.history_window)
        mlflow.log_param("history_used", len(hist))
        mlflow.log_param("station_filter", args.station_filter or "")
        mlflow.log_param("pollutant", args.pollutant or "")
        mlflow.log_param("input_paths", ",".join(str(p) for p in paths))
        if args.model == "rf":
            mlflow.log_param("rf_model_artifact", str(RF_MODEL_PATH))
            mlflow.log_param("rf_features_artifact", str(RF_FEATS_PATH))
        elif args.model == "lstm":
            mlflow.log_param("lstm_state", str(LSTM_STATE_PATH))
            mlflow.log_param("lstm_meta", str(LSTM_META_PATH))

        yhat = fc["forecast"].astype(float).to_numpy()
        mlflow.log_metric("forecast_mean", float(np.mean(yhat)))
        mlflow.log_metric("forecast_std", float(np.std(yhat)))
        mlflow.log_metric("forecast_min", float(np.min(yhat)))
        mlflow.log_metric("forecast_max", float(np.max(yhat)))

        tmp_json = out_path.with_suffix(".json")
        tmp_json.write_text(json.dumps(
            fc.assign(datetime=fc["datetime"].astype(str)).to_dict("records"),
            indent=2
        ))
        mlflow.log_artifact(str(out_path))
        mlflow.log_artifact(str(tmp_json))

    print(f"✅ Forecast saved → {out_path}")
    print("First rows:")
    print(fc.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
