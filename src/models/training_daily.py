# scripts/train_daily_iqa.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import mlflow, joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.features.build_features import build_features_daily_iqa
from src.models.model_factory import save_sklearn, save_prophet, save_lstm
import numpy as np

# Prophet typed against np.float_ etc.; provide aliases when missing (NumPy 2.x)
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

from prophet import Prophet


def _daily_df():
    paths = [Path("data/interim/iqa_daily_2022_2024.parquet"),
             Path("data/interim/iqa_daily_2025_2027.parquet")]
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError("No daily IQA parquet found.")
    df = pd.concat(frames, ignore_index=True)

    if "datetime" not in df.columns:
        for c in ["date", "timestamp", "Date"]:
            if c in df.columns:
                df = df.rename(columns={c: "datetime"})
                break
    if "value" not in df.columns:
        for c in ["iqa", "Indice", "aqi", "valeur"]:
            if c in df.columns:
                df = df.rename(columns={c: "value"})
                break

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"]).sort_values("datetime")
    return df


def _time_split(df, ratio=0.2):
    n = len(df)
    k = int(n * (1 - ratio))
    return df.iloc[:k], df.iloc[k:]


def _metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def train_rf():
    df = _daily_df()
    feat_df, feats = build_features_daily_iqa(df)
    feat_df = feat_df.dropna(subset=feats + ["target"])
    tr, va = _time_split(feat_df)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, n_jobs=-1, random_state=42
    )
    model.fit(tr[feats], tr["target"])
    p_tr = model.predict(tr[feats])
    p_va = model.predict(va[feats])
    mae_tr, rmse_tr = _metrics(tr["target"], p_tr)
    mae_va, rmse_va = _metrics(va["target"], p_va)
    return model, feats, {
        "mae_tr": mae_tr,
        "rmse_tr": rmse_tr,
        "mae_va": mae_va,
        "rmse_va": rmse_va,
    }


def train_prophet():
    """
    Train a daily Prophet model on the IQA series.
    Ensures timezone information is removed from ds (Prophet requirement)
    and splits the data before fitting. Uses batched prediction to avoid
    excessive memory usage.
    """
    from prophet import Prophet
    import pandas as pd

    def _batched_predict(model, df, batch_size: int = 5000):
        """
        Run Prophet.predict in batches to avoid allocating huge arrays.
        """
        preds = []
        n = len(df)
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_ds = df.iloc[start:end][["ds"]]
            batch_pred = model.predict(batch_ds)["yhat"]  # pandas Series
            preds.append(batch_pred)
        return pd.concat(preds, ignore_index=True).to_numpy()

    # Load daily data and coerce columns
    df = _daily_df()[["datetime", "value"]].copy()
    df["ds"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")[["ds", "y"]]

    # Split train and validation (20 % holdout)
    tr, va = _time_split(df, ratio=0.2)

    # Fit Prophet
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(tr)

    # Reduce / disable uncertainty sampling to avoid 1000 x N arrays
    m.uncertainty_samples = 0

    # Evaluate on holdout (batched)
    yhat_va = _batched_predict(m, va, batch_size=5000)
    mae_va, rmse_va = _metrics(va["y"].to_numpy(), yhat_va)

    # Optional: compute train metrics using in-sample predictions (batched)
    yhat_tr = _batched_predict(m, tr, batch_size=5000)
    mae_tr, rmse_tr = _metrics(tr["y"].to_numpy(), yhat_tr)

    return m, None, {
        "mae_tr": mae_tr,
        "rmse_tr": rmse_tr,
        "mae_va": mae_va,
        "rmse_va": rmse_va,
    }



def train_lstm(seq_len=30, epochs=10, lr=1e-3, hidden=64, batch_size=64):
    import torch, torch.nn as nn
    torch.set_num_threads(1)

    df = _daily_df()[["datetime", "value"]].copy()
    s = df["value"].astype(float).values
    mu, sigma = float(np.mean(s)), float(np.std(s) + 1e-8)
    z = (s - mu) / sigma

    def make_xy(arr, L):
        X, y = [], []
        for i in range(L, len(arr)):
            X.append(arr[i - L : i])
            y.append(arr[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X, y = make_xy(z, seq_len)
    k = int(len(X) * 0.8)
    Xtr, Xva = X[:k], X[k:]
    ytr, yva = y[:k], y[k:]

    class LSTMReg(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    device = torch.device("cpu")
    net = LSTMReg(hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()

    def to_batch(a):
        return torch.from_numpy(a.reshape(-1, seq_len, 1)).to(device)

    n = len(Xtr)
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = to_batch(Xtr[idx])
            yb = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad()
            pred = net(xb).squeeze(1)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()

    def predict_batches(Xa):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xa), batch_size):
                out.append(net(to_batch(Xa[i : i + batch_size])).squeeze(1).cpu().numpy())
        return np.concatenate(out, axis=0)

    p_tr = predict_batches(Xtr)
    p_va = predict_batches(Xva)

    p_tr = p_tr * sigma + mu
    ytr_den = ytr * sigma + mu
    p_va = p_va * sigma + mu
    yva_den = yva * sigma + mu

    mae_tr, rmse_tr = _metrics(ytr_den, p_tr)
    mae_va, rmse_va = _metrics(yva_den, p_va)
    return net.state_dict(), {"seq_len": seq_len, "mean": mu, "std": sigma}, {
        "mae_tr": mae_tr,
        "rmse_tr": rmse_tr,
        "mae_va": mae_va,
        "rmse_va": rmse_va,
    }
