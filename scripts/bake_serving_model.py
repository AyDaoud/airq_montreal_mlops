"""Train a small RandomForest so the serving image always contains a model.

Prefers the real gold table; falls back to a deterministic synthetic frame so
CI can bake a model with no network and no data. This is what stops the
published image from being an empty shell (blocker B4).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.build_features import build_features_daily_iqa

GOLD = Path("data/gold/daily_station_iqa.parquet")
OUT_DIR = Path("artifacts/rf")


def _synthetic_gold(n_days: int = 120) -> pd.DataFrame:
    """Deterministic stand-in used when no gold table is available."""
    rows = []
    for station_id in (3, 6):
        for day in range(n_days):
            rows.append(
                {
                    "station_id": station_id,
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": 20 + (day % 17) + (station_id % 5),
                    "n_hours_observed": 24,
                    "temp_mean": 5.0 + (day % 25),
                    "temp_min": 2.0 + (day % 20),
                    "temp_max": 9.0 + (day % 28),
                    "humidity_mean": 60.0 + (day % 30),
                    "precip_sum": float(day % 4),
                    "wind_speed_mean": 3.0 + (day % 7),
                    "wind_speed_max": 8.0 + (day % 11),
                    "pressure_mean": 1000.0 + (day % 15),
                    "wind_dir_sin": 0.3,
                    "wind_dir_cos": 0.6,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    using_gold = GOLD.exists()
    source = pd.read_parquet(GOLD) if using_gold else _synthetic_gold()
    print(f"baking from {'gold table' if using_gold else 'synthetic fixture'}")

    frame, features = build_features_daily_iqa(source)
    model = RandomForestRegressor(
        n_estimators=40, max_depth=8, random_state=42, n_jobs=1
    )
    model.fit(frame[features], frame["target"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "model.pkl")
    (OUT_DIR / "feature_names.json").write_text(json.dumps(features))

    size_kb = (OUT_DIR / "model.pkl").stat().st_size / 1024
    print(f"wrote {OUT_DIR}/model.pkl ({size_kb:.0f} KB), {len(features)} features")


if __name__ == "__main__":
    main()
