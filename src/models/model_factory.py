# src/models/model_factory.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import torch


@dataclass
class SavedModel:
    flavor: str
    path: str
    extra: dict


# src/models/model_factory.py


def save_sklearn(model, feature_names, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.pkl")
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names))
    return out_dir


def save_prophet(model, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "prophet.pkl")
    # simple meta in case you want later
    meta = {"framework": "prophet"}
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def save_lstm(
    state_dict,
    seq_len: int,
    mean: float,
    std: float,
    out_dir: Path,
) -> SavedModel:
    """
    Save LSTM weights + metadata in the format expected by forecast.py.

    - weights -> lstm.pt
    - metadata -> model_meta.json with seq_len + scaler stats
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # weights
    torch.save(state_dict, out_dir / "lstm.pt")

    # metadata
    meta = {
        "flavor": "lstm",
        "seq_len": int(seq_len),
        "scaler": {"mean": float(mean), "std": float(std)},
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    return SavedModel(
        flavor="lstm",
        path=str((out_dir / "lstm.pt").as_posix()),
        extra={"seq_len": int(seq_len), "mean": float(mean), "std": float(std)},
    )


def load_meta(art_dir: Path):
    meta = json.loads((art_dir / "model_meta.json").read_text())
    return meta
