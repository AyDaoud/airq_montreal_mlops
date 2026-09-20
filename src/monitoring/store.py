"""Persistence for served predictions and the scores derived from them.

Everything goes through a ``DATABASE_URL`` so the same code runs on SQLite
locally and in tests, and on Postgres in a deployment, by changing one
variable. Nothing here computes a metric; that is scoring.py's job.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DEFAULT_SQLITE_PATH = Path("data/monitoring.db")


def resolve_database_url() -> str:
    """Resolve the database URL: environment first, then a local SQLite file."""
    from_env = os.getenv("DATABASE_URL", "").strip()
    if from_env:
        return from_env
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    served_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    station_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    target_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    prediction: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    features_json: Mapped[str] = mapped_column(Text)


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    window_days: Mapped[int] = mapped_column(Integer, index=True)
    n: Mapped[int] = mapped_column(Integer)
    mae_model: Mapped[float] = mapped_column(Float)
    mae_persistence: Mapped[float] = mapped_column(Float)
    mase: Mapped[float] = mapped_column(Float)


@dataclass(frozen=True)
class PredictionRecord:
    request_id: str
    served_at: datetime
    station_id: int | None
    target_date: date | None
    prediction: float
    model_name: str
    model_version: str
    features_json: str


@dataclass(frozen=True)
class ScoreRecord:
    scored_at: datetime
    window_days: int
    n: int
    mae_model: float
    mae_persistence: float
    mase: float


class Store:
    """Thin persistence layer. Callers never see SQLAlchemy types."""

    def __init__(self, url: str | None = None, create: bool = True) -> None:
        self.url = url or resolve_database_url()
        if self.url.startswith("sqlite:///") and ":memory:" not in self.url:
            Path(self.url.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        self.engine = create_engine(self.url, future=True)
        if self.url.startswith("sqlite") and ":memory:" not in self.url:
            # WAL lets a reader (the scoring job) run while the API writes.
            with self.engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        if create:
            Base.metadata.create_all(self.engine)

    def log_predictions(self, records: list[PredictionRecord]) -> int:
        if not records:
            return 0
        with Session(self.engine) as session:
            session.add_all([Prediction(**vars(r)) for r in records])
            session.commit()
        return len(records)

    def recent_predictions(self, limit: int = 100) -> list[Prediction]:
        with Session(self.engine, expire_on_commit=False) as session:
            statement = select(Prediction).order_by(Prediction.id.desc()).limit(limit)
            rows = list(session.scalars(statement))
            session.expunge_all()
            return rows

    def matured_predictions(self) -> list[Prediction]:
        """Predictions whose target date has passed and can be scored."""
        with Session(self.engine, expire_on_commit=False) as session:
            statement = select(Prediction).where(Prediction.target_date.is_not(None))
            rows = list(session.scalars(statement))
            session.expunge_all()
            return rows

    def write_score(self, record: ScoreRecord) -> None:
        with Session(self.engine) as session:
            session.add(Score(**vars(record)))
            session.commit()

    def latest_scores(self) -> dict[int, Score]:
        """Most recent score per window, keyed by window length in days."""
        with Session(self.engine, expire_on_commit=False) as session:
            rows = session.scalars(select(Score).order_by(Score.scored_at.desc()))
            latest: dict[int, Score] = {}
            for row in rows:
                latest.setdefault(row.window_days, row)
            session.expunge_all()
            return latest
