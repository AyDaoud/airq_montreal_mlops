"""MLflow tracking configuration.

Resolution order: an explicit argument, then ``MLFLOW_TRACKING_URI``, then a
repository-relative SQLite file. Never a machine-specific absolute path.
"""

from __future__ import annotations

import os

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def resolve_tracking_uri(explicit: str | None = None) -> str:
    """Return the MLflow tracking URI to use.

    An exported-but-empty ``MLFLOW_TRACKING_URI`` is treated as unset, since
    ``export MLFLOW_TRACKING_URI=`` is a common way to clear it.
    """
    if explicit:
        return explicit
    from_env = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    return from_env or DEFAULT_TRACKING_URI
