"""Ingestion watermarks stored in a JSON sidecar.

A watermark is the source's own change token (CKAN ``last_modified`` for
portal resources, the ingest date for the realtime feed). Re-running an
ingest whose token is unchanged is a no-op, which is what makes the
pipeline idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_watermarks(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_watermark(path: Path, key: str, token: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    marks = read_watermarks(path)
    marks[key] = token
    path.write_text(json.dumps(marks, indent=2, sort_keys=True))


def should_skip(path: Path, key: str, token: str | None) -> bool:
    """True when ``key`` was last ingested at exactly ``token``.

    A ``None`` token means the source gave us no change signal, so we must
    re-fetch rather than assume nothing moved.
    """
    if token is None:
        return False
    return read_watermarks(path).get(key) == token
