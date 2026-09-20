"""HTTP access to donnees.montreal.ca.

The portal rejects non-browser User-Agents with HTTP 403 and the body
``RBAC: access denied``. Every download in this project therefore goes
through :func:`build_session`, never through ``requests.get`` or
``pandas.read_csv(url)`` directly.
"""

from __future__ import annotations

from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT = 180
CHUNK_SIZE = 1 << 16


def build_session(
    total_retries: int = 4, backoff_factor: float = 1.5
) -> requests.Session:
    """Return a session that the portal will serve.

    Retries cover the transient 503 observed on the realtime resource.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": BROWSER_UA,
            "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
        }
    )
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _etag_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".etag")


def download(
    url: str,
    dest: Path,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path:
    """Download ``url`` to ``dest``, skipping the transfer when unchanged.

    Uses conditional ``If-None-Match`` requests so re-running ingestion does
    not re-download tens of megabytes that have not changed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = session or build_session()

    headers = {}
    etag_file = _etag_path(dest)
    if dest.exists() and etag_file.exists():
        headers["If-None-Match"] = etag_file.read_text().strip()

    kwargs = {"stream": True, "timeout": timeout}
    if headers:
        kwargs["headers"] = headers

    with session.get(url, **kwargs) as response:
        if response.status_code == 304:
            return dest
        response.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
        tmp.replace(dest)
        etag = response.headers.get("ETag")
        if etag:
            etag_file.write_text(etag)

    return dest
