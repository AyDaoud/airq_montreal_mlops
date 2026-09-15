# Spec A — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `airq_montreal_mlops` run end-to-end from a clean clone by committing a real, contract-validated ingestion layer that also carries weather and station geography.

**Architecture:** A medallion pipeline under `src/data/`. Bronze holds downloads verbatim; silver holds normalized hourly IQA, hourly weather and the station dimension; gold holds one row per `(station_id, date_local)` with weather, geography and the regression + exceedance targets Spec B will consume. Network access is isolated to two modules (`http.py` for the Montréal portal, `weather.py` for Open-Meteo) so every test runs offline against committed fixtures.

**Tech Stack:** Python 3.12, pandas 2.2, pyarrow, pandera (data contracts), requests + urllib3 Retry, pytest, FastAPI, Docker, GitHub Actions.

**Reference spec:** `docs/superpowers/specs/2026-09-14-spec-a-data-foundation-design.md`

---

## Environment notes (read before starting)

- Python 3.12.9 is available at `/home/ayman/miniconda3/bin/python3`. The global env has pandas **3.0.5**, but this project pins pandas **2.2.2** — so **all work happens inside `.venv`**, never the conda base env.
- **Docker is not installed on this machine.** Tasks 16–17 cannot be verified locally; they are verified by GitHub Actions on push. Each such step says so explicitly and gives the CI check to watch.
- `donnees.montreal.ca` returns `403 RBAC: access denied` to non-browser User-Agents. Never use bare `requests.get` or `pd.read_csv(url)` against it.

## File structure

| File | Responsibility |
|---|---|
| `src/data/http.py` | The only module that talks to the Montréal portal. Browser UA, retry/backoff, streaming download. |
| `src/data/sources.py` | Frozen registry of CKAN dataset/resource IDs. No logic. |
| `src/data/ckan.py` | Resolve a resource ID to its current download URL + `last_modified`. |
| `src/data/contracts.py` | pandera schemas, one per boundary. |
| `src/data/watermark.py` | Read/write ingestion watermarks in a JSON sidecar. |
| `src/data/rsqa_ingest.py` | Historical backfill + real-time increment → bronze. |
| `src/data/stations.py` | Station dimension: geo, names, boroughs, weather cell assignment. |
| `src/data/weather.py` | The only module that talks to Open-Meteo. Archive + forecast → bronze. |
| `src/data/aggregate.py` | bronze → silver → gold, including target construction. |
| `src/data/cli.py` | `python -m src.data.cli <ingest\|build\|all>` entry point. |
| `tests/fixtures/` | Small committed slices of every real source, captured 2026-09-14. |

---

## Task 0: Development environment and dependency split

Fixes blocker **B2** (`evidently` imported but undeclared) and implements decision **D7** (image split).

**Files:**
- Create: `requirements-serving.txt`
- Modify: `requirements.txt`
- Modify: `setup.cfg`
- Modify: `makefile`

- [ ] **Step 1: Create the venv and confirm it is isolated from conda**

```bash
cd /home/ayman/airq_montreal_mlops
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -c "import sys; print(sys.prefix)"
```

Expected: prints a path ending in `/airq_montreal_mlops/.venv`.

- [ ] **Step 2: Write `requirements-serving.txt`**

This is the *serving* dependency set. It must not contain `torch`, `prophet`, `mlflow` or `evidently` — that exclusion is the entire point of D7.

```
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.7.4
pandas==2.2.2
numpy==2.0.2
scikit-learn==1.7.2
joblib==1.4.2
pyarrow==17.0.0
```

> **Why `numpy==2.0.2` and not the repo's existing `2.2.4`.** `evidently` 0.4.40 declares
> `numpy<2.1,>=1.22.0`, so `numpy==2.2.4` makes the requirement set unsatisfiable —
> meaning blocker B2 was never merely an undeclared import, it was **unfixable** at the
> repo's own numpy pin. Every release in the `0.4.x`–`0.7.0` range carries that cap; it is
> lifted only from `0.7.1`, which also breaks the `Report`/`Preset` API that
> `src/monitoring/check_iqa.py` is written against. Downgrading numpy keeps monitoring
> working untouched, as spec section 10 requires, and holds one numpy across training and
> serving so pickles and float behaviour cannot skew. Verified by `pip install --dry-run`:
> the full set resolves with numpy 2.0.2, evidently 0.4.40, pandera 0.25.0, torch 2.4.0.
>
> **Spec C must revisit this**: bump to `evidently>=0.7.1` and port `check_iqa.py` to the
> new API, which then frees numpy again.

- [ ] **Step 3: Rewrite `requirements.txt` as the full training set**

`evidently` is added here (blocker B2) and `pandera` + `pyarrow` are new for this spec. Keep `-r requirements-serving.txt` at the top so the two files cannot drift apart.

```
-r requirements-serving.txt

scipy==1.14.1
mlflow>=3,<4
prefect==2.19.9
httpx==0.27.0
prophet==1.2.1
torch==2.4.0
requests==2.32.3
pandera>=0.20,<0.26
evidently>=0.4.40,<0.5
pytest==8.2.1
pytest-cov==5.0.0
flake8
black
pre-commit
```

- [ ] **Step 4: Install and record the resolved pandera version**

```bash
.venv/bin/python -m pip install -r requirements.txt -q
.venv/bin/python -c "import pandera; print('pandera', pandera.__version__)"
```

Expected: prints a version in `[0.20, 0.26)`. Note it — Task 4 imports pandera through a compatibility shim that works across that range.

- [ ] **Step 5: Tighten `setup.cfg`**

The current config hides failures (`--maxfail=1`) and effectively disables linting (`max-line-length = 200`).

```ini
[tool:pytest]
addopts = -q --disable-warnings
testpaths = tests

[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .venv,.git,__pycache__,build,dist
```

- [ ] **Step 6: Add the new make targets**

Append to `makefile` (keep the existing targets):

```makefile
PY := .venv/bin/python

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) -m src.data.cli ingest

build-data:
	$(PY) -m src.data.cli build

data: ingest build-data
```

- [ ] **Step 7: Verify the split excludes heavy packages**

```bash
grep -cE "torch|prophet|mlflow|evidently" requirements-serving.txt
```

Expected: `0`

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements-serving.txt setup.cfg makefile
git commit -m "build: split serving deps from training deps, declare evidently and pandera

Fixes B2 (evidently imported in src/monitoring/check_iqa.py but never
declared). Serving image now installs neither torch nor prophet."
```

---

## Task 1: HTTP layer with browser User-Agent and retry

Without this every download returns `403 RBAC: access denied`. This is the single highest-value module in the plan.

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/http.py`
- Test: `tests/data/__init__.py`, `tests/data/test_http.py`

- [ ] **Step 1: Create package directories**

```bash
mkdir -p src/data tests/data
touch src/data/__init__.py tests/data/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/data/test_http.py`:

```python
import pytest

from src.data.http import BROWSER_UA, build_session, download


def test_session_sends_browser_user_agent():
    session = build_session()
    assert "Mozilla/5.0" in session.headers["User-Agent"]
    assert session.headers["User-Agent"] == BROWSER_UA


def test_download_writes_file_and_creates_parents(tmp_path, monkeypatch):
    dest = tmp_path / "nested" / "out.csv"

    class FakeResponse:
        status_code = 200
        headers = {"ETag": '"abc123"'}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"col\n"
            yield b"1\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeSession:
        headers = {"User-Agent": BROWSER_UA}

        def get(self, url, stream=False, timeout=None):
            return FakeResponse()

    result = download("https://example.invalid/x.csv", dest, session=FakeSession())

    assert result == dest
    assert dest.read_text() == "col\n1\n"


def test_download_skips_when_etag_unchanged(tmp_path):
    dest = tmp_path / "out.csv"
    dest.write_text("cached\n")
    etag_file = dest.with_suffix(dest.suffix + ".etag")
    etag_file.write_text('"abc123"')

    class NotModifiedResponse:
        status_code = 304
        headers = {}

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeSession:
        headers = {"User-Agent": BROWSER_UA}
        seen_headers = {}

        def get(self, url, stream=False, timeout=None, headers=None):
            FakeSession.seen_headers = headers or {}
            return NotModifiedResponse()

    result = download("https://example.invalid/x.csv", dest, session=FakeSession())

    assert result == dest
    assert dest.read_text() == "cached\n"
    assert FakeSession.seen_headers.get("If-None-Match") == '"abc123"'
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_http.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.http'`

- [ ] **Step 4: Implement `src/data/http.py`**

```python
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


def build_session(total_retries: int = 4, backoff_factor: float = 1.5) -> requests.Session:
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
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_http.py -v
```

Expected: 3 passed

- [ ] **Step 6: Prove the UA fix against the live portal**

This is the one manual network check in the plan. It confirms the 403 diagnosis.

```bash
.venv/bin/python - <<'EOF'
import requests
from src.data.http import build_session
url = ("https://donnees.montreal.ca/dataset/ae01f7f3-4d69-404a-9be1-74abfdc96571/"
       "resource/29db5545-89a4-4e4a-9e95-05aa6dc2fd80/download/"
       "liste-des-stations_do_rev_2026-05-22.csv")
print("bare requests :", requests.get(url, timeout=60).status_code)
print("our session   :", build_session().get(url, timeout=60).status_code)
EOF
```

Expected: `bare requests : 403` and `our session   : 200`

- [ ] **Step 7: Commit**

```bash
git add src/data/__init__.py src/data/http.py tests/data/
git commit -m "feat(data): add HTTP layer with browser UA, retry and ETag caching

The Montreal open-data portal returns 403 'RBAC: access denied' to
non-browser User-Agents, which breaks pd.read_csv(url) and bare
requests.get. Centralise the workaround in one module."
```

---

## Task 2: Source registry and CKAN resolution

Implements decision **D8**: the stations filename embeds a revision date (`liste-des-stations_do_rev_2026-05-22.csv`) and will change, so URLs are resolved through the API rather than hardcoded.

**Files:**
- Create: `src/data/sources.py`
- Create: `src/data/ckan.py`
- Test: `tests/data/test_ckan.py`

- [ ] **Step 1: Write `src/data/sources.py`**

No logic lives here — it is the single place where a resource ID is written down.

```python
"""Registry of Montreal open-data resources used by this project.

Resource IDs are stable; download URLs are not (the stations file embeds a
revision date in its filename). Always resolve through :mod:`src.data.ckan`.
"""

from __future__ import annotations

from dataclasses import dataclass

CKAN_BASE = "https://donnees.montreal.ca/api/3/action"


@dataclass(frozen=True)
class Resource:
    """One CKAN resource.

    ``fallback_url`` is used only if the CKAN API is unreachable. It is
    ``None`` where the filename is known to change between revisions.
    """

    name: str
    dataset_id: str
    resource_id: str
    fallback_url: str | None = None


_IQA_DATASET = "547b8052-1710-4d69-8760-beaa3aa35ec6"
_RT_DATASET = "3e9f7b96-3f25-4404-a5ad-22d9a31060e6"
_STATIONS_DATASET = "ae01f7f3-4d69-404a-9be1-74abfdc96571"

IQA_HISTORICAL: tuple[Resource, ...] = (
    Resource(
        name="iqa_2022_2024",
        dataset_id=_IQA_DATASET,
        resource_id="0c325562-e742-4e8e-8c36-971f3c9e58cd",
        fallback_url=(
            f"https://donnees.montreal.ca/dataset/{_IQA_DATASET}/resource/"
            "0c325562-e742-4e8e-8c36-971f3c9e58cd/download/"
            "rsqa-indice-qualite-air-2022-2024.csv"
        ),
    ),
    Resource(
        name="iqa_2025_2027",
        dataset_id=_IQA_DATASET,
        resource_id="6cf08815-49d2-4d2f-a400-ce36ee52b0fc",
        fallback_url=(
            f"https://donnees.montreal.ca/dataset/{_IQA_DATASET}/resource/"
            "6cf08815-49d2-4d2f-a400-ce36ee52b0fc/download/"
            "rsqa-indice-qualite-air-2025-2027.csv"
        ),
    ),
)

IQA_REALTIME = Resource(
    name="iqa_realtime",
    dataset_id=_RT_DATASET,
    resource_id="6554355e-63d1-4a01-a268-91e0763c3606",
    fallback_url=(
        f"https://donnees.montreal.ca/dataset/{_RT_DATASET}/resource/"
        "6554355e-63d1-4a01-a268-91e0763c3606/download/iqa-by-station.csv"
    ),
)

# No fallback: the filename carries a revision date and changes (decision D8).
STATIONS = Resource(
    name="stations",
    dataset_id=_STATIONS_DATASET,
    resource_id="29db5545-89a4-4e4a-9e95-05aa6dc2fd80",
    fallback_url=None,
)

# Station IDs that actually report IQA, verified 2026-09-14.
KNOWN_STATION_IDS: tuple[int, ...] = (3, 6, 17, 28, 31, 50, 55, 66, 80, 99, 103)

POLLUTANTS: tuple[str, ...] = ("PM", "O3", "NO2", "SO2", "CO")

# Montreal IQA scale: 1-25 Bon, 26-50 Acceptable, >50 Mauvais.
IQA_POOR_THRESHOLD = 50
```

- [ ] **Step 2: Write the failing test**

Create `tests/data/test_ckan.py`:

```python
import pytest

from src.data import sources
from src.data.ckan import CkanError, resolve


def _fake_session(payload, status=200):
    class FakeResponse:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        def json(self):
            return payload

    class FakeSession:
        def get(self, url, timeout=None, params=None):
            FakeSession.last_params = params
            return FakeResponse()

    return FakeSession()


def test_resolve_returns_url_and_last_modified():
    payload = {
        "result": {
            "resources": [
                {"id": "other-id", "url": "https://x/other.csv", "last_modified": None},
                {
                    "id": "29db5545-89a4-4e4a-9e95-05aa6dc2fd80",
                    "url": "https://x/liste-des-stations_do_rev_2026-05-22.csv",
                    "last_modified": "2026-05-22T05:00:00",
                },
            ]
        }
    }
    resolved = resolve(sources.STATIONS, session=_fake_session(payload))
    assert resolved.url.endswith("liste-des-stations_do_rev_2026-05-22.csv")
    assert resolved.last_modified == "2026-05-22T05:00:00"


def test_resolve_falls_back_when_api_unavailable():
    class BrokenSession:
        def get(self, url, timeout=None, params=None):
            raise OSError("network down")

    resolved = resolve(sources.IQA_REALTIME, session=BrokenSession())
    assert resolved.url == sources.IQA_REALTIME.fallback_url
    assert resolved.last_modified is None


def test_resolve_raises_when_no_fallback_and_api_unavailable():
    class BrokenSession:
        def get(self, url, timeout=None, params=None):
            raise OSError("network down")

    with pytest.raises(CkanError, match="no fallback"):
        resolve(sources.STATIONS, session=BrokenSession())


def test_resolve_raises_when_resource_missing_from_payload():
    payload = {"result": {"resources": [{"id": "nope", "url": "https://x/a.csv"}]}}
    with pytest.raises(CkanError, match="not found"):
        resolve(sources.STATIONS, session=_fake_session(payload))
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_ckan.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.ckan'`

- [ ] **Step 4: Implement `src/data/ckan.py`**

```python
"""Resolve CKAN resource IDs to current download URLs."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.http import build_session
from src.data.sources import CKAN_BASE, Resource


class CkanError(RuntimeError):
    """Raised when a resource cannot be resolved."""


@dataclass(frozen=True)
class ResolvedResource:
    name: str
    url: str
    last_modified: str | None


def resolve(resource: Resource, session=None, timeout: int = 60) -> ResolvedResource:
    """Return the current download URL for ``resource``.

    Falls back to ``resource.fallback_url`` when the API cannot be reached.
    Raises :class:`CkanError` when there is no fallback to fall back to.
    """
    session = session or build_session()
    try:
        response = session.get(
            f"{CKAN_BASE}/package_show",
            params={"id": resource.dataset_id},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # network, DNS, JSON, HTTP status
        if resource.fallback_url:
            return ResolvedResource(resource.name, resource.fallback_url, None)
        raise CkanError(
            f"Could not resolve resource {resource.name} "
            f"({resource.resource_id}) and no fallback URL is configured"
        ) from exc

    for entry in payload.get("result", {}).get("resources", []):
        if entry.get("id") == resource.resource_id:
            url = entry.get("url")
            if not url:
                raise CkanError(f"Resource {resource.name} has no url field")
            return ResolvedResource(resource.name, url, entry.get("last_modified"))

    raise CkanError(
        f"Resource {resource.resource_id} not found in dataset {resource.dataset_id}"
    )
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_ckan.py -v
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/data/sources.py src/data/ckan.py tests/data/test_ckan.py
git commit -m "feat(data): add source registry and CKAN URL resolution

Station file names embed a revision date, so URLs are resolved through
the CKAN API with hardcoded fallbacks only where filenames are stable."
```

---

## Task 3: Test fixtures captured from the real sources

Every later test runs offline against these. They deliberately include the dirty rows from the real stations file.

**Files:**
- Create: `scripts/make_fixtures.py`
- Create: `tests/fixtures/iqa_historical_sample.csv`
- Create: `tests/fixtures/iqa_realtime_sample.csv`
- Create: `tests/fixtures/stations_sample.csv`
- Create: `tests/fixtures/weather_archive_sample.json`

- [ ] **Step 1: Write the fixture generator**

Create `scripts/make_fixtures.py`. It is run once, by hand, and its output is committed.

```python
"""Regenerate test fixtures from the live sources.

Run manually:  .venv/bin/python -m scripts.make_fixtures
The generated files are committed; tests never hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from src.data import sources
from src.data.ckan import resolve
from src.data.http import build_session, download

OUT = Path("tests/fixtures")
RAW = Path("data/_fixture_raw")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    session = build_session()

    # Historical IQA: keep two stations across a DST transition so the
    # timezone tests have something real to assert against.
    hist = resolve(sources.IQA_HISTORICAL[0], session=session)
    hist_path = download(hist.url, RAW / "iqa_hist.csv", session=session)
    df = pd.read_csv(hist_path)
    sample = df[
        df["stationId"].isin([3, 6])
        & df["date"].isin(["2024-03-09", "2024-03-10", "2024-03-11", "2024-11-03"])
    ]
    sample.to_csv(OUT / "iqa_historical_sample.csv", index=False)
    print("historical sample rows:", len(sample))

    # Realtime IQA: whole file, it is tiny.
    rt = resolve(sources.IQA_REALTIME, session=session)
    rt_path = download(rt.url, RAW / "iqa_rt.csv", session=session)
    rt_df = pd.read_csv(rt_path)
    rt_df.head(200).to_csv(OUT / "iqa_realtime_sample.csv", index=False)
    print("realtime sample rows:", min(len(rt_df), 200))

    # Stations: keep the file verbatim, including the corrupt row and the
    # trailing all-NaN rows. The contract test depends on them being present.
    st = resolve(sources.STATIONS, session=session)
    st_path = download(st.url, RAW / "stations.csv", session=session)
    (OUT / "stations_sample.csv").write_bytes(st_path.read_bytes())
    print("stations bytes:", st_path.stat().st_size)

    # Open-Meteo archive response for one cell, two days.
    params = {
        "latitude": 45.5,
        "longitude": -73.6,
        "start_date": "2024-06-01",
        "end_date": "2024-06-02",
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "surface_pressure",
            ]
        ),
        "timezone": "UTC",
    }
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90
    )
    resp.raise_for_status()
    (OUT / "weather_archive_sample.json").write_text(json.dumps(resp.json(), indent=1))
    print("weather hours:", len(resp.json()["hourly"]["time"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixtures**

```bash
.venv/bin/python -m scripts.make_fixtures
```

Expected output, approximately:
```
historical sample rows: 480
realtime sample rows: 200
stations bytes: 3933
weather hours: 48
```

- [ ] **Step 3: Verify the stations fixture still contains the defects the contract must catch**

```bash
.venv/bin/python - <<'EOF'
import pandas as pd
s = pd.read_csv("tests/fixtures/stations_sample.csv", encoding="utf-8-sig")
print("rows:", len(s))
print("all-NaN rows:", int(s.isna().all(axis=1).sum()))
bad = s[s["latitude"] > 90]
print("out-of-range latitudes:", len(bad), bad["numero_station"].tolist())
EOF
```

Expected: `rows: 37`, `all-NaN rows: 5`, `out-of-range latitudes: 1 [62.0]`

If the corrupt row is absent the upstream file has been fixed; in that case add a synthetic corrupt row to the fixture so the contract test keeps its teeth, and note it in a comment at the top of the fixture generator.

- [ ] **Step 4: Ignore the raw scratch directory**

Append to `.gitignore`:

```
data/_fixture_raw/
*.etag
*.part
```

- [ ] **Step 5: Commit the fixtures**

```bash
git add scripts/make_fixtures.py tests/fixtures/ .gitignore
git commit -m "test: capture offline fixtures from the live RSQA and Open-Meteo sources

Fixtures deliberately retain the stations file's defects (BOM, 5 all-NaN
rows, station 62 latitude 4.5e7) so the contract tests have real input."
```

---

## Task 4: Data contracts

These replace the alias-guessing in `src/features/build_features.py:119-161`, which silently falls back to "the first numeric column".

**Files:**
- Create: `src/data/contracts.py`
- Test: `tests/data/test_contracts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_contracts.py`:

```python
import pandas as pd
import pytest

from src.data.contracts import (
    ContractError,
    RAW_IQA_HISTORICAL,
    RAW_STATIONS,
    validate,
)

FIXTURES = "tests/fixtures"


def test_historical_fixture_passes_contract():
    df = pd.read_csv(f"{FIXTURES}/iqa_historical_sample.csv")
    out = validate(df, RAW_IQA_HISTORICAL)
    assert len(out) == len(df)


def test_hour_out_of_range_is_rejected():
    df = pd.DataFrame(
        {
            "stationId": [3],
            "polluant": ["PM"],
            "valeur": [10],
            "date": ["2024-01-01"],
            "heure": [24],
        }
    )
    with pytest.raises(ContractError):
        validate(df, RAW_IQA_HISTORICAL)


def test_negative_value_is_rejected():
    df = pd.DataFrame(
        {
            "stationId": [3],
            "polluant": ["PM"],
            "valeur": [-1],
            "date": ["2024-01-01"],
            "heure": [5],
        }
    )
    with pytest.raises(ContractError):
        validate(df, RAW_IQA_HISTORICAL)


def test_stations_contract_rejects_corrupt_latitude():
    """Station 62 has latitude 4.504576e+07 in the real file."""
    df = pd.read_csv(f"{FIXTURES}/stations_sample.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["numero_station"])
    with pytest.raises(ContractError, match="latitude"):
        validate(df, RAW_STATIONS)


def test_stations_contract_passes_after_dropping_bad_rows():
    df = pd.read_csv(f"{FIXTURES}/stations_sample.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["numero_station"])
    df = df[(df["latitude"].between(45.2, 45.8)) & (df["longitude"].between(-74.1, -73.4))]
    out = validate(df, RAW_STATIONS)
    assert len(out) > 10
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_contracts.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.contracts'`

- [ ] **Step 3: Implement `src/data/contracts.py`**

The import shim keeps this working across the pandera range pinned in Task 0.

```python
"""pandera contracts for every external data boundary.

A contract failure is a loud, actionable error naming the offending rows.
This deliberately replaces column-name guessing, which fails silently.
"""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.23
    from pandera.pandas import Check, Column, DataFrameSchema
    from pandera.errors import SchemaError, SchemaErrors
except ImportError:  # pandera < 0.23
    from pandera import Check, Column, DataFrameSchema
    from pandera.errors import SchemaError, SchemaErrors

# Montreal island bounding box, generous by ~0.1 degrees.
LAT_MIN, LAT_MAX = 45.2, 45.8
LON_MIN, LON_MAX = -74.1, -73.4

VALID_POLLUTANTS = ["PM", "O3", "NO2", "SO2", "CO"]


class ContractError(ValueError):
    """Raised when a frame violates its contract."""


RAW_IQA_HISTORICAL = DataFrameSchema(
    {
        "stationId": Column(int, Check.ge(0), nullable=False),
        "polluant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "valeur": Column(int, [Check.ge(0), Check.le(1000)], nullable=False),
        "date": Column(str, nullable=False),
        "heure": Column(int, [Check.ge(0), Check.le(23)], nullable=False),
    },
    strict=False,
    coerce=True,
    name="RAW_IQA_HISTORICAL",
)

RAW_IQA_REALTIME = DataFrameSchema(
    {
        "stationId": Column(int, Check.ge(0), nullable=False),
        "pollutant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "valeur": Column(int, [Check.ge(0), Check.le(1000)], nullable=False),
        "date": Column(str, nullable=False),
        "heure": Column(int, [Check.ge(0), Check.le(23)], nullable=False),
    },
    strict=False,
    coerce=True,
    name="RAW_IQA_REALTIME",
)

RAW_STATIONS = DataFrameSchema(
    {
        "numero_station": Column("Int64", Check.ge(0), nullable=False),
        "nom": Column(str, nullable=True),
        "latitude": Column(float, Check.in_range(LAT_MIN, LAT_MAX), nullable=False),
        "longitude": Column(float, Check.in_range(LON_MIN, LON_MAX), nullable=False),
    },
    strict=False,
    coerce=True,
    name="RAW_STATIONS",
)

SILVER_IQA_HOURLY = DataFrameSchema(
    {
        "station_id": Column("int16", nullable=False),
        "ts_utc": Column("datetime64[ns, UTC]", nullable=False),
        "pollutant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "value": Column("int16", Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
    unique=["station_id", "ts_utc", "pollutant"],
    name="SILVER_IQA_HOURLY",
)

SILVER_WEATHER_HOURLY = DataFrameSchema(
    {
        "cell_id": Column(str, nullable=False),
        "ts_utc": Column("datetime64[ns, UTC]", nullable=False),
        "wind_direction_10m": Column(
            float, Check.in_range(0, 360), nullable=True
        ),
        "precipitation": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=True,
    unique=["cell_id", "ts_utc"],
    name="SILVER_WEATHER_HOURLY",
)

GOLD_DAILY_STATION_IQA = DataFrameSchema(
    {
        "station_id": Column("int16", nullable=False),
        "date_local": Column("datetime64[ns]", nullable=False),
        "iqa": Column("int16", Check.ge(0), nullable=False),
        "driving_pollutant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "n_hours_observed": Column(
            "int8", [Check.ge(1), Check.le(24)], nullable=False
        ),
    },
    strict=False,
    coerce=True,
    unique=["station_id", "date_local"],
    name="GOLD_DAILY_STATION_IQA",
)


def validate(df: pd.DataFrame, schema: DataFrameSchema) -> pd.DataFrame:
    """Validate ``df`` against ``schema``, raising :class:`ContractError`.

    Wrapping pandera's exceptions keeps callers free of pandera imports and
    guarantees the schema name appears in the message.
    """
    try:
        return schema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise ContractError(f"{schema.name} contract violated:\n{exc}") from exc


def check_freshness(newest: pd.Timestamp, max_age_days: int = 2) -> str | None:
    """Return a warning string when data is staler than ``max_age_days``.

    This is the check that surfaces the 8-month staleness of the historical
    dump described in the spec, section 2.2.
    """
    now = pd.Timestamp.now(tz="UTC").normalize()
    newest = pd.Timestamp(newest)
    if newest.tzinfo is None:
        newest = newest.tz_localize("UTC")
    age = (now - newest.normalize()).days
    if age > max_age_days:
        return f"Data is {age} days old (newest={newest.date()}, allowed={max_age_days})"
    return None
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_contracts.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/contracts.py tests/data/test_contracts.py
git commit -m "feat(data): add pandera contracts for every external boundary

Replaces silent column-name guessing. The stations contract rejects the
real corrupt latitude (station 62, 4.5e7) rather than ingesting it."
```

---

## Task 5: Ingestion watermarks

**Files:**
- Create: `src/data/watermark.py`
- Test: `tests/data/test_watermark.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_watermark.py`:

```python
from src.data.watermark import read_watermarks, should_skip, write_watermark


def test_roundtrip(tmp_path):
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "iqa_2022_2024", "2024-12-31T05:02:04")
    assert read_watermarks(path)["iqa_2022_2024"] == "2024-12-31T05:02:04"


def test_write_preserves_other_keys(tmp_path):
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "a", "1")
    write_watermark(path, "b", "2")
    marks = read_watermarks(path)
    assert marks == {"a": "1", "b": "2"}


def test_should_skip_only_when_token_matches(tmp_path):
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "iqa", "2024-12-31T05:02:04")
    assert should_skip(path, "iqa", "2024-12-31T05:02:04") is True
    assert should_skip(path, "iqa", "2026-01-19T05:02:19") is False
    assert should_skip(path, "other", "anything") is False


def test_should_never_skip_on_missing_token(tmp_path):
    """A resource with no last_modified must always be re-fetched."""
    path = tmp_path / "_watermarks.json"
    write_watermark(path, "iqa", "2024-12-31T05:02:04")
    assert should_skip(path, "iqa", None) is False


def test_read_missing_file_returns_empty(tmp_path):
    assert read_watermarks(tmp_path / "nope.json") == {}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_watermark.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.watermark'`

- [ ] **Step 3: Implement `src/data/watermark.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_watermark.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/watermark.py tests/data/test_watermark.py
git commit -m "feat(data): add watermark sidecar for idempotent ingestion"
```

---

## Task 6: RSQA ingestion — historical backfill and realtime increment

Fixes blocker **B1**. Implements decision **D3** (fixed-offset EST parsing) — the most defect-prone part of this plan.

**Files:**
- Create: `src/data/rsqa_ingest.py`
- Test: `tests/data/test_rsqa_ingest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_rsqa_ingest.py`. The DST assertions are the point of this test — `tz_localize("America/Montreal")` would raise on the 02:00 row.

```python
import pandas as pd
import pytest

from src.data.rsqa_ingest import normalize_iqa, to_bronze

FIXTURES = "tests/fixtures"


def _historical():
    return pd.read_csv(f"{FIXTURES}/iqa_historical_sample.csv")


def test_normalize_historical_produces_canonical_columns():
    out = normalize_iqa(_historical(), source="historical")
    assert list(out.columns) == [
        "station_id",
        "ts_utc",
        "ts_local",
        "pollutant",
        "value",
        "source",
    ]
    assert str(out["ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert (out["source"] == "historical").all()


def test_spring_forward_hour_two_survives_and_maps_to_edt():
    """02:00 EST on 2024-03-10 is a real observation and must not be dropped.

    Localizing directly to America/Montreal would raise NonExistentTimeError.
    02:00 EST == 07:00 UTC == 03:00 EDT.
    """
    out = normalize_iqa(_historical(), source="historical")
    row = out[
        (out["station_id"] == 3)
        & (out["pollutant"] == "O3")
        & (out["ts_utc"] == pd.Timestamp("2024-03-10 07:00", tz="UTC"))
    ]
    assert len(row) == 1
    assert row["ts_local"].dt.hour.iloc[0] == 3


def test_hour_before_transition_stays_in_est():
    out = normalize_iqa(_historical(), source="historical")
    row = out[
        (out["station_id"] == 3)
        & (out["pollutant"] == "O3")
        & (out["ts_utc"] == pd.Timestamp("2024-03-10 06:00", tz="UTC"))
    ]
    assert len(row) == 1
    assert row["ts_local"].dt.hour.iloc[0] == 1


def test_realtime_english_pollutant_column_is_accepted():
    rt = pd.read_csv(f"{FIXTURES}/iqa_realtime_sample.csv")
    out = normalize_iqa(rt, source="realtime")
    assert "pollutant" in out.columns
    assert (out["source"] == "realtime").all()
    assert len(out) == len(rt)


def test_normalize_rejects_frame_with_neither_spelling():
    df = pd.DataFrame(
        {"stationId": [3], "valeur": [1], "date": ["2024-01-01"], "heure": [0]}
    )
    with pytest.raises(ValueError, match="pollutant"):
        normalize_iqa(df, source="historical")


def test_to_bronze_is_idempotent(tmp_path):
    out = normalize_iqa(_historical(), source="historical")
    p1 = to_bronze(out, tmp_path, partition="year=2024")
    first = pd.read_parquet(p1)
    p2 = to_bronze(out, tmp_path, partition="year=2024")
    second = pd.read_parquet(p2)
    assert p1 == p2
    assert len(first) == len(second)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_rsqa_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.rsqa_ingest'`

- [ ] **Step 3: Implement `src/data/rsqa_ingest.py`**

```python
"""Ingest RSQA air-quality index data into the bronze layer.

Two sources, one schema:

* historical annual dumps  (column ``polluant``,  updated ~yearly)
* the realtime feed        (column ``pollutant``, updated daily)

Timestamps are UTC-5 year-round, NOT local-with-DST. See spec section 2.5b:
02:00 exists on spring-forward days, so ``tz_localize("America/Montreal")``
would raise NonExistentTimeError. Localize to the fixed offset instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import sources
from src.data.ckan import resolve
from src.data.contracts import (
    RAW_IQA_HISTORICAL,
    RAW_IQA_REALTIME,
    validate,
)
from src.data.http import build_session, download
from src.data.watermark import should_skip, write_watermark

SOURCE_TZ = "Etc/GMT+5"  # POSIX sign inversion: this is UTC-5.
CIVIL_TZ = "America/Montreal"

CANONICAL_COLUMNS = [
    "station_id",
    "ts_utc",
    "ts_local",
    "pollutant",
    "value",
    "source",
]


def normalize_iqa(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize either raw IQA schema to the canonical silver shape."""
    if "polluant" in df.columns:
        pollutant_col = "polluant"
        schema = RAW_IQA_HISTORICAL
    elif "pollutant" in df.columns:
        pollutant_col = "pollutant"
        schema = RAW_IQA_REALTIME
    else:
        raise ValueError(
            "Frame has neither 'polluant' nor 'pollutant'; "
            f"got columns {list(df.columns)}"
        )

    validate(df, schema)

    ts_naive = pd.to_datetime(df["date"], format="%Y-%m-%d") + pd.to_timedelta(
        df["heure"].astype(int), unit="h"
    )
    ts_utc = ts_naive.dt.tz_localize(SOURCE_TZ).dt.tz_convert("UTC")

    out = pd.DataFrame(
        {
            "station_id": df["stationId"].astype("int16"),
            "ts_utc": ts_utc,
            "ts_local": ts_utc.dt.tz_convert(CIVIL_TZ),
            "pollutant": df[pollutant_col].astype(str).str.strip().str.upper(),
            "value": df["valeur"].astype("int16"),
            "source": source,
        }
    )
    return out[CANONICAL_COLUMNS]


def to_bronze(df: pd.DataFrame, bronze_dir: Path, partition: str) -> Path:
    """Write ``df`` to ``bronze_dir/partition/part.parquet``, overwriting.

    Overwrite rather than append is what makes re-running a day idempotent.
    """
    out_dir = Path(bronze_dir) / partition
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part.parquet"
    df.to_parquet(path, index=False)
    return path


def ingest_historical(
    bronze_root: Path = Path("data/bronze/rsqa_iqa/historical"),
    watermark_path: Path = Path("data/_watermarks.json"),
    raw_dir: Path = Path("data/_raw"),
    session=None,
    force: bool = False,
) -> list[Path]:
    """Download and normalize the annual historical dumps."""
    session = session or build_session()
    written: list[Path] = []

    for resource in sources.IQA_HISTORICAL:
        resolved = resolve(resource, session=session)
        if not force and should_skip(watermark_path, resource.name, resolved.last_modified):
            print(f"[skip] {resource.name} unchanged ({resolved.last_modified})")
            continue

        raw_path = download(
            resolved.url, Path(raw_dir) / f"{resource.name}.csv", session=session
        )
        raw = pd.read_csv(raw_path)
        normalized = normalize_iqa(raw, source="historical")
        path = to_bronze(normalized, bronze_root, partition=f"resource={resource.name}")
        written.append(path)
        print(f"[ok]   {resource.name}: {len(normalized):,} rows -> {path}")

        if resolved.last_modified:
            write_watermark(watermark_path, resource.name, resolved.last_modified)

    return written


def ingest_realtime(
    bronze_root: Path = Path("data/bronze/rsqa_iqa/realtime"),
    watermark_path: Path = Path("data/_watermarks.json"),
    raw_dir: Path = Path("data/_raw"),
    session=None,
) -> Path | None:
    """Fetch today's realtime slice. Safe to run many times per day."""
    session = session or build_session()
    resolved = resolve(sources.IQA_REALTIME, session=session)

    raw_path = download(
        resolved.url, Path(raw_dir) / "iqa_realtime.csv", session=session
    )
    raw = pd.read_csv(raw_path)
    if raw.empty:
        print("[warn] realtime feed returned no rows")
        return None

    normalized = normalize_iqa(raw, source="realtime")
    day = normalized["ts_local"].dt.date.max()
    path = to_bronze(normalized, bronze_root, partition=f"date={day}")
    write_watermark(watermark_path, "iqa_realtime", str(day))
    print(f"[ok]   realtime {day}: {len(normalized):,} rows -> {path}")
    return path
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_rsqa_ingest.py -v
```

Expected: 6 passed

- [ ] **Step 5: Run a real historical ingest**

```bash
.venv/bin/python -c "
from src.data.rsqa_ingest import ingest_historical
ingest_historical()
"
```

Expected: two `[ok]` lines totalling roughly 1,335,000 rows. Re-running immediately should print two `[skip]` lines — that is idempotency working.

- [ ] **Step 6: Commit**

```bash
git add src/data/rsqa_ingest.py tests/data/test_rsqa_ingest.py
git commit -m "feat(data): commit the RSQA ingestion layer (fixes B1)

Handles both source schemas and parses timestamps as fixed-offset UTC-5.
The source is EST year-round, so localizing to America/Montreal would
raise NonExistentTimeError on every spring-forward date."
```

---

## Task 7: Station dimension with weather cell assignment

Delivers feature ⑤'s geographic foundation and implements decisions **D6** and **D9**.

**Files:**
- Create: `src/data/stations.py`
- Test: `tests/data/test_stations.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_stations.py`:

```python
import pandas as pd

from src.data.sources import KNOWN_STATION_IDS
from src.data.stations import assign_cell, load_stations

FIXTURES = "tests/fixtures"


def _raw():
    return pd.read_csv(f"{FIXTURES}/stations_sample.csv", encoding="utf-8-sig")


def test_load_drops_all_nan_rows_and_corrupt_coordinates():
    out = load_stations(_raw())
    assert out["latitude"].between(45.2, 45.8).all()
    assert out["longitude"].between(-74.1, -73.4).all()
    assert 62 not in out["station_id"].tolist()


def test_every_known_iqa_station_is_present():
    out = load_stations(_raw())
    missing = set(KNOWN_STATION_IDS) - set(out["station_id"])
    assert missing == set(), f"missing geo for stations {sorted(missing)}"


def test_closed_stations_are_not_filtered_out():
    """statut is stale: 28, 50 and 66 report data yet are marked 'ferme'."""
    out = load_stations(_raw())
    for station_id in (28, 50, 66):
        assert station_id in out["station_id"].tolist()


def test_assign_cell_rounds_to_tenth_of_a_degree():
    assert assign_cell(45.5622, -73.5718) == "45.6_-73.6"
    assert assign_cell(45.4265, -73.9289) == "45.4_-73.9"


def test_known_stations_collapse_to_seven_cells():
    out = load_stations(_raw())
    known = out[out["station_id"].isin(KNOWN_STATION_IDS)]
    assert known["cell_id"].nunique() == 7
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_stations.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.stations'`

- [ ] **Step 3: Implement `src/data/stations.py`**

```python
"""Station dimension: geography, names, and weather-cell assignment.

The upstream file is dirty (spec section 2.6): UTF-8 BOM, a duplicated
header fragment, five trailing all-NaN rows, and station 62 carrying
latitude 4.504576e+07. Cleaning happens here, once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import sources
from src.data.ckan import resolve
from src.data.contracts import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW_STATIONS, validate
from src.data.http import build_session, download

CELL_RESOLUTION = 0.1  # ERA5-Land grid; finer rounding would duplicate fetches.


def assign_cell(latitude: float, longitude: float) -> str:
    """Map a coordinate to its Open-Meteo grid cell identifier."""
    lat = round(round(latitude / CELL_RESOLUTION) * CELL_RESOLUTION, 1)
    lon = round(round(longitude / CELL_RESOLUTION) * CELL_RESOLUTION, 1)
    return f"{lat:.1f}_{lon:.1f}"


def load_stations(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw stations frame.

    Note: ``statut`` is deliberately NOT used as a filter. Stations 28, 50
    and 66 are marked ``ferme`` yet report data through 2026-01-18.
    """
    df = raw.dropna(subset=["numero_station"]).copy()
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
    ]
    validate(df, RAW_STATIONS)

    out = pd.DataFrame(
        {
            "station_id": df["numero_station"].astype("int16"),
            "name": df["nom"].astype(str).str.strip(),
            "borough": df["arrondissement_ville"].astype(str).str.strip(),
            "latitude": df["latitude"].astype("float32"),
            "longitude": df["longitude"].astype("float32"),
        }
    )
    out["cell_id"] = [
        assign_cell(lat, lon)
        for lat, lon in zip(out["latitude"], out["longitude"])
    ]
    return out.drop_duplicates(subset=["station_id"]).reset_index(drop=True)


def ingest_stations(
    silver_path: Path = Path("data/silver/stations.parquet"),
    raw_dir: Path = Path("data/_raw"),
    session=None,
) -> Path:
    """Download, clean and persist the station dimension."""
    session = session or build_session()
    resolved = resolve(sources.STATIONS, session=session)
    raw_path = download(resolved.url, Path(raw_dir) / "stations.csv", session=session)

    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    stations = load_stations(raw)

    silver_path = Path(silver_path)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(silver_path, index=False)
    print(
        f"[ok]   stations: {len(stations)} rows, "
        f"{stations['cell_id'].nunique()} weather cells -> {silver_path}"
    )
    return silver_path
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_stations.py -v
```

Expected: 5 passed

If `test_known_stations_collapse_to_seven_cells` fails with a different count, the upstream coordinates moved. Print the mapping, confirm the new count is sane, and update the expected number in the test — do not change `CELL_RESOLUTION` to make the test pass.

- [ ] **Step 5: Commit**

```bash
git add src/data/stations.py tests/data/test_stations.py
git commit -m "feat(data): add station dimension with 0.1deg weather cell assignment

Cleans the upstream file's BOM, all-NaN rows and corrupt station-62
latitude. Does not filter on statut, which contradicts the measurements."
```

---

## Task 8: Open-Meteo weather ingestion

Delivers feature ②. Implements decisions **D5** (no boundary-layer height) and **D6** (fetch per cell).

**Files:**
- Create: `src/data/weather.py`
- Test: `tests/data/test_weather.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_weather.py`:

```python
import json

import pandas as pd

from src.data.weather import HOURLY_VARS, parse_openmeteo

FIXTURES = "tests/fixtures"


def _payload():
    return json.loads(open(f"{FIXTURES}/weather_archive_sample.json").read())


def test_boundary_layer_height_is_not_requested():
    """Archive returns null for it, so training cannot use it (decision D5)."""
    assert "boundary_layer_height" not in HOURLY_VARS


def test_parse_produces_tz_aware_utc_rows():
    out = parse_openmeteo(_payload(), cell_id="45.5_-73.6")
    assert len(out) == 48
    assert str(out["ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert (out["cell_id"] == "45.5_-73.6").all()


def test_parse_includes_every_requested_variable():
    out = parse_openmeteo(_payload(), cell_id="45.5_-73.6")
    for var in HOURLY_VARS:
        assert var in out.columns, f"{var} missing from parsed frame"


def test_wind_direction_within_range():
    out = parse_openmeteo(_payload(), cell_id="45.5_-73.6")
    direction = out["wind_direction_10m"].dropna()
    assert direction.between(0, 360).all()


def test_parse_is_empty_safe():
    out = parse_openmeteo({"hourly": {"time": []}}, cell_id="x")
    assert out.empty
    assert "ts_utc" in out.columns
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_weather.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.weather'`

- [ ] **Step 3: Implement `src/data/weather.py`**

```python
"""Open-Meteo weather ingestion.

Training uses the archive endpoint; inference uses the forecast endpoint.
``boundary_layer_height`` is deliberately absent: the archive returns null
for it, so training on it would create train/serve skew (decision D5).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)

EMPTY_COLUMNS = ["cell_id", "ts_utc", *HOURLY_VARS]


def parse_openmeteo(payload: dict, cell_id: str) -> pd.DataFrame:
    """Convert an Open-Meteo JSON response into a tidy hourly frame."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    data = {"cell_id": cell_id, "ts_utc": pd.to_datetime(times, utc=True)}
    for var in HOURLY_VARS:
        data[var] = pd.to_numeric(pd.Series(hourly.get(var, [None] * len(times))))
    return pd.DataFrame(data)


def _cell_to_latlon(cell_id: str) -> tuple[float, float]:
    lat_str, lon_str = cell_id.split("_")
    return float(lat_str), float(lon_str)


def fetch_archive(
    cell_id: str, start: str, end: str, timeout: int = 120
) -> pd.DataFrame:
    """Fetch historical hourly weather for one grid cell."""
    lat, lon = _cell_to_latlon(cell_id)
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_openmeteo(response.json(), cell_id)


def fetch_forecast(cell_id: str, days: int = 3, timeout: int = 60) -> pd.DataFrame:
    """Fetch forecast hourly weather for one grid cell (used at inference)."""
    lat, lon = _cell_to_latlon(cell_id)
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(HOURLY_VARS),
            "forecast_days": days,
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_openmeteo(response.json(), cell_id)


def ingest_weather_archive(
    stations_path: Path = Path("data/silver/stations.parquet"),
    out_path: Path = Path("data/silver/weather_hourly.parquet"),
    start: str = "2021-12-01",
    end: str | None = None,
) -> Path:
    """Fetch archive weather for every distinct cell and persist it.

    ``start`` precedes the IQA history by a month so lag features at the
    beginning of 2022 have weather to reference.
    """
    stations = pd.read_parquet(stations_path)
    cells = sorted(stations["cell_id"].unique())
    end = end or (pd.Timestamp.utcnow() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    frames = []
    for cell_id in cells:
        frame = fetch_archive(cell_id, start, end)
        frames.append(frame)
        print(f"[ok]   weather {cell_id}: {len(frame):,} hours")

    weather = pd.concat(frames, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(out_path, index=False)
    print(f"[ok]   weather total {len(weather):,} rows, {len(cells)} cells -> {out_path}")
    return out_path
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_weather.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/weather.py tests/data/test_weather.py
git commit -m "feat(data): add Open-Meteo weather ingestion per grid cell

Archive for training, forecast for inference. boundary_layer_height is
excluded because the archive returns null for it (train/serve skew)."
```

---

## Task 9: Silver layer — normalize and reconcile

**Files:**
- Create: `src/data/aggregate.py`
- Test: `tests/data/test_aggregate_silver.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_aggregate_silver.py`:

```python
import pandas as pd

from src.data.aggregate import build_silver_iqa


def _frame(source, value, ts="2026-01-05 12:00"):
    return pd.DataFrame(
        {
            "station_id": pd.array([3], dtype="int16"),
            "ts_utc": pd.to_datetime([ts], utc=True),
            "ts_local": pd.to_datetime([ts], utc=True).tz_convert("America/Montreal"),
            "pollutant": ["PM"],
            "value": pd.array([value], dtype="int16"),
            "source": [source],
        }
    )


def test_historical_wins_over_realtime_on_conflict():
    """The annual dump is the corrected official record."""
    out = build_silver_iqa([_frame("realtime", 99), _frame("historical", 42)])
    assert len(out) == 1
    assert out["value"].iloc[0] == 42
    assert out["source"].iloc[0] == "historical"


def test_non_overlapping_rows_are_all_kept():
    out = build_silver_iqa(
        [
            _frame("historical", 10, "2026-01-05 12:00"),
            _frame("realtime", 20, "2026-01-06 12:00"),
        ]
    )
    assert len(out) == 2
    assert sorted(out["value"].tolist()) == [10, 20]


def test_output_is_sorted_and_unique():
    out = build_silver_iqa(
        [
            _frame("realtime", 20, "2026-01-06 12:00"),
            _frame("historical", 10, "2026-01-05 12:00"),
        ]
    )
    assert out["ts_utc"].is_monotonic_increasing
    assert not out.duplicated(subset=["station_id", "ts_utc", "pollutant"]).any()


def test_empty_input_returns_empty_frame():
    out = build_silver_iqa([])
    assert out.empty
    assert "station_id" in out.columns
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_aggregate_silver.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.aggregate'`

- [ ] **Step 3: Implement the silver half of `src/data/aggregate.py`**

```python
"""bronze -> silver -> gold transformations.

Silver normalizes and reconciles; gold produces one row per
(station_id, date_local) with weather, geography and targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contracts import SILVER_IQA_HOURLY, validate
from src.data.sources import IQA_POOR_THRESHOLD, POLLUTANTS

SILVER_COLUMNS = ["station_id", "ts_utc", "ts_local", "pollutant", "value", "source"]

# Historical first: the annual dump is the corrected official record and
# must win over any realtime row covering the same hour.
_SOURCE_PRIORITY = {"historical": 0, "realtime": 1}


def build_silver_iqa(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate bronze frames, reconcile overlaps, validate."""
    if not frames:
        return pd.DataFrame(columns=SILVER_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["_priority"] = combined["source"].map(_SOURCE_PRIORITY).fillna(9)
    combined = combined.sort_values(
        ["station_id", "ts_utc", "pollutant", "_priority"]
    )
    combined = combined.drop_duplicates(
        subset=["station_id", "ts_utc", "pollutant"], keep="first"
    )
    combined = combined.drop(columns="_priority").reset_index(drop=True)
    combined = combined.sort_values(["ts_utc", "station_id", "pollutant"]).reset_index(
        drop=True
    )
    validate(combined, SILVER_IQA_HOURLY)
    return combined[SILVER_COLUMNS]


def load_bronze_frames(bronze_root: Path) -> list[pd.DataFrame]:
    """Read every ``part.parquet`` under ``bronze_root``."""
    root = Path(bronze_root)
    if not root.exists():
        return []
    return [pd.read_parquet(p) for p in sorted(root.rglob("part.parquet"))]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_aggregate_silver.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/aggregate.py tests/data/test_aggregate_silver.py
git commit -m "feat(data): add silver layer with historical-wins reconciliation"
```

---

## Task 10: Gold layer — daily station IQA with weather, geography and targets

The centrepiece. Implements decisions **D1**, **D2**, **D4** and the §5 schema.

**Files:**
- Modify: `src/data/aggregate.py` (append)
- Test: `tests/data/test_aggregate_gold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_aggregate_gold.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.data.aggregate import add_targets, daily_weather, hourly_to_daily


def _hourly(rows):
    """rows: list of (station_id, 'YYYY-MM-DD HH:MM' local, pollutant, value)."""
    local = pd.to_datetime([r[1] for r in rows]).tz_localize(
        "America/Montreal", ambiguous=True, nonexistent="shift_forward"
    )
    return pd.DataFrame(
        {
            "station_id": pd.array([r[0] for r in rows], dtype="int16"),
            "ts_utc": local.tz_convert("UTC"),
            "ts_local": local,
            "pollutant": [r[2] for r in rows],
            "value": pd.array([r[3] for r in rows], dtype="int16"),
            "source": "historical",
        }
    )


def test_daily_iqa_is_the_max_across_pollutants_and_hours():
    df = _hourly(
        [
            (3, "2024-06-01 01:00", "PM", 10),
            (3, "2024-06-01 01:00", "O3", 30),
            (3, "2024-06-01 14:00", "PM", 55),
            (3, "2024-06-01 14:00", "O3", 20),
        ]
    )
    out = hourly_to_daily(df)
    assert len(out) == 1
    assert out["iqa"].iloc[0] == 55
    assert out["driving_pollutant"].iloc[0] == "PM"
    assert out["n_hours_observed"].iloc[0] == 2


def test_per_pollutant_subindices_are_daily_maxima():
    df = _hourly(
        [
            (3, "2024-06-01 01:00", "PM", 10),
            (3, "2024-06-01 14:00", "PM", 40),
            (3, "2024-06-01 14:00", "NO2", 7),
        ]
    )
    out = hourly_to_daily(df)
    assert out["sub_PM"].iloc[0] == 40
    assert out["sub_NO2"].iloc[0] == 7
    assert pd.isna(out["sub_O3"].iloc[0])


def test_targets_shift_by_one_and_two_days():
    df = pd.DataFrame(
        {
            "station_id": pd.array([3, 3, 3], dtype="int16"),
            "date_local": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
            "iqa": pd.array([10, 20, 60], dtype="int16"),
        }
    )
    out = add_targets(df)
    assert out["target_iqa_h24"].tolist()[:2] == [20, 60]
    assert out["target_iqa_h48"].iloc[0] == 60
    assert out["target_exceed_h24"].tolist()[:2] == [False, True]


def test_targets_never_cross_a_station_boundary():
    df = pd.DataFrame(
        {
            "station_id": pd.array([3, 6], dtype="int16"),
            "date_local": pd.to_datetime(["2024-06-01", "2024-06-02"]),
            "iqa": pd.array([10, 99], dtype="int16"),
        }
    )
    out = add_targets(df)
    station_3 = out[out["station_id"] == 3]
    assert pd.isna(station_3["target_iqa_h24"].iloc[0]), "station 3 borrowed station 6's value"


def test_targets_respect_calendar_gaps():
    """A missing day must not make 'tomorrow' mean three days later."""
    df = pd.DataFrame(
        {
            "station_id": pd.array([3, 3], dtype="int16"),
            "date_local": pd.to_datetime(["2024-06-01", "2024-06-04"]),
            "iqa": pd.array([10, 99], dtype="int16"),
        }
    )
    out = add_targets(df)
    assert pd.isna(out["target_iqa_h24"].iloc[0])


def test_daily_weather_encodes_wind_direction_as_unit_vector():
    ts = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "cell_id": "45.5_-73.6",
            "ts_utc": ts,
            "temperature_2m": np.linspace(10, 20, 24),
            "relative_humidity_2m": 50.0,
            "precipitation": 0.5,
            "wind_speed_10m": 10.0,
            "wind_direction_10m": 90.0,  # due east
            "surface_pressure": 1000.0,
        }
    )
    out = daily_weather(weather)
    row = out.iloc[0]
    assert row["wind_dir_sin"] == pytest.approx(1.0, abs=1e-6)
    assert row["wind_dir_cos"] == pytest.approx(0.0, abs=1e-6)
    assert row["precip_sum"] == pytest.approx(12.0)
    assert row["wind_speed_mean"] == pytest.approx(10.0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/data/test_aggregate_gold.py -v
```

Expected: FAIL — `ImportError: cannot import name 'hourly_to_daily'`

- [ ] **Step 3: Append the gold implementation to `src/data/aggregate.py`**

```python
def hourly_to_daily(silver: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly sub-indices into one row per (station_id, date_local).

    Daily IQA is the maximum over every (hour, pollutant) observation, which
    is identical to the max over hours of the hourly IQA because max is
    associative (decisions D1 and D2).
    """
    if silver.empty:
        return pd.DataFrame(columns=["station_id", "date_local", "iqa"])

    df = silver.copy()
    df["date_local"] = pd.to_datetime(df["ts_local"].dt.date)

    grouped = df.groupby(["station_id", "date_local"], sort=True)
    daily = grouped.agg(
        iqa=("value", "max"),
        n_hours_observed=("ts_utc", "nunique"),
    ).reset_index()

    # The pollutant attaining the daily maximum.
    idx = df.groupby(["station_id", "date_local"])["value"].idxmax()
    driving = df.loc[idx, ["station_id", "date_local", "pollutant"]].rename(
        columns={"pollutant": "driving_pollutant"}
    )
    daily = daily.merge(driving, on=["station_id", "date_local"], how="left")

    # Per-pollutant daily maxima as sub_<POLLUTANT> columns.
    wide = (
        df.pivot_table(
            index=["station_id", "date_local"],
            columns="pollutant",
            values="value",
            aggfunc="max",
        )
        .reindex(columns=list(POLLUTANTS))
        .add_prefix("sub_")
        .reset_index()
    )
    daily = daily.merge(wide, on=["station_id", "date_local"], how="left")

    daily["iqa"] = daily["iqa"].astype("int16")
    daily["n_hours_observed"] = daily["n_hours_observed"].clip(upper=24).astype("int8")
    return daily


def daily_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather to one row per (cell_id, date_local).

    Wind direction is reduced to a speed-weighted resultant unit vector, so
    that 359 degrees and 1 degree are neighbours rather than opposites.
    """
    if weather.empty:
        return pd.DataFrame(columns=["cell_id", "date_local"])

    df = weather.copy()
    df["date_local"] = pd.to_datetime(
        df["ts_utc"].dt.tz_convert("America/Montreal").dt.date
    )

    radians = np.radians(df["wind_direction_10m"].astype(float))
    speed = df["wind_speed_10m"].astype(float)
    df["_u"] = speed * np.sin(radians)
    df["_v"] = speed * np.cos(radians)

    out = (
        df.groupby(["cell_id", "date_local"])
        .agg(
            temp_mean=("temperature_2m", "mean"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            humidity_mean=("relative_humidity_2m", "mean"),
            precip_sum=("precipitation", "sum"),
            wind_speed_mean=("wind_speed_10m", "mean"),
            wind_speed_max=("wind_speed_10m", "max"),
            pressure_mean=("surface_pressure", "mean"),
            _u=("_u", "mean"),
            _v=("_v", "mean"),
        )
        .reset_index()
    )

    magnitude = np.hypot(out["_u"], out["_v"])
    safe = magnitude.replace(0, np.nan)
    out["wind_dir_sin"] = (out["_u"] / safe).fillna(0.0)
    out["wind_dir_cos"] = (out["_v"] / safe).fillna(0.0)
    return out.drop(columns=["_u", "_v"])


def add_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach h24/h48 regression and exceedance targets.

    Each station is reindexed onto a complete daily calendar before shifting,
    so a gap in the record never makes "tomorrow" mean several days later,
    and a shift never reaches across a station boundary (decision D4).
    """
    if daily.empty:
        return daily.assign(
            target_iqa_h24=pd.Series(dtype="Int16"),
            target_iqa_h48=pd.Series(dtype="Int16"),
            target_exceed_h24=pd.Series(dtype="boolean"),
            target_exceed_h48=pd.Series(dtype="boolean"),
        )

    pieces = []
    for station_id, group in daily.groupby("station_id", sort=True):
        group = group.sort_values("date_local").set_index("date_local")
        calendar = pd.date_range(group.index.min(), group.index.max(), freq="D")
        filled = group.reindex(calendar)

        for horizon, shift in (("h24", -1), ("h48", -2)):
            future = filled["iqa"].shift(shift)
            filled[f"target_iqa_{horizon}"] = future.astype("Int16")
            exceed = future > IQA_POOR_THRESHOLD
            filled[f"target_exceed_{horizon}"] = exceed.where(future.notna()).astype(
                "boolean"
            )

        filled = filled.loc[group.index]
        filled["station_id"] = station_id
        filled.index.name = "date_local"
        pieces.append(filled.reset_index())

    return pd.concat(pieces, ignore_index=True)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/data/test_aggregate_gold.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/aggregate.py tests/data/test_aggregate_gold.py
git commit -m "feat(data): build gold daily table with weather, geo and targets

IQA is the max over pollutants and hours. Targets are computed on a
complete per-station calendar so gaps and station boundaries are honoured."
```

---

## Task 11: Pipeline assembly and CLI

**Files:**
- Modify: `src/data/aggregate.py` (append `build_gold`)
- Create: `src/data/cli.py`
- Test: `tests/data/test_cli_smoke.py`

- [ ] **Step 1: Append `build_gold` to `src/data/aggregate.py`**

```python
def build_gold(
    silver_iqa: pd.DataFrame,
    stations: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the gold table from silver inputs."""
    daily = hourly_to_daily(silver_iqa)
    daily = daily.merge(
        stations[["station_id", "name", "borough", "latitude", "longitude", "cell_id"]],
        on="station_id",
        how="left",
    )

    if weather is not None and not weather.empty:
        daily = daily.merge(
            daily_weather(weather), on=["cell_id", "date_local"], how="left"
        )

    daily = add_targets(daily)
    return daily.sort_values(["station_id", "date_local"]).reset_index(drop=True)
```

- [ ] **Step 2: Write `src/data/cli.py`**

```python
"""Command line entry point for the data pipeline.

    python -m src.data.cli ingest    # download sources into bronze/silver
    python -m src.data.cli build     # bronze/silver -> gold
    python -m src.data.cli all       # both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.aggregate import build_gold, build_silver_iqa, load_bronze_frames
from src.data.contracts import GOLD_DAILY_STATION_IQA, check_freshness, validate
from src.data.rsqa_ingest import ingest_historical, ingest_realtime
from src.data.stations import ingest_stations
from src.data.weather import ingest_weather_archive

GOLD_PATH = Path("data/gold/daily_station_iqa.parquet")
SILVER_IQA_PATH = Path("data/silver/iqa_hourly.parquet")
STATIONS_PATH = Path("data/silver/stations.parquet")
WEATHER_PATH = Path("data/silver/weather_hourly.parquet")


def cmd_ingest(args: argparse.Namespace) -> None:
    ingest_stations(silver_path=STATIONS_PATH)
    ingest_historical(force=args.force)
    try:
        ingest_realtime()
    except Exception as exc:  # the realtime feed is known to 503 occasionally
        print(f"[warn] realtime ingest failed, continuing with history: {exc}")
    if not args.skip_weather:
        ingest_weather_archive(stations_path=STATIONS_PATH, out_path=WEATHER_PATH)


def cmd_build(args: argparse.Namespace) -> None:
    frames = load_bronze_frames(Path("data/bronze/rsqa_iqa"))
    if not frames:
        raise SystemExit("No bronze data found. Run 'python -m src.data.cli ingest'.")

    silver = build_silver_iqa(frames)
    SILVER_IQA_PATH.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(SILVER_IQA_PATH, index=False)
    print(f"[ok]   silver iqa: {len(silver):,} rows -> {SILVER_IQA_PATH}")

    stations = pd.read_parquet(STATIONS_PATH)
    weather = pd.read_parquet(WEATHER_PATH) if WEATHER_PATH.exists() else None

    gold = build_gold(silver, stations, weather)
    validate(gold, GOLD_DAILY_STATION_IQA)
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(GOLD_PATH, index=False)
    print(f"[ok]   gold: {len(gold):,} rows -> {GOLD_PATH}")

    warning = check_freshness(gold["date_local"].max())
    if warning:
        print(f"[warn] FRESHNESS: {warning}")
    else:
        print("[ok]   freshness within tolerance")


def main() -> None:
    parser = argparse.ArgumentParser("airq data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download sources into bronze/silver")
    ingest.add_argument("--force", action="store_true", help="ignore watermarks")
    ingest.add_argument("--skip-weather", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    build = sub.add_parser("build", help="bronze/silver -> gold")
    build.set_defaults(func=cmd_build)

    every = sub.add_parser("all", help="ingest then build")
    every.add_argument("--force", action="store_true")
    every.add_argument("--skip-weather", action="store_true")
    every.set_defaults(func=lambda a: (cmd_ingest(a), cmd_build(a)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the CLI smoke test**

Create `tests/data/test_cli_smoke.py`:

```python
import subprocess
import sys


def test_cli_help_lists_all_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "src.data.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for command in ("ingest", "build", "all"):
        assert command in result.stdout


def test_build_without_bronze_exits_with_actionable_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "src.data.cli", "build"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode != 0
```

- [ ] **Step 4: Run the full pipeline for real**

```bash
.venv/bin/python -m src.data.cli all
```

Expected, approximately:
```
[ok]   stations: 32 rows, 12 weather cells -> data/silver/stations.parquet
[ok]   iqa_2022_2024: 1,001,709 rows -> ...
[ok]   iqa_2025_2027: 334,255 rows -> ...
[ok]   realtime 2026-09-14: ... rows -> ...
[ok]   weather 45.4_-73.9: ... hours
...
[ok]   gold: ~16,000 rows -> data/gold/daily_station_iqa.parquet
[warn] FRESHNESS: ...
```

- [ ] **Step 5: Verify the gold table against the spec's acceptance criteria**

```bash
.venv/bin/python - <<'EOF'
import pandas as pd
g = pd.read_parquet("data/gold/daily_station_iqa.parquet")
print("rows:", len(g), "| stations:", g["station_id"].nunique())
print("date range:", g["date_local"].min().date(), "->", g["date_local"].max().date())
print("exceedance rate h24: %.2f%%" % (g["target_exceed_h24"].mean() * 100))
print("driving pollutant:\n", g["driving_pollutant"].value_counts())
print("null weather rows:", int(g["wind_dir_sin"].isna().sum()))
EOF
```

Expected: ~16,000 rows across 11 stations; exceedance rate near **2.98 %**; `PM` dominant in `driving_pollutant`. A materially different exceedance rate means the aggregation is wrong — stop and investigate rather than proceeding.

- [ ] **Step 6: Commit**

```bash
git add src/data/aggregate.py src/data/cli.py tests/data/test_cli_smoke.py
git commit -m "feat(data): add pipeline assembly and CLI entry point"
```

---

## Task 12: Remove the hardcoded Windows MLflow path

Fixes blocker **B3**.

**Files:**
- Modify: `scripts/train_daily_iqa.py:26-29`
- Modify: `src/monitoring/check_iqa.py:155`
- Test: `tests/test_no_hardcoded_paths.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_no_hardcoded_paths.py`:

```python
import re
from pathlib import Path

FORBIDDEN = re.compile(r"[A-Za-z]:[/\\]Users[/\\]", re.IGNORECASE)


def test_no_absolute_user_paths_in_source():
    offenders = []
    for path in list(Path("src").rglob("*.py")) + list(Path("scripts").rglob("*.py")):
        if FORBIDDEN.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert offenders == [], f"hardcoded user paths found in {offenders}"


def test_mlflow_uri_comes_from_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    from scripts.train_daily_iqa import resolve_tracking_uri

    assert resolve_tracking_uri(None) == "sqlite:///custom.db"


def test_mlflow_uri_defaults_to_repo_relative(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from scripts.train_daily_iqa import resolve_tracking_uri

    assert resolve_tracking_uri(None) == "sqlite:///mlflow.db"


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env.db")
    from scripts.train_daily_iqa import resolve_tracking_uri

    assert resolve_tracking_uri("sqlite:///cli.db") == "sqlite:///cli.db"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_no_hardcoded_paths.py -v
```

Expected: FAIL — `test_no_absolute_user_paths_in_source` reports `scripts/train_daily_iqa.py`, and the import tests fail with `ImportError: cannot import name 'resolve_tracking_uri'`.

- [ ] **Step 3: Add the resolver to `scripts/train_daily_iqa.py`**

Insert after the imports:

```python
import os

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def resolve_tracking_uri(explicit: str | None) -> str:
    """Resolve the MLflow tracking URI.

    Precedence: explicit argument, then MLFLOW_TRACKING_URI, then a
    repo-relative SQLite file. Never a machine-specific absolute path.
    """
    return explicit or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
```

Then replace lines 26-29:

```python
    tracking_uri = (
        args.mlflow_uri
        or "sqlite:///C:/Users/AU51870/Downloads/airq_montreal_mlops/mlflow.db"
    )
```

with:

```python
    tracking_uri = resolve_tracking_uri(args.mlflow_uri)
```

- [ ] **Step 4: Apply the same resolver in `src/monitoring/check_iqa.py`**

Replace line 155:

```python
    tracking_uri = mlflow_uri or "sqlite:///mlflow.db"
```

with:

```python
    from scripts.train_daily_iqa import resolve_tracking_uri

    tracking_uri = resolve_tracking_uri(mlflow_uri)
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_no_hardcoded_paths.py -v
grep -rn "C:/Users" src scripts || echo "clean"
```

Expected: 4 passed, then `clean`

- [ ] **Step 6: Commit**

```bash
git add scripts/train_daily_iqa.py src/monitoring/check_iqa.py tests/test_no_hardcoded_paths.py
git commit -m "fix: resolve MLflow tracking URI from env, not a hardcoded path (fixes B3)

Adds a regression test so no absolute user path can return to src/ or scripts/."
```

---

## Task 13: Rewrite feature building on top of the gold table

Deletes the alias-guessing described in the spec, section 2.4. Replaces the vacuous test — the current `test_build_features_basic` passes on an **empty** DataFrame because 10 rows minus a 24-row lag leaves nothing.

**Files:**
- Modify: `src/features/build_features.py`
- Replace: `tests/test_build_features.py`

- [ ] **Step 1: Write the failing test**

Replace the entire contents of `tests/test_build_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features_daily_iqa

LAGS = (1, 2, 3, 7, 14)


def _gold(n_days=60, n_stations=2):
    """A gold-shaped frame with enough rows to survive a 14-day lag."""
    rows = []
    for station_id in range(n_stations):
        for day in range(n_days):
            rows.append(
                {
                    "station_id": np.int16(station_id),
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": np.int16(10 + day + station_id * 100),
                    "temp_mean": 5.0 + day,
                    "wind_speed_mean": 3.0,
                    "wind_dir_sin": 0.5,
                    "wind_dir_cos": 0.5,
                    "precip_sum": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_returns_non_empty_frame():
    """Guards the defect in the previous test, which passed on zero rows."""
    out, feats = build_features_daily_iqa(_gold())
    assert len(out) > 0, "feature frame is empty; the test would be vacuous"
    assert len(feats) > 0


def test_lag_one_equals_previous_days_value():
    out, _ = build_features_daily_iqa(_gold(n_stations=1))
    out = out.sort_values("date_local").reset_index(drop=True)
    for i in range(1, len(out)):
        expected = out["iqa"].iloc[i - 1]
        assert out["lag_1"].iloc[i] == expected, f"lag_1 wrong at row {i}"


def test_lags_never_cross_station_boundaries():
    out, _ = build_features_daily_iqa(_gold(n_days=40, n_stations=2))
    for station_id, group in out.groupby("station_id"):
        group = group.sort_values("date_local").reset_index(drop=True)
        for i in range(1, len(group)):
            assert group["lag_1"].iloc[i] == group["iqa"].iloc[i - 1]


def test_all_expected_lag_columns_present():
    out, feats = build_features_daily_iqa(_gold())
    for lag in LAGS:
        assert f"lag_{lag}" in out.columns
        assert f"lag_{lag}" in feats


def test_rolling_mean_excludes_the_current_day():
    out, _ = build_features_daily_iqa(_gold(n_stations=1))
    out = out.sort_values("date_local").reset_index(drop=True)
    row = out.iloc[20]
    window = out["iqa"].iloc[14:20]
    assert row["roll_7"] == pytest.approx(window.mean(), rel=1e-6)


def test_weather_columns_are_carried_into_features():
    _, feats = build_features_daily_iqa(_gold())
    for column in ("temp_mean", "wind_speed_mean", "wind_dir_sin", "precip_sum"):
        assert column in feats, f"{column} should be a feature"


def test_no_nulls_remain():
    out, feats = build_features_daily_iqa(_gold())
    assert out[feats].isna().sum().sum() == 0


@pytest.mark.xfail(
    reason="Spec B: today's IQA is excluded from features, so this is really a "
    "two-step-ahead model. Fixed in Spec B task 'put value back'.",
    strict=True,
)
def test_current_day_iqa_is_available_as_a_feature():
    _, feats = build_features_daily_iqa(_gold())
    assert "iqa" in feats
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_build_features.py -v
```

Expected: failures on the gold-shaped input, since the current implementation expects the old ad-hoc schema.

- [ ] **Step 3: Replace `build_features_daily_iqa` in `src/features/build_features.py`**

Delete the existing `build_features_daily_iqa` (lines 110-184) and `_choose_station_key` usage within it, and put this in its place. Leave the hourly `build_features` function untouched.

```python
DAILY_LAGS = (1, 2, 3, 7, 14)

_NON_FEATURE_COLUMNS = {
    "station_id",
    "date_local",
    "iqa",
    "driving_pollutant",
    "name",
    "borough",
    "cell_id",
    "target_iqa_h24",
    "target_iqa_h48",
    "target_exceed_h24",
    "target_exceed_h48",
    "target",
}


def build_features_daily_iqa(df_daily: pd.DataFrame, lags=DAILY_LAGS):
    """Build daily features from the gold table.

    The gold table is contract-validated upstream, so this function does no
    column guessing: it requires ``station_id``, ``date_local`` and ``iqa``.
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
        and c != "source"
        and pd.api.types.is_numeric_dtype(x[c])
    ]

    x = x.dropna(subset=feature_columns + ["target"]).reset_index(drop=True)
    return x, feature_columns
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_build_features.py -v
```

Expected: 7 passed, 1 xfailed. The xfail is intentional — it documents a Spec B defect rather than hiding it.

- [ ] **Step 5: Commit**

```bash
git add src/features/build_features.py tests/test_build_features.py
git commit -m "refactor(features): build daily features from the contract-validated gold table

Deletes the column-alias guessing and the 'first numeric column' fallback.
Replaces a test that passed on an empty DataFrame with one that asserts
lag_1[i] == iqa[i-1] over 60 rows and checks station boundaries."
```

---

## Task 14: Point training at the gold table and pin the split defect

**Files:**
- Modify: `src/models/training_daily.py:18-48`
- Test: `tests/test_training_split.py`

- [ ] **Step 1: Write the test that documents the split defect**

Create `tests/test_training_split.py`:

```python
import pandas as pd
import pytest

from src.models.training_daily import _daily_df, _time_split


def test_daily_df_reads_the_gold_table(tmp_path, monkeypatch):
    gold = pd.DataFrame(
        {
            "station_id": pd.array([3, 3], dtype="int16"),
            "date_local": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "iqa": pd.array([10, 20], dtype="int16"),
        }
    )
    path = tmp_path / "gold.parquet"
    gold.to_parquet(path, index=False)
    monkeypatch.setattr("src.models.training_daily.GOLD_PATH", path)

    out = _daily_df()
    assert len(out) == 2
    assert {"station_id", "date_local", "iqa"} <= set(out.columns)


def test_daily_df_error_names_the_fix():
    with pytest.raises(FileNotFoundError, match="src.data.cli"):
        import src.models.training_daily as td

        original = td.GOLD_PATH
        td.GOLD_PATH = td.Path("data/gold/does_not_exist.parquet")
        try:
            td._daily_df()
        finally:
            td.GOLD_PATH = original


@pytest.mark.xfail(
    reason="Spec B: _time_split slices a frame sorted by [station_id, date_local], "
    "so the holdout is the last stations, not the last dates.",
    strict=True,
)
def test_time_split_holdout_starts_after_train_ends():
    df = pd.DataFrame(
        {
            "station_id": pd.array([3] * 10 + [6] * 10, dtype="int16"),
            "date_local": pd.to_datetime(
                list(pd.date_range("2024-01-01", periods=10))
                + list(pd.date_range("2024-01-01", periods=10))
            ),
        }
    ).sort_values(["station_id", "date_local"])

    train, valid = _time_split(df, ratio=0.2)
    assert train["date_local"].max() <= valid["date_local"].min()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_training_split.py -v
```

Expected: the two `_daily_df` tests fail (it still reads `data/interim/*.parquet`); the split test is `xfail`.

- [ ] **Step 3: Rewrite `_daily_df` in `src/models/training_daily.py`**

Replace the whole `_daily_df` function (lines 18-42) with:

```python
GOLD_PATH = Path("data/gold/daily_station_iqa.parquet")


def _daily_df():
    """Load the contract-validated gold daily table."""
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"{GOLD_PATH} not found. Build it with: "
            "python -m src.data.cli all"
        )
    df = pd.read_parquet(GOLD_PATH)
    return df.sort_values(["station_id", "date_local"]).reset_index(drop=True)
```

Then update the three call sites that assume the old `datetime`/`value` column names:

- `train_prophet` (line 112): `_daily_df()[["date_local", "iqa"]]`, then rename to `ds`/`y`
- `train_lstm` (line 157): `_daily_df()[["date_local", "iqa"]]`
- `train_rf` (line 62): unchanged — it delegates to `build_features_daily_iqa`

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_training_split.py -v
```

Expected: 2 passed, 1 xfailed

- [ ] **Step 5: Commit**

```bash
git add src/models/training_daily.py tests/test_training_split.py
git commit -m "refactor(models): read the gold table, pin the split defect with an xfail

The station-vs-time split bug is now a failing, documented test rather
than a silent inheritance. Spec B fixes it."
```

---

## Task 15: Fix the serving layer

Removes the silent feature fallback, loads the model once, and deletes the comment admitting production was shaped by a test.

**Files:**
- Modify: `src/serving/app.py` (full rewrite)
- Modify: `tests/test_api_basic.py`

- [ ] **Step 1: Write the failing test**

Replace `tests/test_api_basic.py`:

```python
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serving.app as serving
from src.serving.app import app, get_model


class DummyModel:
    feature_names_in_ = np.array(["lag_1", "lag_2", "temp_mean"])

    def predict(self, X):
        return np.array([42.0] * len(X))


@pytest.fixture(autouse=True)
def _override_model():
    app.dependency_overrides[get_model] = lambda: (
        DummyModel(),
        ["lag_1", "lag_2", "temp_mean"],
    )
    yield
    app.dependency_overrides.clear()


def test_health_is_ok():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_predict_returns_expected_contract():
    client = TestClient(app)
    payload = {"rows": [{"lag_1": 10, "lag_2": 12, "temp_mean": 5.0}]}
    body = client.post("/predict", json=payload).json()
    assert body["n"] == 1
    assert body["preds"] == [42.0]


def test_missing_feature_returns_422_naming_the_column():
    client = TestClient(app)
    payload = {"rows": [{"lag_1": 10}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "lag_2" in response.text


def test_extra_columns_are_ignored():
    client = TestClient(app)
    payload = {"rows": [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0, "junk": "x"}]}
    assert client.post("/predict", json=payload).status_code == 200


def test_empty_rows_returns_empty_predictions():
    client = TestClient(app)
    body = client.post("/predict", json={"rows": []}).json()
    assert body == {"n": 0, "preds": []}


def test_model_is_not_reloaded_per_request(monkeypatch):
    """The old implementation called joblib.load inside the handler."""
    calls = []
    monkeypatch.setattr(serving, "load", lambda p: calls.append(p) or DummyModel())
    app.dependency_overrides.clear()
    serving._MODEL_CACHE = None
    client = TestClient(app)
    with client:
        client.post("/predict", json={"rows": [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0}]})
        client.post("/predict", json={"rows": [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0}]})
    assert len(calls) <= 1, f"model loaded {len(calls)} times"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_api_basic.py -v
```

Expected: FAIL — `ImportError: cannot import name 'get_model'`

- [ ] **Step 3: Rewrite `src/serving/app.py`**

```python
"""FastAPI service for daily IQA prediction.

The model is loaded once at startup. A request whose columns do not match
the model's feature list is rejected with 422 naming the missing columns —
silently substituting a different feature set produces wrong numbers that
look right.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "artifacts/rf/feature_names.json")

_MODEL_CACHE: tuple[object, list[str]] | None = None


def _load_model() -> tuple[object, list[str]]:
    model = load(MODEL_PATH)
    features_file = Path(FEATURES_PATH)
    if features_file.exists():
        feature_names = json.loads(features_file.read_text())
    elif hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        raise RuntimeError(
            f"No feature list: {FEATURES_PATH} is absent and the model "
            "exposes no feature_names_in_."
        )
    return model, list(feature_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MODEL_CACHE
    try:
        _MODEL_CACHE = _load_model()
    except Exception as exc:  # keep /health usable for diagnostics
        print(f"[warn] model not loaded at startup: {exc}")
        _MODEL_CACHE = None
    yield
    _MODEL_CACHE = None


app = FastAPI(title="AirQ Montreal", version="1.0.0", lifespan=lifespan)


def get_model() -> tuple[object, list[str]]:
    """Dependency returning the cached model. Tests override this."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_model()
    return _MODEL_CACHE


class PredictRequest(BaseModel):
    rows: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest, model_bundle=Depends(get_model)):
    if not req.rows:
        return {"n": 0, "preds": []}

    model, feature_names = model_bundle
    df = pd.DataFrame(req.rows)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Request is missing required feature columns: {missing}",
        )

    preds = model.predict(df[feature_names])
    return {"n": len(preds), "preds": preds.tolist()}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_api_basic.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/serving/app.py tests/test_api_basic.py
git commit -m "fix(serving): load model once, reject unknown feature sets with 422

Removes the silent 'use every column except datetime/pollutant' fallback,
which produced confidently wrong predictions, and the comment documenting
that production behaviour was bent to satisfy a monkeypatched test."
```

---

## Task 16: Split the Docker images and put a real model in the serving image

Fixes blocker **B4**: the published image copies `artifacts/`, which `.gitignore` excludes, so it ships no model.

> **Docker is not installed on this machine.** Steps 3-5 are verified by CI in Task 17. Run steps 1-2 locally.

**Files:**
- Create: `scripts/bake_serving_model.py`
- Modify: `Dockerfile`
- Create: `Dockerfile.train`
- Modify: `.dockerignore` (create if absent)

- [ ] **Step 1: Write the model-baking script**

Create `scripts/bake_serving_model.py`. It works offline from fixtures, so the image build never needs the network.

```python
"""Train a small RandomForest so the serving image always contains a model.

Prefers the real gold table; falls back to fixtures so CI can bake a model
without network access. This is what stops the published image from being
an empty shell (blocker B4).
"""

from __future__ import annotations

import json
from pathlib import Path

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
                    "temp_mean": 5.0 + (day % 25),
                    "wind_speed_mean": 3.0 + (day % 7),
                    "wind_dir_sin": 0.3,
                    "wind_dir_cos": 0.6,
                    "precip_sum": float(day % 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    source = pd.read_parquet(GOLD) if GOLD.exists() else _synthetic_gold()
    print(f"baking from {'gold table' if GOLD.exists() else 'synthetic fixture'}")

    frame, features = build_features_daily_iqa(source)
    model = RandomForestRegressor(
        n_estimators=40, max_depth=8, random_state=42, n_jobs=1
    )
    model.fit(frame[features], frame["target"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, OUT_DIR / "model.pkl")
    (OUT_DIR / "feature_names.json").write_text(json.dumps(features))
    size_kb = (OUT_DIR / "model.pkl").stat().st_size / 1024
    print(f"wrote {OUT_DIR}/model.pkl ({size_kb:.0f} KB), {len(features)} features")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Bake a model and confirm it loads**

```bash
.venv/bin/python -m scripts.bake_serving_model
.venv/bin/python -c "
import joblib, json
m = joblib.load('artifacts/rf/model.pkl')
f = json.load(open('artifacts/rf/feature_names.json'))
print('loaded OK, features:', len(f))
"
```

Expected: a size line, then `loaded OK, features: N`

- [ ] **Step 3: Rewrite `Dockerfile` as the slim serving image**

```dockerfile
# Serving image: no torch, no prophet, no mlflow.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-serving.txt

COPY src/__init__.py ./src/
COPY src/serving ./src/serving
COPY src/features ./src/features
COPY artifacts/rf ./artifacts/rf

ENV MODEL_PATH=/app/artifacts/rf/model.pkl
ENV FEATURES_PATH=/app/artifacts/rf/feature_names.json
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`COPY artifacts/rf` now succeeds because Task 16 step 2 creates it, and CI creates it before building.

- [ ] **Step 4: Create `Dockerfile.train` for the full stack**

```dockerfile
# Training image: the full scientific stack.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serving.txt requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY orchestration ./orchestration

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "src.data.cli", "all"]
```

- [ ] **Step 5: Create `.dockerignore`**

```
.git
.venv
data
mlruns
mlflow.db
tests
docs
*.md
__pycache__
*.pyc
.pytest_cache
```

- [ ] **Step 6: Commit**

```bash
git add Dockerfile Dockerfile.train .dockerignore scripts/bake_serving_model.py
git commit -m "build: split serving and training images, bake a model into serving (fixes B4)

The published image previously copied an artifacts directory that
.gitignore excludes, so /health passed while /predict raised
FileNotFoundError on the first real request."
```

---

## Task 17: CI that proves the image works

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/deploy.yml`

- [ ] **Step 1: Rewrite `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main, master, "spec-*/**"]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: |
            requirements.txt
            requirements-serving.txt

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Lint
        run: flake8 src tests scripts

      - name: Format check
        run: black --check src tests scripts

      - name: Test
        run: pytest --cov=src/data --cov-report=term-missing

  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Bake the serving model
        run: |
          pip install -r requirements-serving.txt
          python -m scripts.bake_serving_model

      - name: Build serving image
        run: docker build -t airq-api:ci .

      - name: Report image size
        run: docker image inspect airq-api:ci --format '{{.Size}}' | awk '{print "image size MB:", $1/1024/1024}'

      - name: Smoke test the running container
        run: |
          docker run -d --name airq -p 8000:8000 airq-api:ci
          for i in $(seq 1 30); do
            if curl -sf http://localhost:8000/health > /dev/null; then break; fi
            sleep 2
          done
          curl -sf http://localhost:8000/health | grep -q '"ok"'
          FEATURES=$(python -c "import json;print(json.dumps({k:1.0 for k in json.load(open('artifacts/rf/feature_names.json'))}))")
          RESPONSE=$(curl -sf -X POST http://localhost:8000/predict \
            -H 'Content-Type: application/json' \
            -d "{\"rows\": [$FEATURES]}")
          echo "$RESPONSE"
          echo "$RESPONSE" | python -c "import json,sys; body=json.load(sys.stdin); assert body['n']==1; assert isinstance(body['preds'][0], float)"
          docker rm -f airq
```

- [ ] **Step 2: Add SHA tagging to `.github/workflows/deploy.yml`**

Replace the final `Build and push` step so images are traceable and rollback is possible:

```yaml
      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_ID }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_ID }}:${{ github.sha }}
```

Also insert a bake step before it, mirroring CI:

```yaml
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Bake the serving model
        run: |
          pip install -r requirements-serving.txt
          python -m scripts.bake_serving_model
```

- [ ] **Step 3: Push and watch CI**

```bash
git add .github/workflows/ci.yml .github/workflows/deploy.yml
git commit -m "ci: add format check, coverage, docker build and container smoke test

The smoke test asserts /predict actually returns a number, which is the
check that would have caught B4."
git push
```

Expected: both the `test` and `docker` jobs pass. Watch at
`https://github.com/AyDaoud/airq_montreal_mlops/actions`.
If `black --check` fails, run `.venv/bin/black src tests scripts`, commit and push again.

---

## Task 18: Repair the Prefect flow

**Files:**
- Modify: `orchestration/flow.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_flow_imports.py`:

```python
def test_flow_module_imports():
    """orchestration/flow.py imported src.data.rsqa_ingest, which never existed."""
    import orchestration.flow as flow

    assert hasattr(flow, "daily_pipeline")
    assert hasattr(flow, "ingest_daily")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_flow_imports.py -v
```

Expected: FAIL — `AttributeError` or an import error on `ingest_hourly`'s dependencies.

- [ ] **Step 3: Replace the ingest tasks in `orchestration/flow.py`**

Delete `ingest_hourly` and `featurize_hourly` (lines 39-91) along with the hardcoded URL dictionary, and replace the import block at lines 30-31 with:

```python
from src.data.cli import cmd_build
from src.data.rsqa_ingest import ingest_historical, ingest_realtime
from src.data.stations import ingest_stations
```

Then add:

```python
@task
def ingest_daily() -> None:
    """Refresh the station dimension and pull the realtime increment."""
    ingest_stations()
    ingest_historical()
    ingest_realtime()


@task
def build_gold_table() -> None:
    """Rebuild silver and gold from bronze."""
    import argparse

    cmd_build(argparse.Namespace())
```

And extend `daily_pipeline` so ingestion runs first:

```python
@flow(name="airq-daily-pipeline")
def daily_pipeline(model: str = "rf", freq: str = "D", horizon: int = 30) -> None:
    """ingest -> build -> train -> forecast -> monitor."""
    ingest_daily()
    build_gold_table()
    train_daily_model(model=model)
    forecast_daily_model(model=model, freq=freq, horizon=horizon)
    monitor_daily_model(model=model)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_flow_imports.py -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add orchestration/flow.py tests/test_flow_imports.py
git commit -m "fix(orchestration): wire the flow to the real ingestion layer

Removes the import of src.data.rsqa_ingest that was never committed and
the hardcoded resource URLs, which now live in src/data/sources.py."
```

---

## Task 19: README

**Files:**
- Modify: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Replace sections 3 onward in `README.md`**

The current README stops mid-document at "3. Repository Structure". Keep sections 1-2, then append:

````markdown
## 3. Quickstart

```bash
git clone git@github.com:AyDaoud/airq_montreal_mlops.git
cd airq_montreal_mlops
make setup          # create .venv and install
make data           # ingest sources and build the gold table (~5 min)
make test           # run the offline test suite
make run-api        # serve on http://localhost:8000
```

## 4. Architecture

```
donnees.montreal.ca            Open-Meteo
  historical dump (annual)       archive (training)
  realtime feed  (daily)         forecast (inference)
        |                              |
        v                              v
   +-----------------------------------------+
   |  src/data  — contracts at every boundary |
   +-----------------------------------------+
        |            |               |
      bronze  ->  silver  ->       gold
     (verbatim)  (normalized)  (daily/station + weather + geo + targets)
                                     |
                      +--------------+--------------+
                      |              |              |
                   training       forecast       monitoring
                   (MLflow)      (batch)        (Evidently)
                      |
                   FastAPI  ->  Docker  ->  GHCR
```

## 5. Data sources

| Source | Cadence | Role |
|---|---|---|
| RSQA IQA historical (per-station, per-pollutant) | annual dump | backfill |
| RSQA IQA realtime | daily | increment |
| RSQA station list | rare | geography |
| Open-Meteo archive | ~2-day lag | training weather |
| Open-Meteo forecast | live | inference weather |

The air-quality index is **not published directly** — it is the maximum of
the per-pollutant sub-indices for a station-hour. Timestamps are UTC-5
year-round, not local time with daylight saving.

## 6. What the data says

- 11 active stations, 2022-01-01 onward, 96-100 % daily completeness
- Daily IQA above 50 ("mauvais") on **2.98 %** of station-days
- **PM drives 99.8 % of exceedances** — Montreal's bad-air days are
  particulate, which is why wind and precipitation are model features

## 7. Results

Populated by Spec B, which adds persistence and seasonal-naive baselines
and a rolling-origin backtest. Until then this project makes no accuracy
claims.

## 8. License

MIT — see `LICENSE`.
````

- [ ] **Step 2: Add the license**

```bash
curl -sL https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt \
  | sed "s/{{ year }}/2026/; s/{{ organization }}/Ayman Daoud/" > LICENSE
head -3 LICENSE
```

If the download fails, write the standard MIT text manually with `Copyright (c) 2026 Ayman Daoud`.

- [ ] **Step 3: Verify the quickstart works from scratch**

```bash
cd /tmp && rm -rf airq-verify && git clone -b spec-a/data-foundation \
  git@github.com:AyDaoud/airq_montreal_mlops.git airq-verify && cd airq-verify
make setup && make data && make test
```

Expected: all three succeed. **This is acceptance criterion 1** — if it fails, the spec is not done.

- [ ] **Step 4: Commit and push**

```bash
cd /home/ayman/airq_montreal_mlops
git add README.md LICENSE
git commit -m "docs: finish the README with quickstart, architecture and data sources"
git push
```

---

## Final verification

- [ ] **Run the whole suite with coverage**

```bash
.venv/bin/python -m pytest --cov=src/data --cov-report=term-missing
```

Expected: all pass, `src/data` coverage ≥ 80 % (acceptance criterion 2).

- [ ] **Check every acceptance criterion in spec section 11**

```bash
grep -rn "C:/Users" src scripts || echo "AC8 ok: no hardcoded paths"
.venv/bin/python -c "
import pandas as pd
g = pd.read_parquet('data/gold/daily_station_iqa.parquet')
print('AC3:', len(g), 'rows,', g['station_id'].nunique(), 'stations')
"
```

- [ ] **Open the pull request**

```bash
git push
```

Then open `https://github.com/AyDaoud/airq_montreal_mlops/pull/new/spec-a/data-foundation`.

---

## Plan self-review

Checked against the spec on 2026-09-14:

| Spec section | Covered by |
|---|---|
| §2.1 UA blocking | Task 1 |
| §2.2 two-source ingestion | Tasks 2, 6 |
| §2.3 schema differences | Task 6 |
| §2.4 IQA = max | Task 10 |
| §2.5b fixed-offset EST | Task 6 |
| §2.6 dirty stations file | Tasks 3, 4, 7 |
| §2.7 Open-Meteo, no BLH | Task 8 |
| §4 module layout | Tasks 1-11 |
| §5 gold schema | Tasks 10, 11 |
| §6 D1-D10 | D1/D2 Task 10, D3 Task 6, D4 Task 10, D5 Task 8, D6 Tasks 7-8, D7 Tasks 0/16, D8 Task 2, D9 Task 7, D10 Task 9 |
| §7 contracts + freshness | Tasks 4, 11 |
| §8 offline tests | Tasks 3-15 |
| §9 serving, Docker, CI | Tasks 15, 16, 17 |
| §10 migration | Tasks 13, 14, 18 |
| §11 acceptance | Final verification |

**Deferred deliberately, with a failing test rather than silence:**
- Station-vs-time split defect → `tests/test_training_split.py` xfail (Spec B)
- Today's IQA excluded from features → `tests/test_build_features.py` xfail (Spec B)

Both xfails are `strict=True`, so they fail loudly the moment Spec B fixes them and the marker is stale.
