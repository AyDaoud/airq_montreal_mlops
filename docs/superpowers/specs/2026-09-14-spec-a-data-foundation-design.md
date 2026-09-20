# Spec A — Data Foundation

- **Date:** 2026-09-14
- **Status:** Approved (pending spec review)
- **Scope:** P0 #1–5 + feature ② (weather) + feature ⑤ (station geo) + target schema for feature ③
- **Branch:** `spec-a/data-foundation`
- **Successors:** Spec B (modelling & honesty), Spec C (observability), Spec D (automation & cloud)

---

## 1. Context

The repository currently cannot run end-to-end from a clean clone. Four blockers were identified:

| ID | Blocker | Evidence |
|----|---------|----------|
| B1 | Ingestion layer never committed | `orchestration/flow.py:30` imports `src.data.rsqa_ingest`; `git log --all --diff-filter=A` shows the file was never added |
| B2 | `evidently` imported but undeclared | `src/monitoring/check_iqa.py:12-16`; absent from `requirements.txt` |
| B3 | Hardcoded personal path as MLflow default | `scripts/train_daily_iqa.py:28` — `sqlite:///C:/Users/AU51870/...` |
| B4 | Published Docker image contains no model | `Dockerfile:18` copies `artifacts/`, which `.gitignore` excludes |

Nothing in the repo produces `data/interim/iqa_daily_*.parquet`, which every training and forecasting path depends on.

Spec A closes all four and, in the same pass, builds the ingestion layer that features ② and ⑤ require — so the ingestion layer is written **once**, not twice.

---

## 2. Verified facts

All facts below were confirmed by live inspection on 2026-09-14. They supersede assumptions in the original code.

### 2.1 The portal blocks the `curl` User-Agent specifically — CORRECTED

An earlier draft of this spec claimed the portal blocks all non-browser User-Agents and that `pd.read_csv(url)` therefore fails. **That was wrong**, and it was wrong because every probe behind it used `curl`. Re-measured across clients on 2026-09-14:

| User-Agent | Status |
|---|---|
| `curl/8.5.0` | **403** `RBAC: access denied` |
| `python-requests/2.32.3` | 200 |
| `Python-urllib/3.12` | 200 |
| *(no UA header)* | 200 |
| browser UA | 200 |

`pd.read_csv(url)` succeeds today. The block is a `curl`-specific WAF rule, not a browser allowlist.

**What this does and does not justify.** It does not justify `src/data/http.py` as a workaround for a broken default client — there is nothing to work around from Python. The module is still worth keeping for the reasons below, which stand on their own:

* retry with backoff for the **HTTP 503** genuinely observed on the realtime resource (§2.2)
* conditional `If-None-Match` requests, so re-ingesting does not re-download 26 MB unchanged
* one place to set timeouts and one place to change if the WAF rule broadens later

Setting a browser UA costs nothing and insures against the rule widening, so it stays — but it is belt-and-braces, not the load-bearing reason for the module.

### 2.1b The real cause of blocker B1: `.gitignore` silently swallowed `src/data/`

The original `.gitignore` line 10 was an unanchored `data/`. Git matches an unanchored directory pattern **at any depth**, so it also matched `src/data/` and `tests/data/`. Verified:

```
$ git check-ignore -v src/data/rsqa_ingest.py
.gitignore:1:data/	src/data/rsqa_ingest.py
```

`git add src/data/rsqa_ingest.py` would have silently refused. This is almost certainly why the ingestion module appears in no commit while `orchestration/flow.py:30` imports it: the file very likely existed on the author's disk and git declined to track it, with no error on a plain `git add <dir>`.

**Fix:** anchor the rule to `/data/` so only the repository-root data directory is ignored. Applied in Task 1. Every later task that adds files under `src/data/` depends on this.

### 2.2 The historical IQA files are annual dumps, not a daily feed

| Resource | Resource ID | Last modified | Data ends |
|---|---|---|---|
| IQA détaillé par station (2022–2024) | `0c325562-e742-4e8e-8c36-971f3c9e58cd` | 2024-12-31 | 2024-12-31 |
| IQA détaillé par station (2025–2027) | `6cf08815-49d2-4d2f-a400-ce36ee52b0fc` | **2026-01-19** | **2026-01-18** |

Today is 2026-09-14 — the historical source is **8 months stale** and updates roughly annually.

A separate **real-time** resource returns current-day data:

- Dataset `3e9f7b96-3f25-4404-a5ad-22d9a31060e6`, resource `6554355e-63d1-4a01-a268-91e0763c3606` (`iqa-by-station.csv`, ~16 KB)
- Verified to contain rows dated `2026-09-14`
- Observed one transient `HTTP 503` → retry with backoff is mandatory

### 2.3 Schemas differ between the two sources

```
historical : stationId,polluant,valeur,date,heure                     # French  "polluant"
realtime   : Id,stationId,address,latitude,longitude,X,Y,
             pollutant,valeur,date,heure                              # English "pollutant"
```

Both are **long format, one row per (station, pollutant, hour)**. `heure` is `0..23` (no 1–24 off-by-one). `valeur` is a non-null integer in `[0, 484]`. No duplicates on `(stationId, polluant, date, heure)` across 1,335,962 historical rows.

**The column name is not the only difference — the pollutant *vocabulary* differs too.** Found during Task 6, missed in the original survey:

| | historical | realtime |
|---|---|---|
| column name | `polluant` | `pollutant` |
| particulate label | `PM` | **`PM2.5`** |
| observed values | `PM, O3, NO2, SO2, CO` | `O3, PM2.5, SO2` (today) |

They denote the same quantity — RSQA's particulate sub-index *is* PM2.5; the historical dump simply labels it `PM`. Normalization maps `PM2.5`/`PM2_5` → `PM` **before** contract validation, so both sources satisfy one vocabulary. An unrecognised spelling therefore fails the contract loudly rather than entering the pipeline silently, which is the intended behaviour.

### 2.4 IQA must be computed, not read

Neither file contains the air-quality index. Both contain **per-pollutant sub-indices**. The official IQA is the **maximum across pollutants** for a station-hour.

The existing `build_features_daily_iqa` fallback ("take the first numeric column", `src/features/build_features.py:153-158`) therefore trained on an arbitrary pollutant sub-index, not IQA.

### 2.5 Data shape

- **11 stations:** `3, 6, 17, 28, 31, 50, 55, 66, 80, 99, 103`
- **Span:** 2022-01-01 → 2026-01-18 (1,479 days); per-station completeness 96.4 %–99.9 %
- **Pollutants:** `PM` (383k rows), `O3` (344k), `NO2` (325k), `SO2` (167k), `CO` (117k)
- **Daily IQA** (max over hours): mean 25.7, median 23, max 484
- **Class balance on `iqa > 50`: 482 / 16,186 station-days = 2.98 %**
- **PM drives 99.8 % of exceedances** (4,007 PM vs 7 O₃ hourly)

The PM finding is the physical justification for feature ②: particulate episodes are governed by wind speed (dispersion), wind direction (transport), and precipitation (scavenging).

### 2.5b Timestamps are fixed-offset EST, not local-with-DST

Verified on the 2024 DST transition days:

| Day | Event | Distinct hours | Station 3 / O3 rows |
|---|---|---|---|
| 2024-03-10 | spring forward | 24 | 24 (hours 0–23, **including 02**) |
| 2024-11-03 | fall back | 24 | 24 |
| 2024-06-15 | normal | 24 | 24 |

Local wall-clock time would give 23 hours on the spring-forward day and a duplicated hour on the fall-back day. Both days have exactly 24 distinct hours and hour `02` is present on 2024-03-10, so `(date, heure)` is **UTC−5 year-round** (Eastern Standard Time, the convention used by air-quality networks).

**Consequence:** `ts.tz_localize("America/Montreal")` raises `NonExistentTimeError` on every spring-forward date. Parsing must localize to the fixed offset `Etc/GMT+5` and only then convert.

```python
ts_est   = (pd.to_datetime(date) + pd.to_timedelta(heure, unit="h")).dt.tz_localize("Etc/GMT+5")
ts_utc   = ts_est.dt.tz_convert("UTC")
ts_local = ts_utc.dt.tz_convert("America/Montreal")   # civil time, for calendar features
```

### 2.6 Station geo joins cleanly — but the file is dirty

Dataset `ae01f7f3-4d69-404a-9be1-74abfdc96571`, resource `29db5545-89a4-4e4a-9e95-05aa6dc2fd80`.

All 11 IQA station IDs match `numero_station`. Defects present in the file:

- UTF-8 BOM
- A malformed duplicated header fragment quoted into the final column
- 5 trailing all-`NaN` rows
- **Station 62 has `latitude = 4.504576e+07`** (corrupt by ~6 orders of magnitude)
- `statut` is unreliable: stations 28, 50 and 66 are marked `fermé` yet report data through 2026-01-18. **Do not filter on `statut`.**
- **The filename embeds a revision date** (`liste-des-stations_do_rev_2026-05-22.csv`) and will change. Resolve the URL through the CKAN API, never hardcode it.

### 2.7 Open-Meteo

| Variable | Archive API | Forecast API |
|---|---|---|
| `temperature_2m` | ✅ | ✅ |
| `relative_humidity_2m` | ✅ | ✅ |
| `precipitation` | ✅ | ✅ |
| `wind_speed_10m` | ✅ | ✅ |
| `wind_direction_10m` | ✅ | ✅ |
| `surface_pressure` | ✅ | ✅ |
| `boundary_layer_height` | ❌ **returns null** | ✅ |

Archive lag is ~2 days (on 2026-09-14 the archive served through 2026-09-12).

**`boundary_layer_height` is dropped.** Training can only use the archive; using a forecast-only variable would create train/serve skew.

---

## 3. Goals and non-goals

### Goals
1. A clean clone can produce every downstream artifact with committed code alone.
2. Ingestion is idempotent, watermarked, and supports explicit backfill.
3. Every external boundary is validated by an explicit contract, not by column-name guessing.
4. Weather and station geography are first-class columns in the gold table.
5. The target schema Spec B needs is decided and written here.
6. The published Docker image is small and actually contains a servable model.
7. CI proves the image works, so B4 cannot recur.

### Non-goals (deferred)
- Model quality, baselines, backtesting, conformal intervals → **Spec B**
- Postgres, Grafana, compose, prediction logging, Streamlit → **Spec C**
- Champion/challenger promotion, Terraform, Cloud Run → **Spec D**
- The `rsqa-polluants-gazeux` multi-pollutant dataset (wide format, `N/M` sentinel, `DD-MM-YYYY` dates) — noted as a future enrichment, not in scope.

---

## 4. Architecture

### 4.1 Modules

```
src/data/
  __init__.py
  sources.py       # Registry of CKAN dataset/resource IDs. Single source of truth for URLs.
  http.py          # Session: browser UA, retry/backoff, ETag caching, streaming download.
  ckan.py          # Resolve a resource ID -> current download URL via the CKAN API.
  rsqa_ingest.py   # Historical backfill + realtime increment -> bronze.
  stations.py      # Station dimension: geo, names, boroughs.
  weather.py       # Open-Meteo archive (training) and forecast (inference) -> bronze.
  aggregate.py     # bronze -> silver -> gold.
  contracts.py     # pandera schemas for every boundary.
  watermark.py     # Read/write ingestion watermarks (JSON sidecar).
```

Each module has one responsibility and is independently testable. `http.py` is the only module that performs network I/O for the portal; `weather.py` is the only one that talks to Open-Meteo.

### 4.2 Medallion layers

```
data/
  bronze/
    rsqa_iqa/historical/year=YYYY/part.parquet    # as-downloaded, normalized column names only
    rsqa_iqa/realtime/date=YYYY-MM-DD/part.parquet
    stations/stations.parquet
    weather/cell=<lat>_<lon>/year=YYYY/part.parquet
  silver/
    iqa_hourly.parquet        # station_id, ts_utc, ts_local, pollutant, value
    weather_hourly.parquet    # cell_id, ts_utc, ts_local, <weather vars>
    stations.parquet          # station_id, name, borough, lat, lon, cell_id
  gold/
    daily_station_iqa.parquet # one row per (station_id, date_local)
  _watermarks.json
```

Bronze is immutable and append-only. Silver and gold are deterministic rebuilds from bronze — so a bug fix means re-running a transform, never re-downloading.

### 4.3 Ingestion strategy (option (a), approved)

**Historical backfill** — run once, or on demand:
`ingest_historical(years)` downloads the annual dumps, writes bronze partitioned by year, records a watermark per resource keyed on the CKAN `last_modified`. Re-running with an unchanged `last_modified` is a no-op.

**Realtime increment** — run daily:
`ingest_realtime()` fetches `iqa-by-station.csv`, writes `bronze/rsqa_iqa/realtime/date=<today>/`, and advances the watermark. Idempotent: re-running the same day overwrites that day's partition rather than appending duplicates.

**Reconciliation** — when an annual dump later covers days already collected in real time, the historical partition wins (it is the corrected, official record). `aggregate.py` applies `drop_duplicates(subset=[station_id, ts_utc, pollutant], keep='first')` with historical ordered first.

`--backfill START END` re-derives any date range from bronze without network access.

### 4.4 Weather fetch de-duplication

The 11 stations span lat 45.427–45.652, lon −73.929 to −73.500 (~40 km). Measured cell counts at candidate roundings:

| Rounding | Unique cells |
|---|---|
| 0.01° (2 decimals) | 11 — no de-duplication |
| **0.1° (ERA5-Land grid)** | **7** ← chosen |
| 0.25° (ERA5 grid) | 4 — flattens spatial signal |

Rounding finer than the source grid requests duplicate data under distinct keys, so `cell_id` is the station's centre rounded to **0.1°**, giving 7 fetches instead of 11 while preserving cross-island variation that feature ⑤ depends on. `weather.py` fetches once per cell and joins back; the assignment lives in `silver/stations.parquet` so the mapping is inspectable.

---

## 5. Gold table schema

One row per `(station_id, date_local)`.

| Column | Type | Notes |
|---|---|---|
| `station_id` | int16 | |
| `date_local` | date | America/Montreal calendar day |
| `iqa` | int16 | **max** over pollutants over the day's hours |
| `driving_pollutant` | category | pollutant attaining the daily max |
| `sub_PM`, `sub_O3`, `sub_NO2`, `sub_SO2`, `sub_CO` | int16, nullable | daily max per pollutant |
| `n_hours_observed` | int8 | 0–24; coverage guard |
| `temp_mean`, `temp_min`, `temp_max` | float32 | |
| `wind_speed_mean`, `wind_speed_max` | float32 | |
| `wind_dir_sin`, `wind_dir_cos` | float32 | circular encoding of the daily resultant vector |
| `precip_sum` | float32 | |
| `humidity_mean`, `pressure_mean` | float32 | |
| `lat`, `lon`, `borough` | float32 / category | station dimension |
| `target_iqa_h24`, `target_iqa_h48` | int16, nullable | regression targets |
| `target_exceed_h24`, `target_exceed_h48` | bool, nullable | `iqa > 50`; base rate ≈ 2.98 % |

**Wind direction is encoded as sin/cos**, never as raw degrees — 359° and 1° must be near-neighbours.

Targets are written by `aggregate.py` and grouped by `station_id`, so no target ever crosses a station boundary.

---

## 6. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | IQA = max over pollutant sub-indices per station-hour | Official definition (§2.4) |
| D2 | Daily IQA = max over the day's **Montreal civil** (`America/Montreal`) hours, and weather is bucketed on the **same** civil day | Montréal bad-air-day convention; matches the alert framing of feature ③ |

> **Two consequences of D2, both verified — do not "fix" either.**
>
> **(a) The published 2.98% exceedance rate assumes EST-date bucketing.** The §2.5 survey grouped by the raw `(date, heure)` EST calendar. Grouping by the civil calendar moves hours across day boundaries during EDT, so some daily maxima land on a different day:
>
> | Bucketing | Station-days | Exceedances | Rate |
> |---|---|---|---|
> | EST date (original survey) | 16,186 | 482 | 2.98% |
> | **Montreal civil date (pipeline)** | 16,187 | 473 | **2.92%** |
>
> Civil date is the correct choice — a "bad air day" is the day residents lived through. **2.92% is the number Spec B should quote.**
>
> **(b) Weather must bucket on the civil date too.** An earlier implementation bucketed weather on the UTC date while IQA used the civil date, which offset every weather feature by 4–5 hours from the air quality it explains (weather "2024-07-15" actually covered local 14th 20:00 → 15th 19:00). Both sides now use the civil date; `test_weather_and_iqa_bucket_on_the_same_calendar_day` pins this.
| D3 | Parse `(date, heure)` as fixed offset `Etc/GMT+5`; store `ts_utc`; derive calendar features from `ts_local` (`America/Montreal`) | Source is EST year-round (§2.5b), so direct `tz_localize("America/Montreal")` crashes on spring-forward dates. Civil local time is still correct for calendar features, since human activity follows the wall clock |
| D4 | Write Spec B's targets now | Schema decided once; avoids reshaping gold later |
| D5 | Drop `boundary_layer_height` | Archive returns null → train/serve skew (§2.7) |
| D6 | Fetch weather per 0.1° grid cell, not per station | 7 calls instead of 11 (measured); finer rounding than the source grid would duplicate data |
| D7 | Split `requirements.txt` / `requirements-serving.txt` | Serving image drops `torch` + `prophet`: ~2.5 GB → ~250 MB, fits the Artifact Registry free tier and cuts Cloud Run cold starts |
| D8 | Resolve portal URLs through the CKAN API | The stations filename embeds a revision date and will change (§2.6) |
| D9 | Do not filter stations on `statut` | The field is stale and contradicts the measurement record (§2.6) |
| D10 | Bronze is immutable; silver/gold are rebuildable | A transform bug never forces a re-download |

---

## 7. Data contracts

`src/data/contracts.py` defines one `pandera.DataFrameSchema` per boundary. Validation failures raise with the offending rows attached — replacing the alias-guessing in `build_features.py`.

| Contract | Key checks |
|---|---|
| `RawIqaHistorical` | required columns; `heure ∈ [0,23]`; `valeur ∈ [0,1000]` non-null; `date` parses |
| `RawIqaRealtime` | same, plus the English `pollutant` spelling |
| `RawStations` | `latitude ∈ [45.2, 45.8]`, `longitude ∈ [-74.1, -73.4]` — **catches the station-62 corruption**; drops all-NaN rows; `numero_station` castable to `Int64` |
| `SilverIqaHourly` | unique on `(station_id, ts_utc, pollutant)`; tz-aware timestamps |
| `SilverWeatherHourly` | `wind_direction ∈ [0,360]`; `precipitation ≥ 0` |
| `GoldDailyStationIqa` | unique on `(station_id, date_local)`; `iqa ≥ 0`; `n_hours_observed ∈ [1,24]` |

A **freshness check** runs after ingestion and warns when the newest `date_local` is more than 2 days behind today — this is the check that would have surfaced §2.2 immediately.

---

## 8. Testing

All tests are **offline**. Network access is never required to run the suite.

- `tests/fixtures/` holds small committed slices of each real source (~200 rows each), including the dirty station rows and a BOM, captured on 2026-09-14.
- `tests/data/test_http.py` — UA header is set; retry fires on 503; ETag short-circuits.
- `tests/data/test_contracts.py` — the corrupt latitude is rejected; valid frames pass.
- `tests/data/test_rsqa_ingest.py` — idempotency (running twice yields identical bronze); historical wins reconciliation.
- `tests/data/test_aggregate.py` — IQA equals the max over pollutants on a hand-computed fixture; targets shift by exactly 24 h / 48 h and never cross a station boundary; `wind_dir_sin/cos` round-trips.
- `tests/data/test_weather.py` — cell de-duplication; join preserves row count.

`tests/test_build_features.py` is rewritten to use ≥ 40 rows and assert real values (`lag_1[i] == value[i-1]`), replacing the current test that passes on an empty DataFrame.

`setup.cfg` drops `--maxfail=1`; `flake8 max-line-length` drops from 200 to 100.

---

## 9. Serving and CI changes

**Image split.** `Dockerfile` (serving) installs `requirements-serving.txt` only. `Dockerfile.train` keeps the full stack.

**B4 fix.** A CI job trains a small RF on a committed fixture and bakes it into the serving image, so the published image always contains a loadable model. The model path stays overridable by `MODEL_PATH` for Spec D, where it will be pulled from object storage instead.

**Serving correctness.** The model is loaded once at startup via a FastAPI `lifespan` handler rather than on every request (`src/serving/app.py:66`). The silent feature fallback (`app.py:50-51`) is removed — a request whose columns do not match the artifact's feature list now returns HTTP 422 naming the missing columns. The test-driven comment at `app.py:62-65` is removed and the test switches to `app.dependency_overrides`.

**CI gains:** `black --check`, `docker build`, and a smoke test that runs the image and asserts `/health` returns 200 and `/predict` returns a numeric prediction for a fixture row.

**B2/B3 fixes.** `evidently` is pinned in `requirements.txt`; the MLflow URI comes from `MLFLOW_TRACKING_URI` with a repo-relative `sqlite:///mlflow.db` default.

---

## 10. Migration of existing code

| Existing | Action |
|---|---|
| `orchestration/flow.py` | `ingest_hourly` replaced by calls into `src.data.rsqa_ingest`; hardcoded URLs move to `sources.py` |
| `src/features/build_features.py` | `build_features_daily_iqa` reads the gold table; alias-guessing and the numeric-column fallback deleted (contracts replace them) |
| `src/models/training_daily.py` | `_daily_df()` reads `data/gold/daily_station_iqa.parquet`; the station-vs-time split defect is left for Spec B and documented with a failing test marked `xfail` |
| `src/monitoring/check_iqa.py` | Import fixed by B2; logic untouched until Spec C |

---

## 11. Acceptance criteria

1. `git clone && make setup && make ingest && make train && make forecast` succeeds on a clean machine with no manual data steps.
2. `pytest` passes offline, with no network calls, and coverage on `src/data/` ≥ 80 %.
3. `data/gold/daily_station_iqa.parquet` contains ≈ 16,000 rows across 11 stations with every column in §5 populated.
4. The freshness check reports the real-time source as current (within 2 days).
5. `docker build` produces a serving image under 400 MB whose `/predict` returns a prediction.
6. CI is green, including the Docker smoke test.
7. README has a working quickstart, an architecture diagram, and a data-sources table.
8. No hardcoded absolute paths remain (`grep -r "C:/Users"` is empty).

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| The portal changes its UA policy or blocks the runner's IP | Ingestion degrades to bronze already on disk; freshness check warns; fixtures keep tests green |
| The real-time resource returns 503 (observed once) | Retry with exponential backoff; a failed day leaves the watermark unadvanced and is retried next run |
| Open-Meteo rate-limits the archive | 7 cells × 5 years is a handful of calls; responses cached in bronze and never re-fetched |
| 2.98 % positive rate makes feature ③ hard | Acknowledged now; Spec B uses PR-AUC, class weights, and reports precision/recall at the operating threshold rather than accuracy |
| Historical dump never updates again | The real-time increment is the live path; this is exactly why option (a) was chosen |
