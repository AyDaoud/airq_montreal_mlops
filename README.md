# AirQ Montréal – End-to-End MLOps Project

![CI](https://github.com/AyDaoud/airq_montreal_mlops/actions/workflows/ci.yml/badge.svg)

End-to-end MLOps pipeline for **air-quality forecasting in Montréal**.

The project ingests Montréal's open-data air-quality and Open-Meteo weather feeds, validates every stage with contracts, and builds a daily per-station gold table meant to support forecasting (Random Forest, Prophet, LSTM), experiment tracking in **MLflow**, orchestration with **Prefect**, and serving via **FastAPI**, **containerized with Docker** and checked by **tests + GitHub Actions CI + pre-commit hooks**. Spec A (this branch) builds the data foundation only — see [Results](#11-results) for what is, and is not, claimed yet.

---

## 1. Problem & Goals

**Problem.**
Given historical air-quality measurements (daily IQA / pollutant values for Montréal), we want to:

- Forecast IQA/pollutant for the next *N* days/hours
- Track all model experiments and artifacts
- Run a **daily pipeline**: ingest → train → forecast → monitor
- Expose a **web API** that serves model predictions
- Package the service into a **Docker image**, ready for deployment to any cloud/container platform

This aligns with the MLOps Zoomcamp project rubric: experiment tracking, model registry, workflow orchestration, deployment, monitoring, tests, linting, and CI. Spec A delivers the data half of this list (ingestion, contracts, the gold table, serving plumbing); honest evaluation, monitoring and deployment hardening are Specs B–D.

---

## 2. Tech Stack

- **Language:** Python 3.12
- **Data / ML:** pandas, scikit-learn, Prophet, PyTorch (LSTM)
- **Data contracts:** pandera (validates every bronze → silver → gold boundary)
- **Data sources:** Montréal open data (CKAN API) for air quality, Open-Meteo archive API for weather
- **Experiment Tracking & Registry:** MLflow (SQLite backend)
- **Orchestration:** Prefect flows (`orchestration/flow.py`)
- **Serving:** FastAPI + Uvicorn (`src/serving/app.py`)
- **Containerization:** Docker — separate serving (`Dockerfile`) and training (`Dockerfile.train`) images
- **Testing:** pytest, fully offline (`tests/`)
- **Code Quality:** black, flake8, pre-commit
- **CI:** GitHub Actions (`.github/workflows/ci.yml`)

---

## 3. Quickstart

```bash
git clone git@github.com:AyDaoud/airq_montreal_mlops.git
cd airq_montreal_mlops
make setup      # create .venv and install
make data       # ingest sources and build the gold table (~5 min)
make test       # 76 tests, fully offline
make run-api    # serve on http://localhost:8000
```

`make data` runs `src.data.cli ingest` (stations, historical + realtime IQA, Open-Meteo weather)
followed by `src.data.cli build` (bronze → silver → gold, contract-validated). It needs network
access; `make test` does not — the 76 tests run entirely offline against fixtures.

To serve predictions you need a baked model (`make bake-model` writes `artifacts/rf/model.pkl`
and `artifacts/rf/feature_names.json`), then `make run-api`. The API takes rows keyed by exact
feature name; a request missing a column is rejected with **422** naming it:

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "rows": [
          {
            "n_hours_observed": 24, "temp_mean": 12.4, "temp_min": 8.1, "temp_max": 17.0,
            "humidity_mean": 65.0, "precip_sum": 0.0, "wind_speed_mean": 14.2,
            "wind_speed_max": 22.0, "pressure_mean": 1013.5,
            "wind_dir_sin": 0.5, "wind_dir_cos": 0.87,
            "lag_1": 20, "lag_2": 18, "lag_3": 22, "lag_7": 19, "lag_14": 21,
            "roll_7": 20.1, "dow": 2, "month": 9
          }
        ]
      }'
# {"n":1,"preds":[23.00...]}
```

The 19 keys above are the full feature set the model was trained on (see
`artifacts/rf/feature_names.json`, produced by `make bake-model`); this exact request was run
against a locally-served instance to confirm the shape.

---

## 4. Architecture

```
   donnees.montreal.ca (CKAN)              Open-Meteo archive API
   historical IQA, realtime IQA,           hourly weather, 7 grid
   station list                            cells covering Montréal
            │                                        │
            └───────────────────┬────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  BRONZE  data/bronze/    │  verbatim downloads
                    └────────────┬────────────┘
                                 │  pandera contract at every hop
                                 ▼
                    ┌─────────────────────────┐
                    │  SILVER  data/silver/    │  iqa_hourly, weather_hourly,
                    │                          │  stations — normalized, typed
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  GOLD    data/gold/      │  daily_station_iqa.parquet
                    │                          │  16,195 rows x 29 cols,
                    │                          │  one row per (station, day)
                    └──────┬───────┬───────┬───┘
                           │       │       │
                 ┌─────────▼─┐ ┌───▼────┐ ┌▼──────────┐
                 │ training / │ │forecast│ │ monitoring │
                 │ MLflow      │ │(CLI)   │ │(check_iqa) │
                 │ RF/Prophet/ │ │        │ │            │
                 │ LSTM        │ │        │ │            │
                 └─────┬───────┘ └────────┘ └────────────┘
                       │ make bake-model
                       ▼
                 ┌─────────────┐
                 │  FastAPI     │  /health, /predict
                 └──────┬──────┘
                        │ docker build (Dockerfile)
                        ▼
                 ┌─────────────┐
                 │ Docker image │
                 └──────┬──────┘
                        │ push
                        ▼
                 ┌─────────────┐
                 │    GHCR      │
                 └─────────────┘
```

`orchestration/flow.py` is the Prefect flow that ties ingest → build → train/forecast/monitor
into one daily pipeline.

---

## 5. Data Sources

| Resource | Cadence | Role |
|---|---|---|
| RSQA historical IQA (CKAN, 2 resources spanning 2022–2027) | updates roughly **annually** (last modified 2026-01-19) | backfills the full history from 2022-01-01 onward |
| RSQA realtime IQA feed | updated daily/hourly | fills the gap between the historical dump's last update and today — required for a live pipeline |
| Station list (CKAN; filename carries a revision date) | resolved via the CKAN API on every run, not a static URL | station dimension: id, name, coordinates |
| Open-Meteo archive API | hourly, 7 grid cells covering Montréal | weather features (temperature, humidity, wind, pressure, precipitation) |

The two-source IQA design exists because neither source alone is sufficient: the historical dump
gives full history but is stale for most of the year, and the realtime feed is current but has no
depth. `src.data.cli ingest` pulls both and reconciles them (see quirk 3 below).

---

## 6. What the Data Says

Across 16,195 station-days (11 stations, 2022-01-01 onward), daily IQA is above 50 ("mauvais") on
**2.92%** of station-days (473 of 16,195). Of those 473 exceedances, **469 (99.2%) are driven by
particulate matter (PM)** rather than O3, NO2, SO2 or CO — Montréal's bad-air days are, almost
without exception, particulate days.

This is why the gold table carries weather (wind speed/direction, precipitation, humidity,
pressure) alongside pollutant history: PM accumulation and dispersion are governed by weather
conditions in a way that ozone or NO2 photochemistry is not. Whether these weather features
actually improve forecasts of exceedance days is an evaluation question for Spec B — Spec A only
establishes that the exceedances are overwhelmingly particulate, which is what motivates including
them.

---

## 7. Notes on the Source Data

Real engineering effort here went into surviving the quirks of the underlying feeds:

1. **The IQA is not published.** It is the maximum of per-pollutant sub-indices per station-hour
   and has to be computed from raw pollutant concentrations.
2. **Timestamps are fixed-offset UTC−5 year-round**, not local time. Hour 02 exists on
   spring-forward days, so a naive `tz_localize("America/Montreal")` raises
   `NonExistentTimeError` every March.
3. **The two IQA sources disagree on schema.** The historical dump uses column `polluant` and
   labels particulate matter `PM`; the realtime feed uses `pollutant` and labels the same thing
   `PM2.5`.
4. **The historical dump updates roughly annually** (last modified 2026-01-19), so a daily
   pipeline needs the separate realtime resource — see [Data Sources](#5-data-sources).
5. **The station list's filename embeds a revision date**, so download URLs are resolved through
   the CKAN API rather than hardcoded.
6. **The station file ships a corrupt coordinate** — station 62 has latitude 4.5e7 — which the
   data contract rejects rather than silently passing through.
7. **`donnees.montreal.ca` returns HTTP 403 to the `curl` user agent specifically**; Python HTTP
   clients are served normally.

---

## 8. Tech Stack

See [section 2](#2-tech-stack) above.

---

## 9. Project Layout

```text
.
├── orchestration/
│   ├── __init__.py
│   └── flow.py                       # Prefect flow: ingest -> build -> train/forecast/monitor
├── scripts/
│   ├── __init__.py
│   ├── bake_serving_model.py          # trains + writes artifacts/rf/{model.pkl,feature_names.json}
│   ├── make_fixtures.py               # builds offline test fixtures
│   └── train_daily_iqa.py             # train RF/Prophet/LSTM, log to MLflow + registry
├── src/
│   ├── __init__.py
│   ├── tracking.py                    # MLflow tracking URI resolution
│   ├── data/
│   │   ├── __init__.py
│   │   ├── aggregate.py                # bronze -> silver -> gold builders
│   │   ├── ckan.py                     # resolves CKAN resource IDs to current URLs
│   │   ├── cli.py                      # `python -m src.data.cli {ingest,build,all}`
│   │   ├── contracts.py                # pandera contracts + freshness checks
│   │   ├── http.py                     # HTTP session (User-Agent workaround, retries)
│   │   ├── rsqa_ingest.py              # historical + realtime IQA ingestion
│   │   ├── sources.py                  # registry of CKAN datasets/resources
│   │   ├── stations.py                 # station dimension ingestion
│   │   └── weather.py                  # Open-Meteo weather ingestion
│   ├── features/
│   │   └── build_features.py           # 19-feature daily builder (lags, rolling mean, weather)
│   ├── models/
│   │   ├── forecast.py                 # batch forecasting CLI
│   │   ├── model_factory.py            # save/load model artifacts
│   │   └── training_daily.py           # RF/Prophet/LSTM training helpers, time split
│   ├── monitoring/
│   │   ├── check_iqa.py                # metrics + drift + alert flag
│   │   └── metrics.py                  # regression metrics, drift score
│   └── serving/
│       └── app.py                      # FastAPI app (`/health`, `/predict`)
├── tests/
│   ├── conftest.py
│   ├── data/                           # ingestion/contract/aggregate tests
│   │   ├── test_aggregate_gold.py
│   │   ├── test_aggregate_silver.py
│   │   ├── test_ckan.py
│   │   ├── test_cli_smoke.py
│   │   ├── test_contracts.py
│   │   ├── test_http.py
│   │   ├── test_rsqa_ingest.py
│   │   ├── test_stations.py
│   │   └── test_weather.py
│   ├── fixtures/                       # offline fixtures used by the tests above
│   ├── test_api_basic.py
│   ├── test_build_features.py
│   ├── test_flow_imports.py
│   ├── test_tracking_config.py
│   └── test_training_split.py
├── Dockerfile                          # serving image (sklearn inference only)
├── Dockerfile.train                    # training image (full scientific stack)
├── makefile                            # setup, data, test, run-api, bake-model, build-docker, …
├── requirements.txt / requirements-serving.txt
├── setup.cfg                           # flake8 / pytest configuration
├── .pre-commit-config.yaml             # black, flake8 hooks
└── .github/
    └── workflows/
        └── ci.yml                      # lint, format check, coverage, docker build + smoke test
```

---

## 10. Testing & CI

`make test` runs **76 tests, fully offline** (3 are `strict=True` xfails pinning known defects,
see [Results](#11-results)) — verified to pass with `data/` and `artifacts/` removed, so cloning
and running tests never requires network access or pre-existing data.

CI (`.github/workflows/ci.yml`) on every push/PR:

1. installs dependencies and runs `flake8` and `black --check`
2. runs `pytest --cov=src/data --cov-report=term-missing`
3. bakes a model, builds the serving Docker image, and smoke-tests the running container by
   polling `/health` and posting a real `/predict` request built from
   `artifacts/rf/feature_names.json`

---

## 11. Results

**Spec A builds the data foundation and makes no accuracy claims.** There is no honest evaluation
in this branch — no train/test split that actually separates by time, no baseline to beat, no
backtest. Reporting a model metric without those would be misleading, so none is reported.

Spec B adds:

- persistence and seasonal-naive baselines
- a rolling-origin backtest
- honest evaluation before any accuracy number is claimed

Three defects were found while building the data/training layer and are deliberately **not fixed**
in Spec A. Each is pinned by a `strict=True` xfail test rather than left silent, so the test suite
fails loudly the moment one is fixed without updating the test:

- **The "time holdout" is really a station holdout.** `_time_split` slices a frame sorted by
  `[station_id, date_local]`, so the split separates stations, not time.
  Pinned by `tests/test_training_split.py::test_time_split_holdout_starts_after_train_ends`.
- **Prophet and the LSTM flatten 11 interleaved station series into one.** Both see one series
  where there should be 11 independent per-station series.
  Pinned by `tests/test_training_split.py::test_series_builder_yields_one_series_per_station`.
- **Today's IQA is excluded from the feature set.** The target is tomorrow's IQA, but the current
  day's own value is not a feature, so the model predicts `t+1` from `t-1` backwards. It is
  effectively a two-step-ahead model reported as one-step, and it discards the single most
  predictive input available.
  Pinned by `tests/test_build_features.py::test_current_day_iqa_is_available_as_a_feature`.

All three are fixed in Spec B.

---

## 12. Roadmap

- **Spec B — honest evaluation.** Fix the two defects above, add persistence/seasonal-naive
  baselines, a rolling-origin backtest, conformal prediction intervals, and an exceedance
  classifier (predicting the rare "mauvais" days directly rather than only via a regression
  threshold).
- **Spec C — operate it.** Docker Compose, Postgres for prediction logging, Grafana dashboards,
  a Streamlit map of current/forecast IQA by station.
- **Spec D — productionize it.** Champion/challenger model promotion, Terraform-managed
  infrastructure, deployment to Cloud Run.

---

## 13. License

[MIT](LICENSE) © 2026 Ayman Daoud
