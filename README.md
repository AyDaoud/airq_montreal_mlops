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
make setup      # create .venv, install, and install the git hooks
make data       # ingest sources and build the gold table (~5 min)
make test       # 129 tests, fully offline
make run-api    # serve on http://localhost:8000
```

`make data` runs `src.data.cli ingest` (stations, historical + realtime IQA, Open-Meteo weather)
followed by `src.data.cli build` (bronze → silver → gold, contract-validated). It needs network
access; `make test` does not — the 129 tests run entirely offline against fixtures.

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
                 │  FastAPI     │  /health, /predict, /metrics
                 └──────┬──────┘
                        │
          ┌─────────────┼──────────────────┐
          │ every served│prediction        │ docker build (Dockerfile)
          ▼             │                  ▼
   ┌─────────────┐      │           ┌─────────────┐
   │ prediction    │     │ /metrics  │ Docker image │
   │ + score log   │     │  scraped  └──────┬──────┘
   └──────┬────────┘     ▼                  │ push
          ▼        ┌─────────────┐          ▼
   ┌─────────────┐ │  Prometheus  │   ┌─────────────┐
   │   SQLite     │ └──────┬──────┘   │    GHCR      │
   │ monitoring.db│        │          └─────────────┘
   └──────┬───────┘        │
          │                │
          └───────┬────────┘
                  ▼
           ┌─────────────┐
           │   Grafana    │  5 panels: MASE, freshness,
           └─────────────┘  API health, predictions, scores
```

`orchestration/flow.py` is the Prefect flow that ties ingest → build → train/forecast/monitor
into one daily pipeline. Prometheus and Grafana run as userspace tarballs
(`scripts/dev_stack.sh`) or, for reviewers with Docker, via `docker-compose.yml` — see
[Observability](#12-observability).

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

`make test` runs **129 tests, fully offline, zero xfails** — verified to pass with `data/` and
`artifacts/` removed, so cloning and running tests never requires network access or
pre-existing data. All three defects that Spec A pinned as `strict=True` xfails are fixed;
see [Results](#11-results).

CI (`.github/workflows/ci.yml`) on every push/PR:

1. installs dependencies and runs `flake8` and `black --check`
2. runs `pytest --cov=src/data --cov-report=term-missing`
3. bakes a model, builds the serving Docker image, and smoke-tests the running container by
   polling `/health` and posting a real `/predict` request built from
   `artifacts/rf/feature_names.json`

---

## 11. Results

### Verdict

Huber regression beats persistence ("tomorrow = today") by roughly **10%**, in **5 of 5**
backtest folds. That is the one model in this repo that earns its keep on a rolling-origin
backtest.

### How it is measured

Every number below comes from `python -m scripts.run_backtest`, which writes
[`docs/RESULTS.md`](docs/RESULTS.md) from scratch — nothing here is hand-typed. The backtest
uses 5 rolling-origin folds, 90-day test windows, an expanding training window, horizons of
24h/48h/72h, and MASE scaled to persistence (**1.00 ties it, below 1.00 beats it**). Data runs
through 2026-01-18. Regenerate with:

```bash
.venv/bin/python -m scripts.run_backtest
```

### The regression scoreboard (h24, mean MASE across 5 folds)

| model | mean MASE | folds beaten |
|---|---|---|
| `huber_level` / `huber_delta` | **0.899** | **5/5** |
| `ridge_delta` / `ridge_level` | 0.970 | 4/5 |
| `rf_level` | 0.971 | 3/5 |
| `persistence` | 1.000 | — |
| `hgb_delta` | 1.008 | 3/5 |
| `climatology` | 1.094 | 1/5 |
| `seasonal_naive` | 1.484 | 0/5 |

Full table, all three horizons: [`docs/RESULTS.md`](docs/RESULTS.md).

### Why Huber wins

A 2×2 ablation (level vs. delta target, squared vs. Huber loss) shows the Δ-target framing
contributes **exactly 0.0000** to the win — `huber_level` and `huber_delta` score identically.
That makes sense: `iqa` is already a feature, so a linear model spans the same hypothesis space
whether it predicts the level or the change from today. **The entire win is the robust loss.**
Squared error is dominated by heavy-tailed exceedance days — persistence's RMSE (14.7) is nearly
double its MAE (7.4) — and Huber behaves linearly beyond its threshold instead of over-weighting
those outliers. Tree models (`rf_level`, `hgb_delta`) lose because they shrink predictions toward
the training mean, which is the wrong bias for a series that behaves close to a random walk.

### The alert product: ranking works, the operating point doesn't

The exceedance classifier is scored against a 5:1 cost ratio (missing a bad-air day costs 5x a
false alarm). Its ranking has genuine signal — PR-AUC of 0.42–0.52 across folds, a **~7.4x lift**
over the base rate. But **at an operating point, it does not reliably beat persistence**
("today was bad, so tomorrow will be too"):

| fold | PR-AUC | classifier P / R | persistence P / R |
|---|---|---|---|
| 1 | 0.424 | 0.636 / 0.135 | 0.490 / 0.462 |
| 3 | 0.524 | 0.622 / 0.500 | 0.518 / 0.518 |
| 4 | 0.474 | 0.489 / 0.228 | 0.554 / 0.554 |

Recall is far too low for a cost ratio meant to favour catching misses. The mechanism:
`class_weight="balanced"` inflates predicted probabilities during training, so a threshold fit
on training predictions is far too conservative once applied to test data.

#### That diagnosis was right, and fixing it did not help

Four alerting rules were compared on total expected cost at 5:1, summed across folds.
Reproduce with `python -m scripts.compare_alert_rules`. **Lower is better:**

| horizon | **persistence** | classifier @0.5 | classifier, calibrated | Huber forecast, thresholded |
|---|---|---|---|---|
| h24 | **581** | 787 | 706 | 650 |
| h48 | **839** | 1060 | 1030 | 1036 |
| h72 | **941** | 1056 | 1030 | 1030 |

Holding out the last fifth of training *by date* to pick the threshold, and isotonic
probability calibration on that slice, both helped relative to the naive threshold — and
both still lose. So does deriving the alert by thresholding the Huber forecast, the one
model that *does* beat persistence on the numeric task.

**Conclusion: machine learning does not improve this alert. Use the persistence rule.**
Flag tomorrow when today's IQA already exceeds 50 — precision ≈0.52, recall ≈0.51, no model
required. It is also trivially explainable to the public-health audience the alert is for.

Why persistence is so strong here while losing the regression: exceedance episodes are
*clustered* — wildfire smoke and winter inversions last several days — so "today was bad"
captures the dominant structure directly. The classifier's ranking does contain extra signal
(7.4× lift), but not enough to convert into better decisions at any threshold tried. The
regression is a different task: it is rewarded for being close on the ~95% of ordinary days,
which is where Huber's robustness pays.

**This is a negative result, published rather than buried.** Spec B was explicitly permitted
to conclude a problem was not learnable, and for the alert task it is not — at least not by
these methods at this horizon.

### Exceedance is violently seasonal

The "mauvais" base rate swings **32x** across the five folds:

| fold | window | exceedances | rate |
|---|---|---|---|
| 1 | 2024-10-24 → 2025-01-21 | 52/962 | 5.41% |
| 2 | 2025-01-22 → 2025-04-21 | 12/985 | 1.22% |
| 3 | 2025-04-22 → 2025-07-20 | 56/990 | 5.66% |
| 4 | 2025-07-21 → 2025-10-18 | 101/990 | 10.20% (wildfire season) |
| 5 | 2025-10-19 → 2026-01-16 | 3/934 | 0.32% |

This means fold-level variance on the classification task is large, and any single train/test
split on this problem is not trustworthy — which is exactly why the backtest reports a spread of
folds rather than one number.

### Does weather help? No — and that contradicts this project's own premise

```
mean PR-AUC   no_weather: 0.4149   with_weather: 0.4136
weather helps in 1 of 4 scored folds
lift over base rate: 7.4x either way
```

Weather and no-weather are essentially tied, and which one wins swings per fold. Spec A's stated
premise was that weather features would be **"the biggest modelling unlock."** They are not: the
honest backtest shows no measurable benefit. This result is reported plainly because it
contradicts an earlier design assumption — a project that publishes a result contradicting its
own premise is more credible than one that quietly drops it.

### The three Spec A defects are fixed

All three defects pinned by `strict=True` xfail tests in Spec A are now fixed, and the
current suite (129 tests) has zero xfails:

- **The "time holdout" was really a station holdout.** `_time_split` now splits by date, not by
  station.
- **Prophet and the LSTM flattened 11 interleaved station series into one.** Both now train on
  independent per-station series.
- **Today's IQA was excluded from the feature set.** It is now included.

One consequence of the fix is visible in `train_rf`: `mae_va` rose from **4.579 to 8.042**
(+76%) on the same model. That is the correct number, not a regression — the old value was
measured against a station leak; the new one is measured against a genuine future-date holdout.

---

## 12. Observability

### What is observed

Every served prediction is written to SQLite (`data/monitoring.db`). A scoring job joins
predictions to ground truth as it arrives and computes rolling MASE against persistence — the
production form of Spec B's headline metric. The dashboard's first panel plots it with a
reference line at **1.0**: above that line the model is worse than doing nothing.

### Running it — no Docker, no root required

Grafana and Prometheus run as userspace tarballs (`scripts/dev_stack.sh` downloads and extracts
pinned releases into `.stack/`, gitignored, ~230 MB on first run; re-runs skip what is already
downloaded). This exists because the development machine has neither Docker nor `sudo`.

```bash
./scripts/dev_stack.sh start
.venv/bin/python -m uvicorn src.serving.app:app --port 8000
# Grafana    http://localhost:3300   (admin / admin)
# Prometheus http://localhost:9090
./scripts/dev_stack.sh stop
```

Grafana listens on port **3300**, not the usual 3000 — port 3000 on the development machine is
occupied by an unrelated `next-server`. Set `GRAFANA_PORT` to override.

### The Docker alternative

`docker-compose.yml` runs the same stack in containers (`docker compose up`, Grafana published on
`3300:3000` to match the local stack and dodge the same occupied port). **This path is
CI-verified rather than run locally** — Docker is not installed on the development machine, so
the compose job in `.github/workflows/ci.yml` is the only place it has actually come up. The
dashboard panels themselves were developed and checked against the local, non-Docker stack.

### The five panels

| panel | datasource | purpose |
|---|---|---|
| Rolling MASE vs persistence | Prometheus (`airq_mase_7d`/`airq_mase_30d`) | the production form of Spec B's headline metric, with a threshold line at 1.0 |
| Data freshness (days) | Prometheus (`airq_data_freshness_days`) | age of the newest gold-table row |
| API health | Prometheus (`http_requests_total`, latency buckets) | request volume and latency |
| Recent predictions | SQLite (`airq-sqlite`) | raw prediction log |
| Scoring history (model vs persistence) | SQLite (`airq-sqlite`) | manually-scored rows, model MASE next to persistence's |

There is deliberately no sixth panel for `P(exceedance)`. Spec B measured the exceedance
classifier's total expected cost at 5:1 against persistence — 787 vs. **581** at h24, 1060 vs.
**839** at h48, 1056 vs. **941** at h72 — and persistence won at every horizon. Alerting uses the
persistence rule, so the dashboard shows that rule rather than a probability gauge that would
look more sophisticated and perform worse. `tests/test_ops_config.py` enforces that no panel
title mentions an exceedance probability.

### Freshness

The freshness gauge currently reads about **6.9 days**. The gold table's newest row is
2026-09-14; upstream's historical dump froze at 2026-01-18, and the realtime feed that is
supposed to fill the gap has not been re-run since. A large number here is the panel doing its
job, not a bug in it.

### The known gap: the MASE panel is not fed by live traffic yet

The API logs every served prediction with `target_date=None`, because the `/predict` request
body carries feature rows without saying which day they are for. The scoring job has nothing to
mature those predictions against, so **the MASE panel is currently populated only by manually
written scores** (3 rows at the time of writing), not by predictions actually served through the
API. Freshness, API health, and the prediction-log panel all populate from live traffic
immediately — only the MASE panel is affected. Wiring a `target_date` through the prediction
request is Spec D's concern.

---

### A limitation of the container path

The freshness gauge needs the gold table. The serving image does not ship `data/`, so
`airq_data_freshness_days` is absent under `docker compose` unless a gold table is mounted
and `GOLD_PATH` points at it. Under `./scripts/dev_stack.sh` it works directly, which is
where the ~6.9-day reading above comes from.

The serving image deliberately does **not** contain the training package. `src/monitoring`
is copied because `app.py` logs predictions through it; `src/models` is not, and a test
(`tests/test_serving_image_contents.py`) asserts the app imports cleanly from exactly the
files the Dockerfile copies and that no `src.models` module is pulled in. That test exists
because Spec C broke the image in precisely this way: `app.py` gained an import the
Dockerfile did not carry, the image built fine, and the container died on startup.

## 13. Roadmap

- **Spec B — honest evaluation. Complete.** All three Spec A defects fixed, persistence /
  seasonal-naive / climatology baselines added, a rolling-origin backtest across 5 folds and
  3 horizons, conformal prediction intervals, an exceedance classifier, and a scoreboard
  (`scripts/run_backtest.py` → `docs/RESULTS.md`) that regenerates every published number.
  Deferred out of Spec B:
  - **The alert task is settled, negatively.** Threshold transfer was diagnosed and fixed
    (date-held-out calibration slice + isotonic calibration); it still loses to persistence
    at every horizon, as does thresholding the Huber forecast. See
    `scripts/compare_alert_rules.py`. Beating it would need a different framing — episode
    onset detection, or exogenous data the RSQA feed does not carry (fire-smoke transport
    forecasts, for instance) — not threshold tuning.
  - Hyperparameter search for the regression candidates (Huber's threshold, tree depths, etc.)
    was not tuned — the scoreboard reports defaults.
  - Per-station retraining for Prophet/LSTM (the series-builder fix makes this possible; it
    has not been re-run at scale).
  - Wiring `src/evaluation/conformal.py` into the backtest scoreboard — it is unit-tested but
    not yet part of `run_backtest.py`'s output.
- **Spec C — operate it. Complete.** Prediction and score logging to SQLite, `/metrics` on the
  API, Prometheus scraping it, a provisioned Grafana dashboard (5 panels), and a
  `docker-compose.yml` reviewers with Docker can run, verified in CI. See
  [Observability](#12-observability). Deferred out of Spec C:
  - The evidently 0.4→0.7 port, still pinning `numpy<2.1`.
  - Postgres as the monitoring backend — the `DATABASE_URL` swap is one variable, unexercised.
  - Wiring `target_date` through the prediction request so the MASE panel populates from live
    traffic instead of manually written scores.
- **Spec D — productionize it.** Champion/challenger model promotion, Terraform-managed
  infrastructure, deployment to Cloud Run.

---

## 14. License

[MIT](LICENSE) © 2026 Ayman Daoud
