# Spec C — Observability

- **Date:** 2026-09-20
- **Status:** Approved (pending spec review)
- **Branch:** `spec-c/observability`, based on `main`
- **Predecessors:** Spec A (data foundation), Spec B (honest evaluation) — both merged
- **Successor:** Spec D (automation and cloud)

---

## 1. Context

Specs A and B produced a pipeline that runs and a model that is honestly measured. Nothing observes either of them at runtime.

Two findings from Spec B constrain what this dashboard may show:

- **Huber regression beats persistence**, mean MASE 0.899 at h24, improving to 0.758 at h72, winning 5/5 folds. This is the product.
- **The exceedance classifier does not beat persistence** at any operating point or horizon. Total expected cost at 5:1: persistence 581 / 839 / 941 against the classifier's 787 / 1060 / 1056. The recommendation is the persistence rule.

A dashboard showing a `P(exceedance)` gauge would therefore be displaying a model measured as worse than a one-line rule. **It will not.**

A third finding shapes a panel: the gold table silently carried a **239-day gap** between the frozen historical dump (ending 2026-01-18) and the realtime feed. Nothing surfaced it. A freshness panel would have been red from January.

### 1.1 Environment constraints, verified 2026-09-20

| Constraint | Consequence |
|---|---|
| **No Docker, no sudo** on `remigpu2` (user not in the `sudo` group) | The dev stack runs as userspace binaries |
| Disk was at 100% (5.9 GB free) | Reclaimed 36 GB of pip/uv/npm cache; now 41 GB free |
| Grafana 11.3 tarball, 127 MB — HTTP 200 | Runs from `~/`, no install |
| Prometheus 2.54 tarball, 101 MB — HTTP 200 | Runs from `~/`, no install |
| `frser-sqlite-datasource` v4.0.6, **community-signed** | Usable without the unsigned-plugin escape hatch |
| `prometheus-fastapi-instrumentator` 8.1.0 | Available |

Running Grafana locally is not merely a workaround — it is **better than the Docker path**, because the dashboard can be opened and iterated on rather than written blind as JSON. Spec A's Docker tasks were CI-verified only; a provisioned dashboard verified the same way would be unexamined.

---

## 2. Goals and non-goals

### Goals
1. Every served prediction is persisted, so model quality can be measured in production rather than only in backtest.
2. A nightly job joins predictions to ground truth as it arrives and computes **rolling MASE against persistence** — the production form of Spec B's headline metric.
3. A Grafana dashboard that a reviewer can bring up with one command.
4. Operational metrics from the API: latency, request rate, error rate.
5. A freshness alarm that would have caught the 239-day gap.

### Non-goals
- **The evidently 0.4 → 0.7 port.** It is a rewrite of working code that serves no panel here, and it is what pins `numpy<2.1`. It gets its own small spec.
- Postgres. SQLite only for now (§4.1); the swap is a URL change.
- Streamlit. Rejected as the primary dashboard.
- Anything in Spec D (champion/challenger, Terraform, Cloud Run).

---

## 3. Architecture

```
FastAPI ──/metrics──────► Prometheus ──┐
   │                                   ├──► Grafana  :3000
   └──prediction log──► SQLite ────────┘
                          ▲
       nightly job ───────┘   join predictions to truth,
                              compute rolling MASE vs persistence
```

Two Grafana datasources, deliberately:

- **Prometheus** — operational metrics and the headline gauges (rolling MASE, freshness days). Time-series shaped, scraped every 15s.
- **SQLite** — detail panels needing SQL: forecast vs actual per station, the prediction log itself.

Splitting them avoids forcing per-station detail into Prometheus labels, which would explode cardinality.

---

## 4. Components

### 4.1 `src/monitoring/store.py` — the prediction log

SQLAlchemy over a `DATABASE_URL`, defaulting to `sqlite:///data/monitoring.db`. The same code runs against Postgres by changing that one variable, which is how Spec D will deploy it. Tests use `sqlite:///:memory:` and stay offline.

Table `predictions`:

| column | type | note |
|---|---|---|
| `id` | int pk | |
| `request_id` | str | uuid4 per request |
| `served_at` | datetime (UTC) | |
| `station_id` | int, nullable | null when the caller sends no station |
| `target_date` | date, nullable | the day predicted |
| `prediction` | float | |
| `model_name` | str | e.g. `huber_level` |
| `model_version` | str | git sha of the serving image |
| `features_json` | text | the row as received, for replay |

Table `scores` (written nightly):

| column | type |
|---|---|
| `scored_at` | datetime (UTC) |
| `window_days` | int (7 or 30) |
| `n` | int |
| `mae_model` | float |
| `mae_persistence` | float |
| `mase` | float |

### 4.2 `src/serving/app.py` — instrumentation

- `prometheus-fastapi-instrumentator` exposes `/metrics` with request count, latency histogram and error rate.
- Every `/predict` call writes one row per prediction to the log. **Logging failures must never fail the request** — the write is wrapped and its failure incremented on a counter instead.
- Custom gauges read from the `scores` table: `airq_mase_7d`, `airq_mase_30d`, `airq_data_freshness_days`.

### 4.3 `src/monitoring/scoring.py` — closing the loop

A function, callable from the CLI and from a Prefect task, that:
1. reads `predictions` whose `target_date` now has ground truth in the gold table,
2. computes model MAE and persistence MAE over 7- and 30-day windows,
3. writes a `scores` row per window.

This is the difference between "I used Evidently" and "I built a feedback loop": the metric is computed from what was actually served, not from a batch artifact.

### 4.4 `ops/` — the stack

```
ops/grafana/provisioning/datasources/datasources.yml
ops/grafana/provisioning/dashboards/dashboards.yml
ops/grafana/dashboards/airq.json
ops/prometheus/prometheus.yml
```

### 4.5 `scripts/dev_stack.sh` — run it without Docker

Downloads Grafana and Prometheus tarballs into `.stack/` (gitignored), points them at `ops/`, starts both, prints the URLs. Idempotent: skips a download that already exists.

### 4.6 `docker-compose.yml`

The same stack for reviewers who do have Docker: api, prometheus, grafana. CI brings it up and health-checks it. **The README will state that the compose path is CI-verified but the panels were developed against the local stack.**

---

## 5. The dashboard

| # | Panel | Source | Why it earns its place |
|---|---|---|---|
| 1 | **Rolling MASE vs persistence** (7d, 30d), with a reference line at 1.0 | Prometheus | Is the model still beating "tomorrow = today" *this week*? Operationalises Spec B's entire result and catches decay. |
| 2 | Forecast vs actual, per station | SQLite | The product. |
| 3 | **Data freshness** — days since the newest gold row, red above 2 | Prometheus | Would have been red from January during the 239-day gap. |
| 4 | API health — p95 latency, request rate, error rate | Prometheus | Standard ops. |
| 5 | Exceedance alerts fired — **persistence rule**, labelled as such | SQLite | Shows what the evidence supports, not what looks impressive. |

Panel 1 carries a permanent reference line at MASE = 1.0. Above it, the model is worse than doing nothing, and the panel says so in plain words rather than leaving the reader to interpret a number.

---

## 6. Decisions

| # | Decision | Rationale |
|---|---|---|
| C1 | Grafana + Prometheus as userspace tarballs | No Docker, no sudo; and the dashboard can actually be opened and iterated |
| C2 | SQLite via `DATABASE_URL`, Postgres deferred | Keeps tests offline; the swap is one variable |
| C3 | Two datasources (Prometheus + SQLite) | Per-station detail as Prometheus labels would explode cardinality |
| C4 | Prediction logging must never fail a request | Observability is not worth an outage; failures go to a counter |
| C5 | Panel 5 uses the persistence rule | The classifier was measured as worse; the dashboard reflects evidence |
| C6 | Evidently port deferred | Rewrite of working code, serves no panel, deserves its own spec |
| C7 | `docker-compose.yml` ships but is CI-verified only | Honest about what was and was not run |

---

## 7. Testing

All tests stay offline, consistent with Specs A and B.

- `store.py` — round-trip against `sqlite:///:memory:`; a logging failure does not raise; concurrent writes do not corrupt.
- `scoring.py` — MASE computed from a synthetic log matches a hand-computed value; a window with no matured ground truth yields no row rather than a divide-by-zero; persistence MAE is computed on the same rows as the model's.
- `app.py` — `/metrics` returns Prometheus text format; `/predict` still succeeds when the store raises; one prediction produces exactly one row.
- `ops/grafana/dashboards/airq.json` — parses as JSON and every panel references a datasource that `datasources.yml` defines. Cheap, and catches the most common provisioning error.

---

## 8. Acceptance criteria

1. `scripts/dev_stack.sh` brings up Grafana and Prometheus with no Docker and no sudo.
2. Grafana at `localhost:3000` shows all five panels populated with real data.
3. `/metrics` exposes request metrics plus `airq_mase_7d`, `airq_mase_30d`, `airq_data_freshness_days`.
4. A `/predict` call writes exactly one row per prediction; a store failure logs a counter and still returns 200.
5. `python -m src.monitoring.scoring` writes `scores` rows and the MASE matches a backtest-derived value on the same window.
6. The freshness gauge reports the true age of the gold table — currently large, and correctly so.
7. Full suite passes offline with `data/` absent.
8. README documents the stack, the two datasources, and which parts are CI-verified rather than run.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Disk fills again (41 GB free, 96% used) | `.stack/` is gitignored; tarballs are ~230 MB; the script skips existing downloads |
| SQLite write contention under concurrent requests | WAL mode; writes are small and single-row; Postgres is the documented escape hatch |
| Grafana plugin install needs network at run time | `dev_stack.sh` installs it once into `.stack/`; failure is reported, not silent |
| Panels look fine but query nothing | The JSON test asserts datasource references resolve; panels are verified against the running stack, not just written |
| Scope creep into the evidently port | Explicit non-goal (C6) |
