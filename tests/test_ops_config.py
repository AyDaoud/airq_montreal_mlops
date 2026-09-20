import json
from pathlib import Path

import yaml

DASHBOARD = Path("ops/grafana/dashboards/airq.json")
DATASOURCES = Path("ops/grafana/provisioning/datasources/datasources.yml")
PROMETHEUS = Path("ops/prometheus/prometheus.yml")


def test_prometheus_config_parses_and_scrapes_the_api():
    config = yaml.safe_load(PROMETHEUS.read_text())
    jobs = {j["job_name"] for j in config["scrape_configs"]}
    assert "airq-api" in jobs


def test_datasources_define_prometheus_and_sqlite():
    config = yaml.safe_load(DATASOURCES.read_text())
    types = {d["type"] for d in config["datasources"]}
    assert "prometheus" in types
    assert "frser-sqlite-datasource" in types


def test_dashboard_is_valid_json_with_panels():
    dashboard = json.loads(DASHBOARD.read_text())
    assert dashboard["panels"], "dashboard has no panels"


def test_every_panel_references_a_defined_datasource():
    """The commonest provisioning error: a panel pointing at a datasource
    that was never declared renders an EMPTY chart, not an error."""
    declared = {
        d["uid"] for d in yaml.safe_load(DATASOURCES.read_text())["datasources"]
    }
    dashboard = json.loads(DASHBOARD.read_text())
    unknown = []
    for panel in dashboard["panels"]:
        source = panel.get("datasource")
        uid = source.get("uid") if isinstance(source, dict) else source
        if uid and uid not in declared:
            unknown.append(f"{panel.get('title')!r} -> {uid}")
    assert unknown == [], f"panels reference undeclared datasources: {unknown}"


def test_every_panel_queries_something():
    """A panel with no targets renders an empty box and looks like 'no data'."""
    dashboard = json.loads(DASHBOARD.read_text())
    empty = [p["title"] for p in dashboard["panels"] if not p.get("targets")]
    assert empty == [], f"panels with no query: {empty}"


def test_the_mase_panel_has_a_threshold_at_one():
    """Above 1.0 the model is worse than doing nothing. The panel must say so."""
    dashboard = json.loads(DASHBOARD.read_text())
    mase_panels = [p for p in dashboard["panels"] if "MASE" in (p.get("title") or "")]
    assert mase_panels, "no MASE panel"
    steps = mase_panels[0]["fieldConfig"]["defaults"]["thresholds"]["steps"]
    assert any(s.get("value") == 1.0 for s in steps), steps


def test_no_panel_shows_an_ml_exceedance_probability():
    """Spec B measured the classifier as worse than the persistence rule at
    every horizon. The dashboard shows the rule, not a P(exceedance) gauge."""
    dashboard = json.loads(DASHBOARD.read_text())
    titles = " ".join((p.get("title") or "").lower() for p in dashboard["panels"])
    assert "p(exceed" not in titles
    assert "probability" not in titles


def test_prometheus_panels_query_metrics_that_actually_exist():
    """Guards the defect found in Task 3: gauges that were built but never
    wired, so the panels queried series that never appeared."""
    dashboard = json.loads(DASHBOARD.read_text())
    known = {
        "airq_mase_7d",
        "airq_mase_30d",
        "airq_data_freshness_days",
        "http_request_duration_seconds_bucket",
        "http_requests_total",
    }
    unknown = []
    for panel in dashboard["panels"]:
        source = panel.get("datasource") or {}
        if (
            source.get("uid") if isinstance(source, dict) else source
        ) != "airq-prometheus":
            continue
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            if expr and not any(metric in expr for metric in known):
                unknown.append(f"{panel['title']!r}: {expr}")
    assert unknown == [], f"panels query unknown metrics: {unknown}"


COMPOSE_PROMETHEUS = Path("ops/prometheus/prometheus-compose.yml")
COMPOSE = Path("docker-compose.yml")


def test_compose_prometheus_targets_the_service_not_localhost():
    """Inside the compose network, localhost is the Prometheus container
    itself - it would scrape nothing and report success."""
    config = yaml.safe_load(COMPOSE_PROMETHEUS.read_text())
    targets = [
        t
        for job in config["scrape_configs"]
        for sc in job["static_configs"]
        for t in sc["targets"]
    ]
    assert any("api:" in t for t in targets), targets
    assert not any("localhost" in t for t in targets), targets


def test_compose_mounts_the_compose_specific_prometheus_config():
    compose = yaml.safe_load(COMPOSE.read_text())
    mounts = compose["services"]["prometheus"]["volumes"]
    assert any("prometheus-compose.yml" in m for m in mounts), mounts


def test_compose_publishes_grafana_on_3300():
    """Port 3000 is occupied on the development machine."""
    compose = yaml.safe_load(COMPOSE.read_text())
    assert "3300:3000" in compose["services"]["grafana"]["ports"]


def test_compose_sets_both_provisioning_variables():
    compose = yaml.safe_load(COMPOSE.read_text())
    env = compose["services"]["grafana"]["environment"]
    assert "AIRQ_DB_PATH" in env
    assert "AIRQ_DASHBOARD_PATH" in env
