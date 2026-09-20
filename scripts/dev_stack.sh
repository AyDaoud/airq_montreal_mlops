#!/usr/bin/env bash
# Run Grafana and Prometheus locally with no container runtime and no root.
#
# This machine has neither, so the stack is userspace tarballs extracted
# into .stack/ (gitignored). Re-running skips downloads that already exist.
#
#   ./scripts/dev_stack.sh start
#   ./scripts/dev_stack.sh stop
set -euo pipefail

GRAFANA_VERSION=11.3.0
PROMETHEUS_VERSION=2.54.1
SQLITE_PLUGIN_VERSION=4.0.6

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK="$ROOT/.stack"
GRAFANA_DIR="$STACK/grafana-v$GRAFANA_VERSION"
PROM_DIR="$STACK/prometheus-$PROMETHEUS_VERSION.linux-amd64"

export AIRQ_DB_PATH="$ROOT/data/monitoring.db"
export AIRQ_DASHBOARD_PATH="$ROOT/ops/grafana/dashboards"

fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "[skip] $(basename "$out") already downloaded"
    return
  fi
  echo "[get]  $(basename "$out")"
  curl -fsSL -o "$out" "$url"
}

install_stack() {
  mkdir -p "$STACK"
  fetch "https://dl.grafana.com/oss/release/grafana-$GRAFANA_VERSION.linux-amd64.tar.gz" \
        "$STACK/grafana.tar.gz"
  fetch "https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" \
        "$STACK/prometheus.tar.gz"
  [ -d "$GRAFANA_DIR" ] || tar -xzf "$STACK/grafana.tar.gz" -C "$STACK"
  [ -d "$PROM_DIR" ]    || tar -xzf "$STACK/prometheus.tar.gz" -C "$STACK"

  if [ ! -d "$STACK/plugins/frser-sqlite-datasource" ]; then
    echo "[get]  sqlite datasource plugin"
    mkdir -p "$STACK/plugins"
    "$GRAFANA_DIR/bin/grafana" cli \
      --pluginsDir "$STACK/plugins" \
      plugins install frser-sqlite-datasource "$SQLITE_PLUGIN_VERSION" \
      || echo "[warn] plugin install failed; SQL panels will be empty (Prometheus panels still work)"
  fi
}

start() {
  install_stack
  mkdir -p "$STACK/logs" "$ROOT/data"

  "$PROM_DIR/prometheus" \
    --config.file="$ROOT/ops/prometheus/prometheus.yml" \
    --storage.tsdb.path="$STACK/prometheus-data" \
    --web.listen-address=":9090" \
    > "$STACK/logs/prometheus.log" 2>&1 &
  echo $! > "$STACK/prometheus.pid"

  GF_PATHS_PROVISIONING="$ROOT/ops/grafana/provisioning" \
  GF_PATHS_DATA="$STACK/grafana-data" \
  GF_PATHS_LOGS="$STACK/logs" \
  GF_PATHS_PLUGINS="$STACK/plugins" \
  GF_SERVER_HTTP_PORT="${GRAFANA_PORT:-3300}" \
  "$GRAFANA_DIR/bin/grafana" server \
    --homepath "$GRAFANA_DIR" \
    --config "$ROOT/ops/grafana/grafana.ini" \
    > "$STACK/logs/grafana.log" 2>&1 &
  echo $! > "$STACK/grafana.pid"

  echo
  echo "  Grafana     http://localhost:${GRAFANA_PORT:-3300}  (admin / admin)"
  echo "  Prometheus  http://localhost:9090"
  echo "  logs        $STACK/logs/"
  echo
  echo "  Start the API separately so Prometheus has something to scrape:"
  echo "    .venv/bin/python -m uvicorn src.serving.app:app --port 8000"
}

stop() {
  for name in grafana prometheus; do
    if [ -f "$STACK/$name.pid" ]; then
      kill "$(cat "$STACK/$name.pid")" 2>/dev/null && echo "[ok]   stopped $name" || true
      rm -f "$STACK/$name.pid"
    fi
  done
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 {start|stop}" >&2; exit 2 ;;
esac
