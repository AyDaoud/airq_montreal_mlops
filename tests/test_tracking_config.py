import re
from pathlib import Path

from src.tracking import DEFAULT_TRACKING_URI, resolve_tracking_uri

FORBIDDEN = re.compile(r"[A-Za-z]:[/\\]Users[/\\]", re.IGNORECASE)


def test_no_absolute_user_paths_in_source():
    """Regression guard: no machine-specific path may return to src/ or scripts/."""
    offenders = []
    for path in list(Path("src").rglob("*.py")) + list(Path("scripts").rglob("*.py")):
        if FORBIDDEN.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert offenders == [], f"hardcoded user paths found in {offenders}"


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env.db")
    assert resolve_tracking_uri("sqlite:///cli.db") == "sqlite:///cli.db"


def test_environment_is_used_when_no_argument(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    assert resolve_tracking_uri(None) == "sqlite:///custom.db"


def test_defaults_to_repo_relative(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri(None) == DEFAULT_TRACKING_URI
    assert DEFAULT_TRACKING_URI == "sqlite:///mlflow.db"


def test_empty_string_is_treated_as_unset(monkeypatch):
    """An exported-but-empty env var must not become the tracking URI."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    assert resolve_tracking_uri(None) == DEFAULT_TRACKING_URI


def test_training_script_uses_the_shared_resolver():
    source = Path("scripts/train_daily_iqa.py").read_text(encoding="utf-8")
    assert "resolve_tracking_uri" in source
    assert "C:/Users" not in source


def test_monitoring_uses_the_shared_resolver():
    source = Path("src/monitoring/check_iqa.py").read_text(encoding="utf-8")
    assert "resolve_tracking_uri" in source
