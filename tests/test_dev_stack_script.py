import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/dev_stack.sh")


def test_script_exists_and_is_executable():
    assert SCRIPT.exists()
    assert os.stat(SCRIPT).st_mode & stat.S_IXUSR, "not executable"


def test_script_is_valid_bash():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_script_downloads_into_the_ignored_stack_directory():
    text = SCRIPT.read_text()
    assert ".stack" in text
    assert ".stack/" in Path(".gitignore").read_text()


def test_script_requires_neither_sudo_nor_docker():
    """The whole point: this machine has neither."""
    text = SCRIPT.read_text()
    assert "sudo" not in text
    assert "docker" not in text.lower()


def test_script_pins_versions():
    """An unpinned download turns a working stack into a moving target."""
    text = SCRIPT.read_text()
    assert "GRAFANA_VERSION=" in text
    assert "PROMETHEUS_VERSION=" in text


def test_script_exports_both_provisioning_variables():
    """Grafana interpolates these in the provisioning files; unset means
    a datasource pointing at a literal '${AIRQ_DB_PATH}'."""
    text = SCRIPT.read_text()
    assert "AIRQ_DB_PATH" in text
    assert "AIRQ_DASHBOARD_PATH" in text


def test_script_supports_start_and_stop():
    text = SCRIPT.read_text()
    assert "start)" in text
    assert "stop)" in text
