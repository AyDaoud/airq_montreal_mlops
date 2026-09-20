import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_help_lists_all_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "src.data.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for command in ("ingest", "build", "all"):
        assert command in result.stdout


def test_build_without_bronze_exits_with_actionable_message(tmp_path):
    """Running build in a directory with no bronze data must fail loudly."""
    result = subprocess.run(
        [sys.executable, "-m", "src.data.cli", "build"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode != 0
    assert "src.data.cli" in result.stdout + result.stderr
