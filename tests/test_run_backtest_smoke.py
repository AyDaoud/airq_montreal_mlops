import subprocess
import sys

import pandas as pd

from scripts.run_backtest import render_markdown


def test_script_exposes_help():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_backtest", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--out" in result.stdout
    assert "--folds" in result.stdout


def test_render_markdown_produces_a_table():
    frame = pd.DataFrame(
        {
            "candidate": ["persistence", "huber_level"],
            "horizon": [1, 1],
            "mase": [1.0, 0.9],
            "mae": [7.0, 6.3],
            "fold": [1, 1],
        }
    )
    out = render_markdown(frame, cost_ratio=5.0, extra_sections=None)
    assert "huber_level" in out
    assert "MASE" in out
    assert "|" in out


def test_render_markdown_flags_when_nothing_beats_persistence():
    frame = pd.DataFrame(
        {
            "candidate": ["persistence", "rf_level"],
            "horizon": [1, 1],
            "mase": [1.0, 1.2],
            "mae": [7.0, 8.4],
            "fold": [1, 1],
        }
    )
    out = render_markdown(frame, cost_ratio=5.0, extra_sections=None)
    assert "no model beat persistence" in out.lower()
