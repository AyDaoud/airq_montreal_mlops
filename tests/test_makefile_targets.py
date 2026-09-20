"""The makefile must never invoke a bare Python tool.

`make setup` creates .venv but does not activate it, so `pytest`, `flake8`,
`black` and `uvicorn` are not on PATH afterwards. Targets calling them
directly fail on a fresh clone with "No such file or directory" - which is
exactly what the README quickstart told people to run.
"""

from __future__ import annotations

import re
from pathlib import Path

BARE_TOOLS = ("pytest", "flake8", "black", "uvicorn", "pip")
ALLOWED_BARE = ("python3 -m venv",)  # bootstrapping the venv itself


def _recipe_lines() -> list[tuple[int, str]]:
    """Every tab-indented recipe line in the makefile."""
    text = Path("makefile").read_text().splitlines()
    return [(n, line) for n, line in enumerate(text, 1) if line.startswith("\t")]


def test_no_recipe_invokes_a_bare_python_tool():
    offenders = []
    for number, line in _recipe_lines():
        stripped = line.strip()
        if any(stripped.startswith(a) for a in ALLOWED_BARE):
            continue
        for tool in BARE_TOOLS:
            if re.match(rf"^{tool}\b", stripped):
                offenders.append(f"makefile:{number}: {stripped}")
    assert offenders == [], (
        "these recipes call a tool that is not on PATH after `make setup`; "
        "use $(PY) -m <tool> instead:\n" + "\n".join(offenders)
    )


def test_no_recipe_invokes_bare_python():
    """`python` may be the wrong interpreter entirely (conda base, py2)."""
    offenders = [
        f"makefile:{n}: {line.strip()}"
        for n, line in _recipe_lines()
        if re.match(r"^python\b(?!3 -m venv)", line.strip())
    ]
    assert offenders == [], "use $(PY) instead of bare python:\n" + "\n".join(offenders)


def test_py_is_defined_before_any_recipe_uses_it():
    text = Path("makefile").read_text()
    definition = text.index("PY :=")
    first_use = text.index("$(PY)")
    assert definition < first_use, "PY is used before it is defined"


def test_quickstart_targets_all_exist():
    """Every target the README quickstart names must be real."""
    text = Path("makefile").read_text()
    for target in ("setup", "data", "test", "run-api", "bake-model", "hooks"):
        assert re.search(
            rf"^{target}:", text, re.MULTILINE
        ), f"missing target: {target}"
