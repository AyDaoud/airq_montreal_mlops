"""The serving image must contain every module its entrypoint imports.

Spec C added `from src.monitoring.store import ...` to app.py but the
Dockerfile copied only src/serving and src/features. The image built
fine, then the container died on startup with ModuleNotFoundError and
CI failed at the smoke test.

Docker is not installed on the development machine, so this reproduces
the image's file set from the Dockerfile's own COPY lines and imports
the app against it. It catches a missing module in about a second,
without a container runtime.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

DOCKERFILE = Path("Dockerfile")

# The package root is created by the image, not copied as a unit.
IGNORED_SOURCES = {"requirements-serving.txt"}


def _copy_sources() -> list[str]:
    """The source paths the serving Dockerfile copies into the image."""
    sources = []
    for line in DOCKERFILE.read_text().splitlines():
        match = re.match(r"^COPY\s+(\S+)\s+(\S+)", line.strip())
        if not match:
            continue
        source = match.group(1)
        if source not in IGNORED_SOURCES:
            sources.append(source)
    return sources


def test_dockerfile_copies_the_monitoring_package():
    """Regression: app.py imports src.monitoring.store."""
    assert any("src/monitoring" in s for s in _copy_sources()), _copy_sources()


def test_the_app_imports_with_only_the_files_the_image_carries(tmp_path):
    for source in _copy_sources():
        origin = Path(source)
        if not origin.exists():
            continue  # artifacts/rf is baked in CI, absent in the test job
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target, dirs_exist_ok=True)
        else:
            shutil.copy2(origin, target)

    script = textwrap.dedent("""
        import sys
        sys.path.insert(0, ".")
        import src.serving.app  # noqa: F401
        print("IMPORT_OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert "IMPORT_OK" in result.stdout, (
        "the serving app cannot be imported from the files the Dockerfile "
        f"copies.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_the_serving_image_does_not_need_the_training_package(tmp_path):
    """Splitting the images is pointless if serving imports training."""
    for source in _copy_sources():
        origin = Path(source)
        if not origin.exists():
            continue
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target, dirs_exist_ok=True)
        else:
            shutil.copy2(origin, target)

    script = textwrap.dedent("""
        import sys
        sys.path.insert(0, ".")
        import src.serving.app  # noqa: F401
        leaked = [m for m in sys.modules if m.startswith("src.models")]
        print("LEAKED:", leaked)
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    assert "LEAKED: []" in result.stdout, result.stdout + result.stderr
