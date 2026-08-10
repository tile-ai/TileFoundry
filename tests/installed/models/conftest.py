"""The model source the installation ships, copied out and asked about.

The ``models <name> --source`` directory is copied whole before use. No test joins
installation paths, so omitted package files cannot leak in from the checkout.
The local case list supplies selectors and extents only; every question about
model source runs against the copied directory named by the installed command.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def _copied_sources() -> dict[str, Path]:
    """One copy per model, not one per case.

    Every ``tilefoundry`` invocation pays its own interpreter start and import --
    measured at about 2.4 s -- so asking ``--source`` again for each of a model's
    cases would cost more than all of that model's questions put together. The copy
    is only ever read from, so the cases can share it.
    """
    return {}


@pytest.fixture
def shipped_source(tf, tmp_path_factory, _copied_sources) -> Callable[[str], Path]:
    """``tilefoundry models <name> --source``, copied whole into an empty directory."""

    def copy(model: str) -> Path:
        if model not in _copied_sources:
            done = tf("models", model, "--source")
            assert done.returncode == 0, done.stderr
            named = done.stdout.splitlines()[0]
            source = Path(named)
            assert source.is_absolute(), (
                f"--source named {named!r}, which is not an absolute directory"
            )
            copied = tmp_path_factory.mktemp(f"shipped-{model}-") / model
            shutil.copytree(source, copied)
            _copied_sources[model] = copied
        return _copied_sources[model]

    return copy
