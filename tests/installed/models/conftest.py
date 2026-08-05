"""The model source the installation ships, copied out and asked about.

The directory comes from ``tilefoundry models <name> --source`` and is copied whole
before anything is asked of it, which is the use ``--source`` promises. Nothing here
joins an installation path together, so a file the packaging manifest forgot to ship
is missing here too instead of being reached in the checkout.

The test process imports ``tests.models.<name>.case`` for the case list. That is the
test side's own glue -- which selectors to ask about, and at what extents -- and it
never reaches the source under test: that arrives only as a directory the command
named, and every question about it is asked by running the command.
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
