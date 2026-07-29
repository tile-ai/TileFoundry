"""The shared installed-environment build, when the build does not work."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from tests.installed_env import InstalledEnv, shared_build


def _explode(directory: Path) -> InstalledEnv:
    raise subprocess.CalledProcessError(
        1, ["pip", "wheel", str(directory)], stderr="no wheel was built"
    )


def _must_not_run(directory: Path) -> InstalledEnv:
    raise AssertionError(f"a waiting process rebuilt {directory}")


def test_a_failed_build_fails_the_processes_waiting_on_it(tmp_path) -> None:
    """The waiter raises the builder's own diagnostic, and does it now.

    Timed against a timeout it would otherwise sit out: waiting for the deadline
    reports the same failure much later, when the run has nothing to show for it.
    """
    shared = tmp_path / "installed"
    with pytest.raises(subprocess.CalledProcessError):
        shared_build(shared, builder=_explode, timeout=30.0)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as waited:
        shared_build(shared, builder=_must_not_run, timeout=30.0)

    assert "no wheel was built" in str(waited.value)
    assert time.monotonic() - started < 5.0
