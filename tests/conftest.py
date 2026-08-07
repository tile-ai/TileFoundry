"""Test infrastructure: autouse ``DumpScope`` wiring.

Autouse fixture: each test runs inside a ``DumpScope(dumper=FileDumper(...),
flags=ALL)`` rooted at ``test_results/{file_stem}/{test_name}`` for the
common single-worker case. When pytest-xdist uses additional workers, the
worker id is appended to the leaf test directory name so workers still do
not collide.
Two orthogonal isolation layers:

- filesystem path keyed by pytest file stem first, with optional
  ``__{worker_id}`` suffix on the test leaf so parallel pytest-xdist
  workers do not collide.
- ``ContextVar``-backed scope so multiple tests in the same process — and
  any threads / asyncio Tasks they spawn — see only their own scope.

Tests that produce a lot of dump output and want to opt out can mark
themselves with ``pytest.mark.no_dump``; the autouse fixture then
installs ``NullDumper`` for that test, leaving the rest of the wiring
intact.

All tests run by default; there is no marker-based skipping. Tests that
need nvcc or a CUDA device will fail (not skip) on a machine that lacks
them.

On a machine with more than one CUDA device, each xdist worker is given one of
them round-robin (see ``_spread_workers_across_devices``). Nothing in the tests
names a device index, so ``"cuda"`` means whichever one the worker was given.

``test_results/`` is gitignored.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from tilefoundry.dump import DumpFlags, DumpScope, FileDumper, NullDumper
from tilefoundry.target.base import _TARGET_CLASSES, _TARGET_PROVIDERS

_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "test_results"
_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


def _split_nodeid(nodeid: str) -> tuple[str, str | None]:
    """Return ``(file_stem, test_name_or_none)`` for a pytest nodeid.

    ``tests/e2e/test_mma_runtime.py::test_mma_sm80_16x8x16_bf16_matches_torch_matmul``
    → ``("test_mma_runtime", "mma_sm80_16x8x16_bf16_matches_torch_matmul")``.

    Drops the ``tests/.../`` directory prefix and the ``.py`` suffix from
    the file part, and strips a redundant leading ``test_`` from the test
    name (the file stem already starts with ``test_``). Parametrize
    brackets and other unsafe chars get sanitized to ``_``.
    """
    file_part, sep, test_part = nodeid.partition("::")
    file_stem = _SANITIZE.sub("_", Path(file_part).stem).strip("_")
    if not sep:
        return file_stem, None
    if test_part.startswith("test_"):
        test_part = test_part[len("test_"):]
    safe_test = _SANITIZE.sub("_", test_part).strip("_")
    return file_stem, safe_test


def _dump_relpath(nodeid: str, worker_id: str) -> Path:
    """Map a pytest nodeid to the per-test dump root under ``test_results/``.

    The top-level visible directory is always the pytest file stem. For
    xdist workers other than ``master``, append ``__{worker_id}`` to the
    test leaf to keep different workers isolated without inserting a
    worker directory above the pytest file level.
    """
    file_stem, test_name = _split_nodeid(nodeid)
    if test_name is None:
        return Path(file_stem)
    leaf = test_name if worker_id == "master" else f"{test_name}__{worker_id}"
    return Path(file_stem) / leaf


def _spread_workers_across_devices() -> None:
    """Give each xdist worker one CUDA device, round-robin.

    Every worker defaulting to device 0 puts the whole suite's model weights on one
    card while the others idle: measured, eight workers held 6 to 33 GB each on one
    140 GB device and the production-sized decoder References ran it out of memory.
    The models under test are real ones, so the fix is to use the machine that is
    there rather than to shrink them.

    Called at import so it lands before any test touches CUDA. `set_device` rather
    than `CUDA_VISIBLE_DEVICES`, because the environment variable is read when the
    driver initialises and a worker process has already imported torch by now.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None or not worker.startswith("gw"):
        return
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    count = torch.cuda.device_count()
    if count > 1:
        torch.cuda.set_device(int(worker[2:]) % count)


_spread_workers_across_devices()


@pytest.fixture(autouse=True)
def _tilefoundry_dump_scope(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("no_dump") is not None:
        with DumpScope(dumper=NullDumper, flags=DumpFlags.NONE) as scope:
            yield scope
        return

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    test_root = _RESULTS_ROOT / _dump_relpath(request.node.nodeid, worker_id)
    dumper = FileDumper(test_root)
    with DumpScope(dumper=dumper, flags=DumpFlags.ALL) as scope:
        yield scope


@pytest.fixture(autouse=True)
def _restore_target_registry():
    classes = dict(_TARGET_CLASSES)
    providers = dict(_TARGET_PROVIDERS)
    try:
        yield
    finally:
        _TARGET_CLASSES.clear()
        _TARGET_CLASSES.update(classes)
        _TARGET_PROVIDERS.clear()
        _TARGET_PROVIDERS.update(providers)

