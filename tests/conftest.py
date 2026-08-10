"""Configure isolated dumps and devices for the test suite.

Each test gets a ``ContextVar``-backed ``DumpScope`` rooted by file, test, and
xdist worker under gitignored ``test_results``; ``no_dump`` selects a null sink.
Tests are never skipped for missing nvcc or CUDA. On multi-GPU hosts, xdist
workers select devices round-robin before tests touch CUDA, so an unindexed
``"cuda"`` consistently means that worker's device.
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


def pytest_configure(config) -> None:
    extensions = Path(config.rootpath) / ".torch_extensions"
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extensions))
    if hasattr(config, "workerinput"):
        return
    for lock in extensions.glob("*/lock"):
        lock.unlink(missing_ok=True)


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
        test_part = test_part[len("test_") :]
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

    Defaulting every worker to device 0 exhausts that card while others idle.
    Called at import before tests touch CUDA. ``set_device`` is required because
    workers have already imported torch, too late for ``CUDA_VISIBLE_DEVICES``.
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
