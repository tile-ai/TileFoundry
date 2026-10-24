"""Tests and helpers for the owning target."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.core.installed.conftest import REPO
from tests.core.installed.conftest import (
    fused_twin as fused_twin,
)
from tests.core.installed.conftest import (
    handwritten_twin as handwritten_twin,
)
from tests.core.installed.conftest import (
    installation as installation,
)
from tests.core.installed.conftest import (
    mine as mine,
)
from tests.core.installed.conftest import (
    mislabelled_twin as mislabelled_twin,
)
from tests.core.installed.conftest import (
    nested_twin as nested_twin,
)
from tests.core.installed.conftest import (
    orchestrated_twin as orchestrated_twin,
)
from tests.core.installed.conftest import (
    outside as outside,
)
from tests.core.installed.conftest import (
    shipped as shipped,
)
from tests.core.installed.conftest import (
    square_cpu as square_cpu,
)
from tests.core.installed.conftest import (
    square_twin as square_twin,
)
from tests.core.installed.conftest import (
    tf as tf,
)
from tests.core.installed.conftest import (
    weighted_twin as weighted_twin,
)


@pytest.fixture
def cmine() -> Path:
    """The corpus' unplaced matmul followed by RMS norm."""
    return REPO / "tests" / "nvidia" / "cuda" / "fixtures" / "logical" / "matmul_rms_norm.py"


@pytest.fixture
def siblings(tmp_path) -> Callable[[str, str], Path]:
    """Write a model and, beside it, a module that imports it and prints."""

    def write(directory: str, name: str) -> Path:
        home = tmp_path / directory
        home.mkdir()
        (home / "model.py").write_text(
            (
                REPO / "tests" / "nvidia" / "cuda" / "fixtures" / "logical" / "matmul_rms_norm.py"
            ).read_text(encoding="utf-8").replace("class CMine:", f"class {name}:"),
            encoding="utf-8",
        )
        runtime = home / "runtime_model.py"
        runtime.write_text(
            f"from model import {name}\n"
            "if __spec__ is None:\n"
            "    raise RuntimeError('source was not loaded as a module')\n"
            "print('source output')\n",
            encoding="utf-8",
        )
        return runtime

    return write


@pytest.fixture(scope="module")
def mesh_slice_start() -> Path:
    return _fixture_path("placed", "mesh_slice_start.py")


@pytest.fixture(scope="module")
def tuple_projection_diagnostic() -> Path:
    return _fixture_path("diagnostics", "tuple_projection_diagnostic.py")


@pytest.fixture(scope="module")
def dynamic_trip_count() -> Path:
    return _fixture_path("diagnostics", "dynamic_trip_count.py")


@pytest.fixture(scope="module")
def derived_prefill() -> Path:
    return _fixture_path("placed", "derived_prefill.py")


@pytest.fixture
def cwide() -> Path:
    return _fixture_path("placed", "square_cuda.py")


@pytest.fixture(scope="module")
def leaf_weights() -> Path:
    return _fixture_path("placed", "leaf_weights.py")


@pytest.fixture(scope="module")
def hir_composition() -> Path:
    return _fixture_path("logical", "hir_composition.py")


@pytest.fixture(scope="module")
def specialize_through_call() -> Path:
    return _fixture_path("placed", "specialize_through_call.py")


def _fixture_path(category: str, name: str) -> Path:
    return Path(__file__).parents[1] / "fixtures" / category / name
