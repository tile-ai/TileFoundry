"""One installation, and commands aimed at it.

Named ``smoke_*.py``, so default collection misses them::

    pytest tests/installed -o python_files='smoke_*.py' -q

Every command runs from outside the checkout with ``PYTHONPATH`` removed, or it
would reach unshipped source.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PROBE = """\
import json

import tilefoundry
from tilefoundry.cli import data

print(json.dumps({
    "the tilefoundry package": tilefoundry.__file__,
    "spec": str(data.directory("spec")),
    "models": str(data.directory("models")),
    "tutorial": str(data.directory("tutorial")),
}))
"""

def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """*argv* to completion, with no ``PYTHONPATH`` for it to read a checkout by."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(part) for part in argv],
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="session")
def installation(tmp_path_factory) -> Path:
    """Build the environment under test or reuse ``TF_SMOKE_VENV``.

    Xdist workers must share a prebuilt environment. Per-worker wheel builds
    race in setuptools' checkout-local build directory and can remove files
    another worker is copying.
    """
    reused = os.environ.get("TF_SMOKE_VENV")
    if reused:
        venv = Path(reused)
        if not (venv / "bin" / "tilefoundry").is_file():
            pytest.fail(f"TF_SMOKE_VENV={venv} holds no installed tilefoundry")
        return venv
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail(
            "-n needs one environment for the workers to share: build it and point "
            "TF_SMOKE_VENV at it, as .github/workflows/ci.yml does"
        )

    build = tmp_path_factory.mktemp("installation")
    wheels, venv = build / "wheel", build / "venv"
    for argv in (
        [sys.executable, "-m", "pip", "wheel", REPO, "--no-deps", "--wheel-dir", wheels],
        [sys.executable, "-m", "venv", "--system-site-packages", venv],
    ):
        done = _run(argv, build)
        if done.returncode != 0:
            pytest.fail(f"{' '.join(str(p) for p in argv)}\n{done.stderr}")
    built = sorted(wheels.glob("tilefoundry-*.whl"))
    if len(built) != 1:
        pytest.fail(f"expected one wheel in {wheels}, found {len(built)}")
    done = _run([venv / "bin" / "pip", "install", "--no-deps", built[0]], build)
    if done.returncode != 0:
        pytest.fail(f"installing {built[0].name}\n{done.stderr}")
    return venv


@pytest.fixture(scope="session")
def outside(tmp_path_factory) -> Path:
    """A directory that is not a checkout, for every command to run from."""
    return tmp_path_factory.mktemp("outside")


@pytest.fixture
def tf(installation: Path, outside: Path) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run the installed command: ``tf("models", "qwen3_1_7b")``."""

    def run(*arguments: object) -> subprocess.CompletedProcess[str]:
        return _run([installation / "bin" / "tilefoundry", *arguments], outside)

    return run


@pytest.fixture(scope="session")
def shipped(installation: Path, tmp_path_factory) -> dict[str, str]:
    """Where the installation reads each shipped kind from, as it reports it."""
    where = tmp_path_factory.mktemp("probe")
    done = _run([installation / "bin" / "python", "-c", _PROBE], where)
    if done.returncode != 0:
        pytest.fail(f"probing the installation\n{done.stderr}")
    return json.loads(done.stdout)


@pytest.fixture
def mine(tmp_path) -> Path:
    """A two-file model of one's own, which the installation never shipped."""
    home = tmp_path / "mine"
    home.mkdir()
    source = REPO / "tests" / "fixtures" / "placed"
    (home / "model.py").write_text(
        (source / "square_cpu.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    runtime = home / "runtime_model.py"
    runtime.write_text(
        (source / "square_cpu_runtime.py")
        .read_text(encoding="utf-8")
        .replace("from square_cpu import Mine", "from model import Mine"),
        encoding="utf-8",
    )
    return runtime


@pytest.fixture
def cmine() -> Path:
    """The corpus' unplaced matmul followed by RMS norm."""
    return REPO / "tests" / "fixtures" / "logical" / "matmul_rms_norm.py"


@pytest.fixture
def siblings(tmp_path) -> Callable[[str, str], Path]:
    """Write a model and, beside it, a module that imports it and prints."""

    def write(directory: str, name: str) -> Path:
        home = tmp_path / directory
        home.mkdir()
        (home / "model.py").write_text(
            (
                REPO / "tests" / "fixtures" / "logical" / "matmul_rms_norm.py"
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


def _fixture_path(category: str, name: str) -> Path:
    return REPO / "tests" / "fixtures" / category / name


@pytest.fixture(scope="module")
def square_twin() -> Path:
    return _fixture_path("placed", "square_twin.py")


@pytest.fixture(scope="module")
def weighted_twin() -> Path:
    return _fixture_path("placed", "weighted_twin.py")


@pytest.fixture(scope="module")
def fused_twin() -> Path:
    return _fixture_path("placed", "fused_twin.py")


@pytest.fixture(scope="module")
def orchestrated_twin() -> Path:
    return _fixture_path("logical", "orchestrated_twin.py")


@pytest.fixture(scope="module")
def nested_twin() -> Path:
    return _fixture_path("placed", "nested_twin.py")


@pytest.fixture(scope="module")
def handwritten_twin() -> Path:
    return _fixture_path("logical", "handwritten_twin.py")


@pytest.fixture(scope="module")
def mislabelled_twin() -> Path:
    return _fixture_path("logical", "mislabelled_twin.py")


@pytest.fixture
def cwide() -> Path:
    return _fixture_path("placed", "square_cuda.py")
