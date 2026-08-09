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
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

#: The checkout to build a wheel from.
REPO = Path(__file__).resolve().parents[2]

# Every kind that ships, looked up inside the installation.
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

# A model and its twin in a sibling module: the two-file layout.
_MINE_MODEL = '''\
from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CpuTarget


@module(entry="main", target=CpuTarget(), topologies=(Topology("cta", 168),))
class Mine:

    @func
    def main(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            local = tf.reshard(x, (168 @ cta.block,), "rmem")
            return tf.reshard(tf.square(local), (168 @ cta.block,), "gmem")
'''

_MINE_RUNTIME = '''\
from model import Mine
from tilefoundry.runtime import runtime_func, runtime_module


@runtime_module(Mine)
class MineTwin:
    @runtime_func
    def main(self, x):
        return x * x
'''

_CMINE_MODEL = '''\
from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import matmul, rms_norm
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


@module(entry="root", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 128)))
class CMine:

    @func
    def root(
        x: Tensor[(16, 16), "bf16"],
        w: Tensor[(16, 16), "bf16"],
        weight: Tensor[(16,), "f32"],
    ) -> Tensor[(16, 16), "bf16"]:
        h = matmul(x, w)
        return rms_norm(h, weight)

    @module(entry="inner")
    class child:
        @func
        def inner(
            x: Tensor[(16, 16), "bf16"],
            w: Tensor[(16, 16), "bf16"],
            weight: Tensor[(16,), "f32"],
        ) -> Tensor[(16, 16), "bf16"]:
            h = matmul(x, w)
            return rms_norm(h, weight)
'''


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
    """The environment under test; ``TF_SMOKE_VENV`` reuses one already built."""
    reused = os.environ.get("TF_SMOKE_VENV")
    if reused:
        venv = Path(reused)
        if not (venv / "bin" / "tilefoundry").is_file():
            pytest.fail(f"TF_SMOKE_VENV={venv} holds no installed tilefoundry")
        return venv
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # This fixture is session-scoped per worker, so every worker would build a
        # wheel from the same checkout at once, and setuptools' build directory sits
        # inside it: one build removes what another is still copying into, and the
        # whole run errors in the fixture with a missing file.
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
    (home / "model.py").write_text(_MINE_MODEL, encoding="utf-8")
    runtime = home / "runtime_model.py"
    runtime.write_text(_MINE_RUNTIME, encoding="utf-8")
    return runtime


@pytest.fixture
def cmine(tmp_path) -> Path:
    """A CUDA model of one's own."""
    path = tmp_path / "cmine.py"
    path.write_text(textwrap.dedent(_CMINE_MODEL), encoding="utf-8")
    return path


@pytest.fixture
def siblings(tmp_path) -> Callable[[str, str], Path]:
    """Write a model and, beside it, a module that imports it and prints."""

    def write(directory: str, name: str) -> Path:
        home = tmp_path / directory
        home.mkdir()
        (home / "model.py").write_text(
            textwrap.dedent(_CMINE_MODEL).replace("class CMine:", f"class {name}:"),
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


_TWIN_SOURCE = '''
from dataclasses import replace

from tilefoundry import module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, func, tf
from tilefoundry.runtime import RuntimeModule, runtime_func, runtime_module
from tilefoundry.target import CpuTarget


@module(entry="main", target=CpuTarget(), topologies=(Topology("cta", 168),))
class Model:

    @func
    def main(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            return tf.reshard(squared, (168 @ cta.block,), "gmem")

    @func
    def zeroed(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            nothing = tf.sub(x_local, x_local)
            return tf.reshard(nothing, (168 @ cta.block,), "gmem")


@runtime_module(Model)
class Twin:
    @runtime_func
    def main(self, x):
        return x * x

    @runtime_func
    def zeroed(self, x):
        return x - x


@runtime_module(Model)
class Drifted:
    @runtime_func
    def main(self, x):
        return x * x + 0.5

    @runtime_func
    def zeroed(self, x):
        return x - x


@module(entry="scaled", topologies=(Topology("cta", 168),))
class Weighted:

    @func
    def scaled(
        x: Tensor[(168,), "f32"], w: ConstTensor[(168,), "f32"]
    ) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            w_local = tf.reshard(w, (168 @ cta.block,), "rmem")
            weighted = tf.mul(x_local, w_local)
            return tf.reshard(weighted, (168 @ cta.block,), "gmem")


@runtime_module(Weighted)
class WeightedTwin:
    @runtime_func
    def scaled(self, x, w):
        return x * w


# Only a root may declare a target, so the child above declares none and this
# copy is where the standalone check gets its CpuTarget.
WeightedRoot = replace(Weighted, target=CpuTarget())


@runtime_module(WeightedRoot)
class WeightedRootTwin:
    @runtime_func
    def scaled(self, x, w):
        return x * w


@module(entry="fused", target=CpuTarget(), topologies=(Topology("cta", 168),))
class Fused:

    @func
    def fused(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            shifted = tf.sub(squared, x_local)
            return tf.reshard(shifted, (168 @ cta.block,), "gmem")


@runtime_module(Fused)
class FusedTwin:
    @runtime_func
    def fused(self, x):
        return x * x - x


@module(entry="add_pair", target=CpuTarget(), topologies=(Topology("cta", 168),))
class Orchestrated:

    @func
    def add_pair(
        x: Tensor[(168,), "f32"], a: Tensor[(168,), "f32"], b: Tensor[(168,), "f32"]
    ) -> Tensor[(168,), "f32"]:
        return x + a + b

    @func
    def affine_pair(
        x: Tensor[(168,), "f32"], scale: Tensor[(168,), "f32"], bias: Tensor[(168,), "f32"]
    ) -> Tensor[(168,), "f32"]:
        return x * scale + bias

    def forward(self, x, pair):
        a, b, scale, bias = pair
        return self.add_pair(x, a, b), self.affine_pair(x, scale, bias)


@runtime_module(Orchestrated)
class OrchestratedTwin:
    @runtime_func
    def add_pair(self, x, a, b):
        return x + a + b

    @runtime_func
    def affine_pair(self, x, scale, bias):
        return x * scale + bias

    def forward(self, x, pair):
        a, b, scale, bias = pair
        return self.add_pair(x, a, b), self.affine_pair(x, scale, bias)


@module(target=CpuTarget())
class Nested:
    child = Weighted


@runtime_module(Nested)
class NestedTwin:
    child = WeightedTwin


class Handwritten(RuntimeModule):
    def __init__(self):
        super().__init__(name="handwritten")


class Mislabelled(RuntimeModule):
    module = "not a Module"

    def __init__(self):
        super().__init__(name="mislabelled")
'''


@pytest.fixture(scope="module")
def twin(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("authored") / "mine.py"
    path.write_text(textwrap.dedent(_TWIN_SOURCE), encoding="utf-8")
    return path


# A statement body, which `analyze` annotates per statement, over 168 CTAs.
_CWIDE_MODEL = '''\
from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 168),))
class Model:

    @func
    def main(x: Tensor[(168,), "f32"]):
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            return tf.reshard(squared, (168 @ cta.block,), "gmem")
'''


@pytest.fixture
def cwide(tmp_path) -> Path:
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_CWIDE_MODEL), encoding="utf-8")
    return path
