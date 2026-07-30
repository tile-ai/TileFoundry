"""Install a built wheel into a new environment and read the shipped files back.

    python tests/integration/installed_wheel.py --wheel dist --venv /tmp/v

The commands run from a directory that is not a checkout, because `cli/data.py`
prefers a checkout whenever it is standing in one. The environment inherits the
image's site-packages, so the prefix check is what makes the smoke below mean
anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Asked inside the installation, whose prefix is the one the commands resolve
# against. Every kind that ships is looked up, so an undeclared data-files
# destination is named here rather than missing later.
_PROBE = """\
import json

import tilefoundry
from tilefoundry.cli import data

print(json.dumps({
    "the tilefoundry package": tilefoundry.__file__,
    "shipped spec": str(data.directory("spec")),
    "shipped models": str(data.directory("models")),
    "shipped tutorial": str(data.directory("tutorial")),
}))
"""

# One command each, and one thing its output must contain. `tutorial migrate`
# splices its code out of the packaged model source, so a parameter only that
# source declares proves the shipped fixture was what it read.
_SMOKES = (
    (("spec", "dsl"), "silu"),
    (("models",), "qwen3_1_7b"),
    (("models", "qwen3_1_7b", "--source"), "class Qwen3_1_7B"),
    (("tutorial", "migrate"), "w_router: ConstTensor"),
    (("check", "--help"), "--dim"),
)

# A file the installation never shipped: one authored function and its twin, which
# is the least that reaches `check` at all.
_MINE = '''\
from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CudaTarget


@module(entry="main", target=CudaTarget())
class Mine:
    topologies = (Topology("cta", 168),)

    @func
    def main(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(Topology("cta", 168), (168,), ("block",)) as cta:
            local = tf.reshard(x, (168 @ cta.block,), "rmem")
            return tf.reshard(tf.square(local), (168 @ cta.block,), "gmem")


@runtime_module(Mine)
class MineTwin:
    @runtime_func
    def main(self, x):
        return x * x
'''


def _wheel(where: Path) -> Path:
    """The one wheel to install, from a file or the directory holding it."""
    if where.is_file():
        return where
    found = sorted(where.glob("tilefoundry-*.whl")) if where.is_dir() else []
    if len(found) != 1:
        raise SystemExit(
            f"expected exactly one tilefoundry-*.whl in {where}, found "
            f"{len(found)}: {', '.join(p.name for p in found) or 'nothing'}"
        )
    return found[0]


def _run(argv: list[str], cwd: Path) -> str:
    """*argv* to completion, its stdout, and its stderr if it failed."""
    done = subprocess.run(
        argv, cwd=str(cwd), capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise SystemExit(
            f"{' '.join(argv)}\n  in {cwd}\n  exited {done.returncode}\n"
            f"{done.stderr.strip() or done.stdout.strip()}"
        )
    return done.stdout


def install(wheel: Path, venv: Path) -> Path:
    """Build *venv* around *wheel* alone, and return the command it installs.

    *venv* must not exist: creating over a directory keeps what is installed in it,
    and this refuses rather than deleting a path it was handed.
    """
    if venv.exists():
        raise SystemExit(f"--venv {venv} exists; pass a path to be created")
    venv.parent.mkdir(parents=True, exist_ok=True)
    here = Path.cwd()
    _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], here)
    _run([str(venv / "bin" / "pip"), "install", "--no-deps", str(wheel)], here)
    return venv / "bin" / "tilefoundry"


def resolves_under(venv: Path, outside: Path) -> dict[str, str]:
    """Every shipped kind, and where the installation reads it from.

    Raises unless all of them are inside *venv*, naming the ones that were not.
    """
    reported = json.loads(_run([str(venv / "bin" / "python"), "-c", _PROBE], outside))
    prefix = venv.resolve()
    astray = {
        what: where
        for what, where in reported.items()
        if not Path(where).resolve().is_relative_to(prefix)
    }
    if astray:
        raise SystemExit(
            f"resolved outside {venv}:\n"
            + "\n".join(f"  {what}: {where}" for what, where in astray.items())
        )
    return reported


def smoke(command: Path, outside: Path) -> None:
    """Each shipped kind, read through the command that ships to read it."""
    for tail, wanted in _SMOKES:
        printed = _run([str(command), *tail], outside)
        if wanted not in printed:
            raise SystemExit(
                f"`tilefoundry {' '.join(tail)}` printed {len(printed)} characters "
                f"without {wanted!r}"
            )
        print(f"  tilefoundry {' '.join(tail)}: {wanted!r}")


def own_file(command: Path, work: Path) -> None:
    """`check` compares a Module in a file the installation never shipped."""
    source = work / "mine.py"
    source.write_text(_MINE, encoding="utf-8")
    printed = _run(
        [
            str(command), "check", f"{source}:MineTwin.main", "--inputs", "random",
            "--out", "output", "--fn", "allclose", "--atol", "1e-6", "--rtol", "1e-6",
        ],
        work,
    )
    if "reference: evaluator on Mine.main" not in printed or "PASS" not in printed:
        raise SystemExit(f"`check` on a file of one's own did not pass:\n{printed}")
    print(f"  tilefoundry check {source.name}:MineTwin.main: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wheel", required=True, type=Path, help="the wheel, or the directory with it"
    )
    parser.add_argument(
        "--venv", required=True, type=Path, help="environment to create and then read"
    )
    parser.add_argument(
        "--outside",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="a directory that is not a checkout, to run the commands from",
    )
    args = parser.parse_args(argv)

    if not args.outside.is_dir():
        raise SystemExit(f"--outside {args.outside} is not a directory")

    wheel = _wheel(args.wheel)
    print(f"installing {wheel.name} into {args.venv}")
    command = install(wheel, args.venv)

    print(f"reading from {args.outside}:")
    for what, where in resolves_under(args.venv, args.outside).items():
        print(f"  {what}: {where}")
    smoke(command, args.outside)

    # Beside the environment: unique per run, and outside any checkout.
    work = args.venv.parent / f"{args.venv.name}-work"
    work.mkdir(parents=True, exist_ok=True)
    own_file(command, work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
