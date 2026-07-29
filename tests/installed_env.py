"""TileFoundry installed from a built wheel, for behaviour only an installation has.

`assert_installed` fails an environment that resolves back to this checkout rather
than skipping it. Why an installed environment is tested at all is in
[cli.md](../docs/spec/cli.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_FACTS = """
import json, sys
import tilefoundry
from tilefoundry.cli.spec import spec_path
print(json.dumps({
    "package": tilefoundry.__file__,
    "spec": str(spec_path("dsl")),
    "prefix": sys.prefix,
}))
"""


@dataclass(frozen=True)
class InstalledEnv:
    """A virtual environment with the wheel installed in it."""

    root: Path

    @property
    def python(self) -> Path:
        return self.root / "bin" / "python"

    @property
    def tilefoundry(self) -> Path:
        return self.root / "bin" / "tilefoundry"

    def run(self, *arguments: str) -> str:
        """Run the installed console script and return its stdout."""
        return subprocess.run(
            [str(self.tilefoundry), *arguments],
            check=True, capture_output=True, text=True,
        ).stdout

    def facts(self) -> dict[str, str]:
        """Where this environment's import of TileFoundry actually comes from."""
        completed = subprocess.run(
            [str(self.python), "-c", _FACTS],
            check=True, capture_output=True, text=True,
        )
        return json.loads(completed.stdout)

    def assert_installed(self) -> dict[str, str]:
        """Fail unless both the package and its documents come from this root."""
        facts = self.facts()
        root = str(self.root)
        assert facts["prefix"] == root, facts
        assert facts["package"].startswith(root), (
            f"the nested environment imported TileFoundry from {facts['package']}, "
            f"not from {root}; an inherited editable install is shadowing the wheel, "
            f"so this would exercise the source tree instead of the installed one"
        )
        assert not facts["spec"].startswith(str(_REPO_ROOT / "docs")), facts["spec"]
        assert facts["spec"].startswith(root), (
            f"the nested environment read its specifications from {facts['spec']}, "
            f"which is not under {root}; the installed-data branch was not taken"
        )
        return facts


def build(destination: Path) -> InstalledEnv:
    """Build a wheel of this checkout and install it, without its dependencies, into
    a nested environment under `destination`."""
    wheels = destination / "wheel"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(_REPO_ROOT),
         "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels)],
        check=True, capture_output=True, text=True,
    )
    built = sorted(wheels.glob("tilefoundry-*.whl"))
    assert len(built) == 1, [path.name for path in built]

    environment = InstalledEnv(destination / "venv")
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment.root)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [str(environment.python), "-m", "pip", "install", "--no-deps", str(built[0])],
        check=True, capture_output=True, text=True,
    )
    return environment


def shared_build(
    directory: Path,
    builder: Callable[[Path], InstalledEnv] = build,
    timeout: float = 600.0,
) -> InstalledEnv:
    """Build into `directory` once, in whichever process creates it first.

    Creating the directory is the lock. Its winner writes `ready` when the
    environment is usable, or `failed` carrying the diagnostic; the other processes
    wait for `ready` and raise what `failed` says as soon as it appears, rather than
    waiting out `timeout`.
    """
    ready, failed = directory / "ready", directory / "failed"
    try:
        directory.mkdir(parents=True)
    except FileExistsError:
        deadline = time.monotonic() + timeout
        while not ready.is_file():
            if failed.is_file():
                raise RuntimeError(failed.read_text(encoding="utf-8"))
            if time.monotonic() > deadline:
                raise RuntimeError(f"the process building {directory} never finished")
            time.sleep(0.5)
    else:
        try:
            builder(directory).assert_installed()
        except BaseException as error:
            output = getattr(error, "stderr", "") or ""
            failed.write_text(
                f"building an installed TileFoundry in {directory} failed: "
                f"{error!r}\n{output}",
                encoding="utf-8",
            )
            raise
        ready.write_text("ready", encoding="utf-8")

    environment = InstalledEnv(directory / "venv")
    environment.assert_installed()
    return environment


__all__ = ["InstalledEnv", "build", "shared_build"]
