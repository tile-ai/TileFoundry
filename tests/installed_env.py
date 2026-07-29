"""TileFoundry installed from a built wheel, for behaviour only an installation has.

The nested environment inherits its base's packages, so an editable install in
that base can shadow the wheel. `assert_installed` refuses that, and refuses it
rather than skipping: a fallback to this checkout reports the same green as the
installed branch actually running.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
        """Refuse an environment that resolves back to this checkout.

        Both halves are checked because neither implies the other: the package can
        be the wheel's while the documents still come from `docs/spec/`.
        """
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
    """Build a wheel of this checkout and install it into a nested environment.

    `--no-deps` because the dependencies come from the base environment: what is
    under test is where TileFoundry itself is imported from.
    """
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


__all__ = ["InstalledEnv", "build"]
