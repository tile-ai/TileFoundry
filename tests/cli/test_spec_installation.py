from __future__ import annotations

import os
import site
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_wheel_help_dsl_matches_hir_spec(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from setuptools.build_meta import build_wheel; "
                f"build_wheel({str(wheel_dir)!r})"
            ),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("tilefoundry-*.whl"))

    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = Path(
        subprocess.run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if len(parts) >= 3 and parts[0].endswith(".data") and parts[1] == "data":
                destination = environment.joinpath(*parts[2:])
            elif parts and parts[0].endswith(".data"):
                continue
            else:
                destination = purelib.joinpath(*parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))

    parent_sites = os.pathsep.join(site.getsitepackages())
    result = subprocess.run(
        [str(python), "-m", "tilefoundry.cli", "help", "dsl"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": parent_sites},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert result.stdout == (_ROOT / "docs/spec/hir.md").read_text(encoding="utf-8")
