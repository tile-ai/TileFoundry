"""Run the fixture corpus through the authored HIR command boundary."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tilefoundry.cli.source import load_namespace
from tilefoundry.ir.core.module import Module

CORPUS = Path(__file__).parent / "flashinfer"
CLI = Path(sys.executable).with_name("tilefoundry")


def _fixtures(pattern: str) -> tuple[Path, ...]:
    return tuple(sorted(CORPUS.glob(pattern)))


def _ok_fixtures() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _fixtures("*.py")
        if path.name != "__init__.py" and not path.name.endswith(".blocked.py")
    )


def _doc_fields(path: Path) -> dict[str, str]:
    document = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert document is not None
    fields: dict[str, str] = {}
    for line in document.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"blocked", "phase", "error", "got", "expected", "why"}:
            fields[key] = value.strip()
    return fields


def _ok_sources() -> tuple[str, ...]:
    sources = []
    for path in _ok_fixtures():
        namespace, _ = load_namespace(str(path))
        modules = sorted(value.name for value in namespace.values() if isinstance(value, Module))
        sources.extend(f"{path}:{name}" for name in modules)
    return tuple(sources)


def _analyze(source: str, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(CLI),
            "analyze",
            source,
            str(report),
            "--compute-cost",
            "--memory",
            "--roofline",
            "--performance",
            "--topology",
            "cta",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


OK_SOURCES = _ok_sources()


@pytest.mark.parametrize(
    "source",
    OK_SOURCES,
    ids=lambda source: source.removeprefix(f"{CORPUS}/"),
)
def test_ok_fixtures_load_and_analyze(source: str, tmp_path: Path) -> None:
    """Every Module in every unblocked fixture must load and produce a report."""
    selector = source.rsplit(":", maxsplit=1)[-1]
    report = tmp_path / f"{selector}.report"
    completed = _analyze(source, report)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert report.is_file()
    assert "# analysis" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", _fixtures("*.blocked.py"), ids=lambda path: path.name)
def test_blocked_fixtures_preserve_their_observed_phase_and_result(
    path: Path, tmp_path: Path
) -> None:
    """Blocked fixtures must fail or report exactly where their docstring says."""
    fields = _doc_fields(path)
    assert fields["phase"] in {"load", "selection/analysis"}
    report = tmp_path / f"{path.stem}.report"
    prefix = path.stem.partition(".")[0]
    before = {name for name in sys.modules if name.partition(".")[0] == prefix}
    try:
        load_namespace(str(path))
        loaded = True
    except Exception:
        loaded = False
    assert {name for name in sys.modules if name.partition(".")[0] == prefix} == before

    completed = _analyze(str(path), report)
    if fields["blocked"] == "refused":
        if fields["phase"] == "load":
            assert not loaded
        else:
            assert fields["phase"] == "selection/analysis"
            assert loaded
        assert completed.returncode != 0
        assert not report.exists()
        assert fields["error"] in completed.stderr
        return

    assert fields["blocked"] == "mis-analyzed"
    assert fields["phase"] == "selection/analysis"
    assert loaded
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert report.is_file()
    assert fields["got"] in report.read_text(encoding="utf-8")
