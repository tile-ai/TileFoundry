"""The spec-reference checker, held to the anchoring each kind of file reads.

A renderer resolves a page's links beside the page; a source file has no anchor
but the repository root. A checker that takes both spellings everywhere passes a
page whose link renders dead, which is how one reached `main`. So both
directions are asserted here: the spelling each kind of file must use, and the
spelling it must refuse.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "spec_refs_lint.py"


_COVERED = re.compile(r"^(docs/spec/.*\.md|src/.*\.py|tests/.*\.py|include/.*\.(h|cuh))$")


_SECTION = "§" + "2"


_ANCHOR = "#2-parameters-and-inputs"


def _lint():
    """The checker, loaded from the script the hook runs."""
    spec = importlib.util.spec_from_file_location("spec_refs_lint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference(path: str) -> str:
    """One whole reference to the fixture's numbered section, assembled not written.

    The hook reads `tests/*.py`, so a reference spelled out here would be a
    reference the checker resolves against this repository rather than against
    the fixture tree, and this file would be reported for a fixture's spelling.
    A section mark left standing in prose is a bare reference to the same hook,
    so the mark is assembled too.
    """
    return f"[runtime {_SECTION}]({path}{_ANCHOR})"


@pytest.fixture
def lint(tmp_path, monkeypatch):
    """The checker over a fixture tree holding one spec page with a section 2."""
    module = _lint()
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    page = tmp_path / "docs" / "spec" / "runtime.md"
    page.parent.mkdir(parents=True)
    page.write_text("# Runtime\n\n## 2. Parameters and inputs\n\nProse.\n", encoding="utf-8")
    return module


def _page(tmp_path, text: str) -> Path:
    target = tmp_path / "docs" / "spec" / "cli.md"
    target.write_text(text, encoding="utf-8")
    return target


def _source(tmp_path, text: str) -> Path:
    target = tmp_path / "src" / "tilefoundry" / "runtime.py"
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_a_page_links_beside_itself(lint, tmp_path) -> None:
    assert lint.findings(_page(tmp_path, _reference("./runtime.md"))) == []


def test_a_page_refuses_the_repository_root_path(lint, tmp_path) -> None:
    """The form that resolves from the root and renders dead from the page."""
    found = lint.findings(_page(tmp_path, _reference("docs/spec/runtime.md")))

    assert len(found) == 1
    assert "repository-root path" in found[0][1]


def test_code_links_from_the_repository_root(lint, tmp_path) -> None:
    """The root path is what a source file has, and stays accepted there."""
    assert lint.findings(_source(tmp_path, f"# {_reference('docs/spec/runtime.md')}\n")) == []


def test_a_page_naming_a_missing_document_is_still_reported(lint, tmp_path) -> None:
    """Refusing the root path did not swallow the check it used to fail."""
    found = lint.findings(_page(tmp_path, _reference("./gone.md")))

    assert len(found) == 1
    assert "does not exist" in found[0][1]


def test_the_exit_status_and_the_report_name_the_line(lint, tmp_path, capsys) -> None:
    """The hook's own contract: non-zero, and enough on stdout to go fix it."""
    target = _page(tmp_path, f"Prose.\n\nA run says so ({_reference('docs/spec/runtime.md')}).\n")

    status = lint.main([str(target)])

    assert status == 1
    assert f"{target}:3:" in capsys.readouterr().out


def test_this_repository_is_clean() -> None:
    """Every path the spec-refs-lint hook reads, mirroring its `files:`.

    So the guard is a fact about the tree and not only about the commit in
    flight. A reference that resolves for the checker and not for the renderer
    is only visible from the whole tree: the page holding it passes on its own,
    and the site build that reports it runs in another repository.
    """
    lint = _lint()
    root = _SCRIPT.parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")

    offenders = {name: lint.findings(root / name) for name in tracked if _COVERED.match(name)}
    assert not {name: found for name, found in offenders.items() if found}
