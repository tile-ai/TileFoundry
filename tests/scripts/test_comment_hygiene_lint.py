"""Behavioral checks for source documentation placement and size."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "comment_hygiene_lint.py"


def _lint():
    """Load the exact checker used by the hook."""
    spec = importlib.util.spec_from_file_location("comment_hygiene_lint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lint():
    return _lint()


def _find(lint, tmp_path, text, suffix=".py"):
    target = tmp_path / f"sample{suffix}"
    target.write_text(text, encoding="utf-8")
    return lint.findings(target)


@pytest.mark.parametrize(
    "text",
    [
        "# module prose\n",
        "# declaration prose\ndef f():\n    pass\n",
        "class C:\n    # class prose\n    value = 1\n",
        "def f():\n    # function prose\n    return 1\n",
    ],
)
def test_python_prose_comments_are_rejected(lint, tmp_path, text) -> None:
    assert _find(lint, tmp_path, text)


@pytest.mark.parametrize(
    "text",
    [
        "#!/usr/bin/env python3\nvalue = 1\n",
        "value = call()  # noqa: F841\n",
        "value = call()  # type: ignore[arg-type]\n",
        "# ruff: noqa\nvalue = 1\n",
    ],
)
def test_tool_directives_and_the_first_line_shebang_are_allowed(lint, tmp_path, text) -> None:
    assert _find(lint, tmp_path, text) == []


def test_docstring_prose_budget_is_eight_lines(lint, tmp_path) -> None:
    eight = "\n".join(f"line {number}" for number in range(8))
    assert _find(lint, tmp_path, f'"""{eight}\n"""\n') == []
    assert _find(lint, tmp_path, f'"""{eight}\nline 8\n"""\n')


def test_google_sections_do_not_spend_prose_lines(lint, tmp_path) -> None:
    arguments = "\n".join(f"    p{number}: Value {number}." for number in range(13))
    text = f'"""Summary.\n\nArgs:\n{arguments}\n"""\n'
    assert _find(lint, tmp_path, text) == []


def test_docstring_and_directive_width_is_limited(lint, tmp_path) -> None:
    assert _find(lint, tmp_path, f'"""{"x" * 95}"""\n')
    assert _find(lint, tmp_path, f"value = 1  # noqa: {'x' * 90}\n")


def test_hygiene_only_excuses_narration(lint, tmp_path) -> None:
    escaped = '"""The docs/plans/x.md name is required here. hygiene: generated path"""\n'
    assert _find(lint, tmp_path, escaped) == []
    assert _find(lint, tmp_path, "# hygiene: generated path\n")
    assert _find(lint, tmp_path, f'"""{"x" * 101} hygiene:"""\n')


def test_narration_is_checked_in_comments_and_docstrings(lint, tmp_path) -> None:
    found = _find(lint, tmp_path, '"""Kept for AC-2-1."""\n# milestone M2\n')
    assert len(found) == 3


def test_transition_exemption_only_skips_placement_and_size(lint, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    monkeypatch.setattr(lint, "EXEMPT_PREFIXES", ("src/legacy.py",))
    target = tmp_path / "src" / "legacy.py"
    target.parent.mkdir()
    target.write_text("# milestone M2\n", encoding="utf-8")
    assert lint.findings(target) == [(1, "documentation carries a milestone reference")]


@pytest.mark.parametrize("prefix", ["tests/models", "examples"])
def test_product_source_directories_are_exempt(lint, monkeypatch, tmp_path, prefix) -> None:
    monkeypatch.setattr(lint, "ROOT", tmp_path)
    target = tmp_path / prefix / "legacy.py"
    target.parent.mkdir(parents=True)
    target.write_text("# teaching prose\n", encoding="utf-8")
    assert lint.findings(target) == []


@pytest.mark.parametrize(
    ("text", "allowed"),
    [
        ("/** Contract. */\nint f();\n", True),
        ("/// Contract.\nint f();\n", True),
        ("int value; ///< Contract.\n", True),
        ("// prose\nint f();\n", False),
        ("/* prose */\nint f();\n", False),
        ("namespace n {\n/// Contract.\nint f();\n}\n", True),
    ],
)
def test_c_family_comment_forms(lint, tmp_path, text, allowed) -> None:
    assert bool(_find(lint, tmp_path, text, ".h")) is not allowed


def test_doxygen_prose_budget_is_eight_lines(lint, tmp_path) -> None:
    lines = "\n".join(f" * line {number}" for number in range(9))
    assert _find(lint, tmp_path, f"/**\n{lines}\n */\nint f();\n", ".hpp")
    line_docs = "\n".join(f"/// line {number}" for number in range(9))
    assert _find(lint, tmp_path, f"{line_docs}\nint f();\n", ".hpp")


@pytest.mark.parametrize(
    "literal",
    [
        '"https://example.test/path"',
        '"not /* a comment */"',
        "'/'",
        'R"tag(// not a comment\n/* still a string */)tag"',
    ],
)
def test_c_family_comment_markers_inside_literals_are_ignored(lint, tmp_path, literal) -> None:
    assert _find(lint, tmp_path, f"auto value = {literal};\n", ".hpp") == []


def test_inline_doxygen_block_is_not_a_declaration_comment(lint, tmp_path) -> None:
    assert _find(lint, tmp_path, "int value; /** misplaced */\n", ".hpp")


def test_exit_status_names_the_file_and_line(lint, tmp_path, capsys) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n# prose\n", encoding="utf-8")
    assert lint.main([str(target)]) == 1
    assert f"{target}:2:" in capsys.readouterr().out
