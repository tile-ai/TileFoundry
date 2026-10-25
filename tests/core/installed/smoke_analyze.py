"""`analyze` against a file of one's own."""

from __future__ import annotations

from pathlib import Path

from tests.core.installed._analyze import check_usage_error

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_analyze_usage_error_prints_usage(tf) -> None:
    done = tf("analyze")
    check_usage_error(done)
    assert "tilefoundry analyze: error: the following arguments are required: SOURCE" in done.stderr
    assert "SOURCE" in done.stderr
    assert "model.py[:Module[.child_module...][.function]]" not in done.stderr


def test_analyze_names_missing_sibling_directory(tf, tmp_path) -> None:
    source = FIXTURES / "cli/missing_sibling/runtime_model.py"

    done = tf("analyze", str(source), str(tmp_path / "missing.py"), "--memory")
    assert done.returncode == 1

    assert str(source.parent) in done.stderr
    assert "a sibling module must sit in that directory" in done.stderr


def test_analyze_refuses_duplicate_dim(tf, tmp_path) -> None:
    done = tf(
        "analyze",
        str(FIXTURES / "cli/empty.py"),
        str(tmp_path / "duplicate.py"),
        "--dim=ctx_len=8",
        "--dim=ctx_len=512",
    )
    assert done.returncode == 1

    assert done.stdout == ""
    assert "ctx_len was given twice" in done.stderr
