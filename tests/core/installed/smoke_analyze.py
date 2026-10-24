"""`analyze` against a file of one's own."""

from __future__ import annotations


def test_usage_errors_print_usage_before_error(tf) -> None:
    done = tf("analyze")
    assert done.returncode == 2

    assert done.stdout == ""
    assert done.stderr.startswith("usage: tilefoundry analyze")
    assert (
        "tilefoundry analyze: error: the following arguments are required: SOURCE"
        in done.stderr
    )
    assert "SOURCE" in done.stderr
    assert "model.py[:Module[.child_module...][.function]]" not in done.stderr


def test_analyze_names_the_source_directory_when_a_sibling_is_missing(tf, tmp_path) -> None:
    source = tmp_path / "runtime_model.py"
    source.write_text("from model import Model\n", encoding="utf-8")

    done = tf("analyze", str(source), str(tmp_path / "missing.py"), "--memory")
    assert done.returncode == 1

    assert str(tmp_path) in done.stderr
    assert "a sibling module must sit in that directory" in done.stderr


def test_the_cli_reports_a_duplicate_dimension_and_analyses_nothing(tf, tmp_path) -> None:
    source = tmp_path / "empty.py"
    source.write_text("", encoding="utf-8")

    done = tf(
        "analyze",
        str(source),
        str(tmp_path / "duplicate.py"),
        "--dim=ctx_len=8",
        "--dim=ctx_len=512",
    )
    assert done.returncode == 1

    assert done.stdout == ""
    assert "ctx_len was given twice" in done.stderr
