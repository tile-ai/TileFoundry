"""tilefoundry autodiff: the two generated Modules as one loadable file, and the saved table."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry import cli, inspection
from tilefoundry.passes.transforms import differentiate

FIXTURES = Path(__file__).parents[1] / "fixtures" / "autodiff"


def test_written_file_is_the_printed_pair(tmp_path, capsys) -> None:
    """D44: PATH holds the two Modules as one script; each fixture's case reloads it."""
    written = tmp_path / "trained.py"
    command = ["autodiff", str(FIXTURES / "child_call.py:ChildCall"), "--wrt", "x,w1,w2"]
    assert cli.main([*command, "--recompute", "ChildCall.rest.1:h", str(written)]) == 0
    assert capsys.readouterr().out == ""
    module = cli.load_authored_ir(str(FIXTURES / "child_call.py:ChildCall"))
    generated = differentiate.differentiate(module, wrt=("x", "w1", "w2"), recompute=("ChildCall.rest.1:h",))
    assert written.read_text() == inspection.as_script(generated)


def test_without_path_prints_both_modules(capsys) -> None:
    assert cli.main(["autodiff", str(FIXTURES / "elementwise.py"), "--wrt", "x"]) == 0
    printed = capsys.readouterr().out
    assert "class ElementwiseForward:" in printed
    assert "class ElementwiseBackward:" in printed


def saved_rows(capsys, *extra: str) -> dict[str, list[str]]:
    assert cli.main(["autodiff", str(FIXTURES / "sharded.py"), "--wrt", "x,w", "--saved", *extra]) == 0
    header, *rows = capsys.readouterr().out.splitlines()
    assert header.split() == ["saved", "shape", "bytes", "layout", "from"]
    return {row.split()[0]: row.split()[1:] for row in rows}


def test_saved_table_counts_bytes_per_device(capsys) -> None:
    """Two GPUs, f32: h (8, 6) Split(0) holds 4 x 6 x 4 bytes on each device; full is whole."""
    rows = saved_rows(capsys)
    assert set(rows) == {"x", "h", "full"}
    assert rows["h"][:4] == ["(8,", "6)", "96", "Split(0)"]
    assert rows["full"][:4] == ["(6,", "4)", "96", "-"]
    assert rows["x"][4:] == ["parameter"]


def test_recompute_changes_the_saved_table(capsys) -> None:
    rows = saved_rows(capsys, "--recompute", "Layer.forward:full")
    assert set(rows) == {"x", "h", "w"}
    assert rows["w"][2:4] == ["48", "Split(0)"]


def test_missing_wrt_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["autodiff", str(FIXTURES / "elementwise.py")])
    assert stopped.value.code == 2
    assert "--wrt" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("wrt", "recompute"),
    (("q", ()), ("x", ("--recompute", "Layer.forward:nope"))),
    ids=("unknown wrt", "unknown recompute"),
)
def test_refusals_carry_the_differentiate_message(capsys, wrt, recompute) -> None:
    module = cli.load_authored_ir(str(FIXTURES / "sharded.py"))
    with pytest.raises(ValueError) as refused:
        differentiate.differentiate(module, wrt=(wrt,), recompute=tuple(recompute[1:]))
    assert cli.main(["autodiff", str(FIXTURES / "sharded.py"), "--wrt", wrt, *recompute]) == 1
    assert capsys.readouterr().err == f"tilefoundry: error: {refused.value}\n"
