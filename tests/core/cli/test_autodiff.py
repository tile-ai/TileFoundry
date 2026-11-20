"""tilefoundry autodiff: the two generated Modules as one loadable file, and the saved table."""

from __future__ import annotations

import re
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


def saved_rows(capsys, fixture: str, wrt: str, *extra: str) -> tuple[dict[str, list[str]], dict[str, int]]:
    """The cells of each row of the saved table by name, and its two totals."""
    assert cli.main(["autodiff", str(FIXTURES / fixture), "--wrt", wrt, "--saved", *extra]) == 0
    header, *rows, held, parameters = capsys.readouterr().out.splitlines()
    assert header.split() == ["saved", "shape", "bytes", "layout", "buffer", "from"]
    totals = {line.split()[0]: int(line.split()[1]) for line in (held, parameters)}
    assert list(totals) == ["held", "parameters"]
    cells = [re.split(" {2,}", row) for row in rows]
    return {name: rest for name, *rest in cells}, totals


def test_saved_table_counts_bytes_per_device(capsys) -> None:
    """Two GPUs, f32: h (8, 6) Split(0) holds 4 x 6 x 4 bytes on each device; full is whole."""
    rows, totals = saved_rows(capsys, "sharded.py", "x,w")
    assert set(rows) == {"x", "h", "full"}
    assert rows["h"][:4] == ["(8, 6)", "96", "Split(0)", "h"]
    assert rows["full"][:4] == ["(6, 4)", "96", "-", "full"]
    assert rows["x"][3:] == ["x", "parameter"]
    assert totals == {"held": 192, "parameters": 64}


def test_recompute_changes_the_saved_table(capsys) -> None:
    rows, _ = saved_rows(capsys, "sharded.py", "x,w", "--recompute", "Layer.forward:full")
    assert set(rows) == {"x", "h", "w"}
    assert rows["w"][1:3] == ["48", "Split(0)"]


@pytest.mark.parametrize(
    ("recompute", "saved", "held"),
    (
        ((), {"qk", "v", "q", "k"}, 2048),
        (("Fused:k",), {"qk", "v", "q", "qkv"}, 2048),
        (("Fused:q", "Fused:k", "Fused:v"), {"qk", "qkv"}, 2048),
        (("Fused:qk",), {"v", "q", "k"}, 1536),
    ),
    ids=("none", "k", "q,k,v", "qk"),
)
def test_a_saved_view_holds_its_whole_buffer_once(capsys, recompute, saved, held) -> None:
    """Dropping a saved slice of qkv frees nothing while another slice keeps qkv."""
    selected = [word for selector in recompute for word in ("--recompute", selector)]
    rows, totals = saved_rows(capsys, "fused_views.py", "w", *selected)
    assert set(rows) - {"x"} == saved
    assert {name: row[3] for name, row in rows.items()} == {
        name: "qkv" if name in ("q", "k", "v") else name for name in rows
    }
    assert totals == {"held": held, "parameters": 512}


def test_a_saved_view_of_a_parameter_holds_nothing_new(capsys) -> None:
    rows, totals = saved_rows(capsys, "param_view.py", "x")
    assert rows["head"][3] == "w"
    assert totals == {"held": 512, "parameters": 3072}


def test_two_values_of_one_name_are_two_buffers(capsys) -> None:
    _, totals = saved_rows(capsys, "rebound.py", "x")
    assert totals == {"held": 96, "parameters": 48}


def test_a_stack_and_a_callee_item_are_their_own_buffer(capsys) -> None:
    rows, _ = saved_rows(capsys, "layers_loop.py", "x,wn0,norms,up,down")
    assert all(row[3] == name for name, row in rows.items())


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
