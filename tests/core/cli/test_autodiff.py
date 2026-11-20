"""tilefoundry autodiff: the two generated Modules as one loadable file, and the saved table."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from tilefoundry import cli, inspection
from tilefoundry.passes.transforms import differentiate

FIXTURES = Path(__file__).parents[1] / "fixtures" / "autodiff"


class RoundTrip(NamedTuple):
    fixture: str
    name: str
    wrt: str
    recompute: tuple[str, ...] = ()
    same_text: bool = True
    """False: both Modules only load; a dim operand that is not a dim prints inline, and reloads bound."""


ROUND_TRIP = (
    RoundTrip("elementwise.py", "Elementwise", "x,w,b"),
    RoundTrip("sharded.py", "Layer", "x,w"),
    RoundTrip("child_call.py:ChildCall", "ChildCall", "x,w1,w2"),
    RoundTrip("child_call.py:ChildCall", "ChildCall", "x,w1,w2", ("ChildCall.rest.1:h",)),
    RoundTrip("inactive_call.py:InactiveCall", "InactiveCall", "x"),
    RoundTrip("loop_range.py", "LoopRange", "x,w,h0"),
    RoundTrip("two_loops.py", "TwoLoops", "x,w"),
    RoundTrip("loop_call.py", "LoopCall", "keys,decay"),
    RoundTrip("loop_nested.py", "LoopNested", "x,w"),
    RoundTrip("mesh_data_parallel.py", "MeshDataParallel", "x,w"),
    RoundTrip("mesh_two_regions.py", "MeshTwoRegions", "x,w"),
    RoundTrip("megatron_block.py", "TensorParallel", "x,wn1,wq,wk,wv,wo,wn2,wg,wu,wd"),
    RoundTrip("layers_loop.py", "Layers", "x,wn0,norms,up,down"),
)


@pytest.mark.parametrize("row", ROUND_TRIP, ids=lambda row: str(tuple(row[:4])))
def test_written_file_loads_both_modules_and_reprints_the_same(tmp_path, capsys, row: RoundTrip) -> None:
    """D44: one file, one header, each class once; both Modules load and print back to the file."""
    fixture, name, wrt, recompute, same_text = row
    written = tmp_path / "trained.py"
    selected = [word for selector in recompute for word in ("--recompute", selector)]
    assert cli.main(["autodiff", str(FIXTURES / fixture), "--wrt", wrt, *selected, str(written)]) == 0
    assert capsys.readouterr().out == ""
    text = written.read_text()
    assert text.count("from __future__ import annotations") == 1
    classes = [line for line in text.splitlines() if line.startswith("class ")]
    assert len(classes) == len(set(classes))
    forward = cli.load_authored_ir(f"{written}:{name}Forward")
    backward = cli.load_authored_ir(f"{written}:{name}Backward")
    assert not same_text or inspection.as_script((forward, backward)) == text


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
