"""tilefoundry autodiff: the two generated Modules as one loadable file, and the saved table."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry import cli, inspection
from tilefoundry.passes.transforms import differentiate

FIXTURES = Path(__file__).parents[1] / "fixtures" / "autodiff"

ROUND_TRIP = (
    ("elementwise.py", "Elementwise", "x,w,b"),
    ("sharded.py", "Layer", "x,w"),
    ("child_call.py:ChildCall", "ChildCall", "x,w1,w2"),
    ("inactive_call.py:InactiveCall", "InactiveCall", "x"),
    ("loop_range.py", "LoopRange", "x,w,h0"),
    ("loop_call.py", "LoopCall", "keys,decay"),
    ("loop_nested.py", "LoopNested", "x,w"),
    ("mesh_data_parallel.py", "MeshDataParallel", "x,w"),
    ("mesh_two_regions.py", "MeshTwoRegions", "x,w"),
    ("fsdp_model.py", "FsdpModel", "we,q1,k1,v1,o1,u1,d1,q2,k2,v2,o2,u2,d2,wh"),
    ("megatron_block.py", "TensorParallel", "x,wn1,wq,wk,wv,wo,wn2,wg,wu,wd"),
    ("layers_loop.py", "Layers", "x,wn0,norms,up,down"),
)


@pytest.mark.parametrize(("fixture", "name", "wrt"), ROUND_TRIP, ids=lambda value: str(value))
def test_written_file_loads_both_modules_and_reprints_the_same(tmp_path, capsys, fixture, name, wrt) -> None:
    """D44: one file, one header, each class once; both Modules load and print back to the file."""
    written = tmp_path / "trained.py"
    assert cli.main(["autodiff", str(FIXTURES / fixture), "--wrt", wrt, str(written)]) == 0
    assert capsys.readouterr().out == ""
    text = written.read_text()
    assert text.count("from __future__ import annotations") == 1
    classes = [line for line in text.splitlines() if line.startswith("class ")]
    assert len(classes) == len(set(classes))
    forward = cli.load_authored_ir(f"{written}:{name}Forward")
    backward = cli.load_authored_ir(f"{written}:{name}Backward")
    assert inspection.as_script((forward, backward)) == text


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
    assert rows["full"][:4] == ["(6,", "4)", "96", "Broadcast"]
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
