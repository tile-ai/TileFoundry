"""tilefoundry autodiff: the two generated Modules as one loadable file, and the saved table."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

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


def saved_rows(capsys, fixture: str, wrt: str, *extra: str) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Each row of the saved table by name, its cells by column, and the two totals."""
    assert cli.main(["autodiff", str(FIXTURES / fixture), "--wrt", wrt, "--saved", *extra]) == 0
    header, *rows, held, parameters = capsys.readouterr().out.splitlines()
    columns = header.split()
    assert columns == ["saved", "shape", "bytes", "layout", "buffer", "from"]
    totals = {line.split()[0]: int(line.split()[1]) for line in (held, parameters)}
    assert list(totals) == ["held", "parameters"]
    cells = [dict(zip(columns, re.split(" {2,}", row))) for row in rows]
    return {row["saved"]: row for row in cells}, totals


def test_saved_table_counts_bytes_per_device(capsys) -> None:
    """Two GPUs, f32: h (8, 6) Split(0) holds 4 x 6 x 4 bytes on each device; full is whole."""
    rows, totals = saved_rows(capsys, "sharded.py", "x,w")
    assert set(rows) == {"x", "h", "full"}
    assert [rows["h"][column] for column in ("shape", "bytes", "layout")] == ["(8, 6)", "96", "Split(0)"]
    assert [rows["full"][column] for column in ("shape", "bytes", "layout")] == ["(6, 4)", "96", "-"]
    assert rows["x"]["from"] == "parameter"
    assert totals == {"held": 192, "parameters": 64}


def test_recompute_changes_the_saved_table(capsys) -> None:
    rows, _ = saved_rows(capsys, "sharded.py", "x,w", "--recompute", "Layer.forward:full")
    assert set(rows) == {"x", "h", "w"}
    assert [rows["w"]["bytes"], rows["w"]["layout"]] == ["48", "Split(0)"]


class Buffers(NamedTuple):
    fixture: str
    wrt: str
    totals: tuple[int, int]
    """``held`` and ``parameters``."""
    recompute: tuple[str, ...] = ()
    shared: Mapping[str, str] = MappingProxyType({})
    """The rows whose buffer is another value; every other row is its own buffer."""
    saved: frozenset[str] | None = None
    """The rows, the parameters left out, when the case says which."""


QKV = MappingProxyType({"q": "qkv", "k": "qkv", "v": "qkv"})
BUFFERS = (
    Buffers("fused_views.py", "w", (2048, 512), shared=QKV, saved=frozenset({"qk", "v", "q", "k"})),
    Buffers("fused_views.py", "w", (2048, 512), ("Fused:k",), MappingProxyType({"q": "qkv", "v": "qkv"}),
            frozenset({"qk", "v", "q", "qkv"})),
    Buffers("fused_views.py", "w", (2048, 512), ("Fused:q", "Fused:k", "Fused:v"), saved=frozenset({"qk", "qkv"})),
    Buffers("fused_views.py", "w", (1536, 512), ("Fused:qk",), QKV, frozenset({"v", "q", "k"})),
    Buffers("param_view.py", "x", (512, 3072), shared=MappingProxyType({"head": "w"})),
    Buffers("rebound.py", "x", (96, 48)),
    Buffers("layers_loop.py", "x,wn0,norms,up,down", (2688, 3328)),
)


@pytest.mark.parametrize("case", BUFFERS, ids=lambda case: f"{case.fixture}{list(case.recompute)}")
def test_a_buffer_is_held_once_whole(capsys, case: Buffers) -> None:
    """A saved view holds its whole buffer, once.

    Dropping one slice of qkv frees nothing while another keeps it. A
    parameter's buffer is held anyway; a stack, a callee's item and each binding
    of a name are buffers of their own.
    """
    selected = [word for selector in case.recompute for word in ("--recompute", selector)]
    rows, totals = saved_rows(capsys, case.fixture, case.wrt, *selected)
    assert {name: row["buffer"] for name, row in rows.items()} == {name: case.shared.get(name, name) for name in rows}
    assert (totals["held"], totals["parameters"]) == case.totals
    assert case.saved is None or {name for name, row in rows.items() if row["from"] != "parameter"} == case.saved


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
