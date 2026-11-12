"""The viewer builds its DOT graph for the placed rmsnorm programs, without a server."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry import inspection
from tilefoundry.inspection import viewer
from tilefoundry.ir import core, expr, types

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"


class ViewerCase(Case):
    """A program whose entry function the viewer draws."""

    def guard(self, state, problems):
        graph = viewer.Builder(state, collapsed=set()).build()

        assert graph.source.strip()


@case
class Rmsnorm(ViewerCase):
    path: Path = PLACED / "rmsnorm.py"
    name: str | None = "RmsnormModule"


@case
class RmsnormSeq2(ViewerCase):
    path: Path = PLACED / "rmsnorm_seq2.py"
    name: str | None = "RmsnormSeq2Module"


@case
class RmsnormQuantSeq2(ViewerCase):
    path: Path = PLACED / "rmsnorm_quant_seq2.py"
    name: str | None = "RmsnormQuantSeq2Module"


CASES = special(ViewerCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_viewer_case(case: ViewerCase) -> None:
    check_case(case)


def test_instance_calls_name_index_and_argument_ports():
    """An instance call node is titled by its path; its index and argument ports match its detail.

    A run-time index takes its own port, not the first parameter's name, and
    the weight an instance entry reads from its instance is not shown as an
    argument.
    """
    tensor = types.TensorType((4,), expr.DType.f32, None, expr.StorageKind.GMEM)
    x, w = expr.Var("x", type=tensor), expr.Var("w", type=tensor, is_const=True)
    leaf = core.Module("Leaf", (core.Function.from_parts(name="run", params=(x, w), body=x, return_type=tensor),), "run")
    a, i = expr.Var("a", type=tensor), expr.Var("i", type=types.TensorType.from_scalar(expr.DType.i32))
    child = expr.Call(core.FunctionRef(("child",), None), (a,), type=tensor)
    layer = expr.Call(core.FunctionRef(("layers", None), None), (i, child), type=tensor)
    main = core.Function.from_parts(name="main", params=(a, i), body=layer, return_type=tensor)
    root = core.Module("Root", (main,), "main", children={"child": leaf, "layers": (leaf, leaf)})

    builder = viewer.Builder(root, collapsed=set())
    source = builder.build().source
    vids = {ref.hir_expr: vid for vid, ref in builder.index.entries.items() if ref.kind != "Param"}
    names = {
        call: [param["name"] for param in viewer.format_detail(vids[call], builder.index.get(vids[call]))["params"]]
        for call in (child, layer)
    }

    assert [builder.index.get(vids[call]).kind for call in (child, layer)] == ["child", "layers.*"]
    assert names == {child: ["x"], layer: ["index", "x"]}
    assert f"{vids[main]}:pout1 -> {vids[layer]}:in0" in source
    assert f"{vids[child]} -> {vids[layer]}:in1" in source
    assert f"{vids[main]}:pout0 -> {vids[child]}:in0" in source
    dot = inspection.hir_function_to_dot(main)
    assert "child" in dot and "layers.*" in dot
