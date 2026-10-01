"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.isl_interop import normalize_dim
from tilefoundry.ir.types.dim import DimMul, DimSub, simplify_dim
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.parser import ParseError


@pytest.mark.parametrize(
    ("index", "message"),
    [
        ("t:t + 128", r"tile loop variable is already a window.*x\[:, t, :\].*base = t \+ 0"),
        ("t / 2", r"tile windows only support ± c and \* c"),
        ("8 / 2", r"index arithmetic uses //"),
        ("N / 2", r"index arithmetic uses //"),
        ("t * 2", None),
        ("2 * t", None),
        ("t * 2 + 1", None),
        ("t * 2 - 1", None),
    ],
)
def test_a_tile_window_bound_names_the_authored_fix(index, message) -> None:
    source = f"""from tilefoundry import func
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget
from tilefoundry.ir.types.dim import DimVar

N = DimVar("N", 1, 257)

@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def stage(x: Tensor[(1, 256, 64), "f32"], out: Tensor[(1, 128, 64), "f32"]):
    with Mesh(("cta",), layout=(1,), names=("unit",)) as mesh:
        acc = out
        for t in tile(128, 128):
            window = tf.reshard(x[:, {index}, :], (1, 128 @ mesh.unit, 64), "smem")
            acc = tf.insert_slice(acc, window, (0, t, 0))
        return acc
"""
    if message is not None:
        with pytest.raises(ParseError, match=message):
            import_dsl(source, "stage")
        return
    stage = import_dsl(source, "stage")
    loop = next(
        expr for expr in collect_exprs(stage.entry_function().body) if isinstance(expr, LoopRegion)
    )
    sliced = next(
        expr
        for expr in collect_exprs(stage.entry_function().body)
        if isinstance(expr, Call) and isinstance(expr.target, Slice)
    )
    assert sliced.target.sizes[1] == 128
    assert sliced.target.strides[1] == 2
    raw_start = sliced.args[1].elements[1]
    scaled = [
        expr for expr in collect_exprs(raw_start)
        if isinstance(expr, Call) and isinstance(expr.target, DimMul)
        and loop.induction_var in expr.args
    ]
    assert len(scaled) == 1
    start = scaled[0]
    offset = static_dim_value(normalize_dim(simplify_dim(DimSub, (raw_start, start))))
    assert offset == (1 if "+ 1" in index else -1 if "- 1" in index else 0)
    assert isinstance(start.target, DimMul)
    assert loop.induction_var in start.args
    assert any(static_dim_value(arg) == 2 for arg in start.args)
    printed = as_script(stage)
    restored = import_dsl(printed, "stage")
    assert as_script(restored) == printed
    restored_loop = next(
        expr
        for expr in collect_exprs(restored.entry_function().body)
        if isinstance(expr, LoopRegion)
    )
    assert len(restored_loop.params) == len(loop.params)
    assert len(restored_loop.args) == len(loop.args)
