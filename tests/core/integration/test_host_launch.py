"""Invalid symbolic launch extents at the host boundary."""

from __future__ import annotations

import pytest

from tilefoundry import verifier
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.target import cpu


def _launch_entry_with_grid_x(extent):
    """A CPU host entry whose single launch uses *extent* as ``grid_x``.

    A CPU host entry whose single launch uses *extent* as ``grid_x`` — for
    exercising the grid/block extent verifier on constructed IR.
    """
    t = TensorType(shape=(8,), dtype=DType.f32, layout=types.Layout((8,)), storage="gmem")
    a = expr.Var(type=t, name="a")
    i64 = TensorType.from_scalar(DType.i64, storage=StorageKind.RMEM)
    one = expr.Constant(type=i64, value=1)
    ref = core.FunctionRef((), "dev")
    args = (ref, extent, one, one, one, one, one, a)
    body = core.Sequential(body=(core.Evaluate(callable=tir.launch.Launch(), args=args),))
    return core.PrimFunction(name="host", params=(a,), body=body, target=cpu.Target())


def test_launch_extent_rejects_raw_dimvar() -> None:
    """A grid/block extent slot must be an Expr.

    A grid/block extent slot must be an Expr; a raw ``DimVar`` Op (which is
    not an Expr) is rejected by verify ([tir §1.3](docs/spec/tir.md#13-primfunction)).
    """
    entry = _launch_entry_with_grid_x(DimVar("S", 1, 7))
    with pytest.raises(expr.VerifyError, match="extent"):
        verifier.verify_prim_function(entry)


def test_launch_extent_rejects_external_shapeof() -> None:
    """A grid/block ``ShapeOf`` extent must reference a forwarded / entry parameter.

    A grid/block ``ShapeOf`` extent must reference a forwarded / entry
    parameter; a ShapeOf of an unrelated Var is rejected by verify.
    """
    external = expr.Var(
        type=TensorType(shape=(8,), dtype=DType.f32, layout=types.Layout((8,)), storage="gmem"),
        name="y_external",
    )
    bad = core.ShapeOf(
        type=TensorType.from_scalar(DType.i32, storage=StorageKind.RMEM),
        param=external,
        axis=0,
    )
    entry = _launch_entry_with_grid_x(bad)
    with pytest.raises(expr.VerifyError, match="not a forwarded"):
        verifier.verify_prim_function(entry)
