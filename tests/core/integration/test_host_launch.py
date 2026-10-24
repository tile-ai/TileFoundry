"""End-to-end checks for the explicit host/device boundary."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import *
from tilefoundry.ir.core import Constant, Var, VerifyError
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types.layout import Layout
from tilefoundry.target import CpuTarget
from tilefoundry.visitor_registry.verify import verify_prim_function


def _launch_entry_with_grid_x(extent):
    """A CPU host entry whose single launch uses *extent* as ``grid_x``.

    A CPU host entry whose single launch uses *extent* as ``grid_x`` — for
    exercising the grid/block extent verifier on constructed IR.
    """
    t = TensorType(shape=(8,), dtype=DType.f32, layout=Layout((8,)), storage="gmem")
    a = Var(type=t, name="a")
    i64 = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
    one = Constant(type=i64, value=1)
    ref = SymbolRef(name="dev", type=CallableType(return_type=UnitType(), parameters=(t,)))
    args = (ref, extent, one, one, one, one, one, a)
    body = Sequential(body=(Evaluate(callable=Launch(), args=args),))
    return PrimFunction(name="host", params=(a,), body=body, target=CpuTarget())


def test_launch_extent_rejects_raw_dimvar() -> None:
    """A grid/block extent slot must be an Expr.

    A grid/block extent slot must be an Expr; a raw ``DimVar`` Op (which is
    not an Expr) is rejected by verify ([tir §1.3](docs/spec/tir.md#13-primfunction)).
    """
    entry = _launch_entry_with_grid_x(DimVar("S", 1, 7))
    with pytest.raises(VerifyError, match="extent"):
        verify_prim_function(entry)


def test_launch_extent_rejects_external_shapeof() -> None:
    """A grid/block ``ShapeOf`` extent must reference a forwarded / entry parameter.

    A grid/block ``ShapeOf`` extent must reference a forwarded / entry
    parameter; a ShapeOf of an unrelated Var is rejected by verify.
    """
    external = Var(
        type=TensorType(shape=(8,), dtype=DType.f32, layout=Layout((8,)), storage="gmem"),
        name="y_external",
    )
    bad = ShapeOf(
        type=TensorType.scalar(DType.i32, storage=StorageKind.RMEM),
        param=external,
        axis=0,
    )
    entry = _launch_entry_with_grid_x(bad)
    with pytest.raises(VerifyError, match="not a forwarded"):
        verify_prim_function(entry)
