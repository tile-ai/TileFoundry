"""End-to-end host-launch checks for the split host/device pipeline.

Exercises non-MMA kernels through the explicit host/device boundary:
- a multi-CTA elementwise kernel (each CTA owns a distinct output row) run via
  both the implicit auto-inserted CPU entry and an explicit
  ``@prim_func(target="cpu")`` + ``launch``;
- a within-CTA per-row reduce;
- a device-placement negative (a CPU tensor handed to a CUDA launch errors at
  the host wrapper);
- a dynamic (launch-provided) CTA extent: a kernel tiled ``(Ntile, TILE)`` with
  the dynamic outer ``Ntile`` axis split across CTAs, launched explicitly so one
  compiled artifact runs two different ``Ntile`` shapes without recompiling, plus
  the negative that such a kernel cannot use the implicit auto-inserted entry.
"""
from __future__ import annotations

import pytest
import torch

import tilefoundry
from tilefoundry import func, prim_func
from tilefoundry.dsl import DimVar, ReduceKind, Tensor, tf
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.dsl.tf import *  # noqa: F401,F403  -- bind bare op names (reshard, relu, ...)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology

_ROWS = 128
_COLS = 12


# Multi-CTA elementwise: the cta mesh splits the rows, so CTA i owns row i.
# Running only one CTA would leave the other rows wrong, which the full-output
# assertion catches.
@func(topologies=(Topology("cta", _ROWS),))
def double_rows(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        reg = reshard(a, layout=(128 @ cta, 12), storage=rmem)  # noqa: F405
        out = tf.mul(reg, reg)
        return reshard(out, layout=(128 @ cta, 12), storage=gmem)  # noqa: F405


# Explicit host entry launching the device kernel above.
@prim_func(target="cpu")
def host_double(a: Tensor[(_ROWS, _COLS), "f32"], out: Tensor[(_ROWS, _COLS), "f32"]):
    launch(double_rows, a, out, grid=(_ROWS, 1, 1), block=(1, 1, 1))  # noqa: F821


# Within-CTA per-row reduce: a single CTA reduces the row across its threads.
@func(topologies=(Topology("thread", 6 * 32),))
def row_mean(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1), "f32"]:
    with Mesh(Topology("thread", 6 * 32), (6, 32), ("w", "t")) as m:
        a_reg = tf.reshard(a, (1, 1536 @ (m.w, m.t)), rmem)
        a_mean = tf.reduce(a_reg, (-1,), True, ReduceKind.MEAN)
        return tf.reshard(a_mean, (1, 1), gmem)


def _randn_rows() -> torch.Tensor:
    torch.manual_seed(0)
    # Non-uniform per-row data so a wrong CTA region cannot pass by luck.
    return torch.randn(_ROWS, _COLS, dtype=torch.float32, device="cuda")


def test_implicit_entry_elementwise_multi_cta() -> None:
    """Multi-CTA elementwise via the implicit auto-inserted host entry."""
    rm = tilefoundry.compile(
        Module(name="m", functions=(double_rows,), entry="double_rows"),
        target="cuda",
    )
    x = _randn_rows()
    out = torch.empty_like(x)
    rm(x, out)
    torch.cuda.synchronize()
    assert torch.allclose(out, x * x, rtol=0, atol=0)


def test_explicit_host_entry_elementwise_multi_cta() -> None:
    """Same kernel launched from an explicit ``@prim_func(target="cpu")`` entry."""
    rm = tilefoundry.compile(
        Module(
            name="m",
            functions=(double_rows, host_double),
            entry="host_double",
        ),
        target="cuda",
    )
    assert rm.entry == "host_double"
    x = _randn_rows()
    out = torch.empty_like(x)
    rm(x, out)
    torch.cuda.synchronize()
    assert torch.allclose(out, x * x, rtol=0, atol=0)


def test_implicit_entry_within_cta_reduce() -> None:
    """Within-CTA per-row reduce via the implicit host entry."""
    rm = tilefoundry.compile(
        Module(name="m", functions=(row_mean,), entry="row_mean"),
        target="cuda",
    )
    torch.manual_seed(1)
    x = torch.randn(1, 1536, dtype=torch.float32, device="cuda")
    out = torch.empty(1, 1, dtype=torch.float32, device="cuda")
    rm(x, out)
    torch.cuda.synchronize()
    assert torch.allclose(out, x.mean(dim=1, keepdim=True), rtol=1e-5, atol=1e-5)


def test_launch_rejects_cpu_tensor_at_host_wrapper() -> None:
    """A CPU tensor handed to a CUDA launch must error at the host wrapper's
    device-placement check (naming the argument and the expected device)."""
    rm = tilefoundry.compile(
        Module(name="m", functions=(double_rows,), entry="double_rows"),
        target="cuda",
    )
    x_cpu = torch.randn(_ROWS, _COLS, dtype=torch.float32)  # host tensor
    out = torch.empty(_ROWS, _COLS, dtype=torch.float32, device="cuda")
    # The host wrapper's placement check names the offending argument and the
    # expected device type: ``tilefoundry: argument 'a' must be a kDLCUDA tensor``.
    with pytest.raises(Exception, match=r"argument 'a' must be a kDLCUDA tensor"):
        rm(x_cpu, out)


# --- Dynamic (launch-provided) CTA extent ------------------------------------
#
# The tensor is tiled ``(Ntile, TILE)``: the outer ``Ntile`` axis is a dynamic
# dimension variable (the CTA count, supplied by the host launch) and ``TILE``
# is the static inner extent each CTA owns. The cta mesh extent is ``None``
# (launch-provided), and the cta ``Split`` acts only on the dynamic outer axis,
# so each CTA owns a single ``(1, TILE)`` tile — a static register fragment with
# a dynamic number of CTAs.
_TILE = 12
_NT = DimVar("Ntile", 1, 64)


@func(topologies=(Topology("cta", None),))
def dyn_double(a: Tensor[(_NT, _TILE), "f32"]) -> Tensor[(_NT, _TILE), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(None,), strides=(1,))) as cta:
        reg = reshard(  # noqa: F405
            a,
            layout=ShardLayout(
                layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                attrs=(S(0),),
                mesh=cta,
            ),
            storage=rmem,
        )
        out = tf.mul(reg, reg)
        return reshard(  # noqa: F405
            out,
            layout=ShardLayout(
                layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                attrs=(S(0),),
                mesh=cta,
            ),
            storage=gmem,
        )


# A dynamic CTA extent has no compile-time grid, so it must be launched
# explicitly — the host wrapper reads the grid from the tensor's runtime shape.
@prim_func(target="cpu")
def host_dyn_double(a: Tensor[(_NT, _TILE), "f32"], out: Tensor[(_NT, _TILE), "f32"]):
    launch(dyn_double, a, out, grid=(_NT, 1, 1), block=(1, 1, 1))  # noqa: F821


def _dyn_module() -> Module:
    return Module(
        name="m",
        functions=(dyn_double, host_dyn_double),
        entry="host_dyn_double",
    )


def test_dynamic_cta_two_shapes_one_compile() -> None:
    """One compiled artifact launches the dynamic-CTA kernel at two different
    ``Ntile`` shapes via the host-computed grid; both match torch with no
    recompile."""
    rm = tilefoundry.compile(_dyn_module(), target="cuda")
    for nt in (4, 8):
        torch.manual_seed(nt)
        x = torch.randn(nt, _TILE, dtype=torch.float32, device="cuda")
        out = torch.empty_like(x)
        rm(x, out)
        torch.cuda.synchronize()
        assert torch.allclose(out, x * x, rtol=0, atol=0)


def test_dynamic_cta_device_source_reads_runtime_extent() -> None:
    """The device fragment reads the dynamic CTA extent at runtime: the global
    layout dim is the tensor's hidden shape scalar, the cta mesh extent is
    ``program_dim<cta>()``, and no constexpr ``program_shape<cta>`` is emitted
    (a regression that hardcoded the cta count would fail here)."""
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(_dyn_module(), target="cuda")
    cuda_fns = group_functions_by_target(lowered)["cuda"]
    src = emit_cuda_module(cuda_fns).source

    assert "tilefoundry::program_dim<tilefoundry::TopologyScope::cta>()" in src
    assert "program_shape<tilefoundry::TopologyScope::cta>" not in src
    # The dynamic global dim flows through the kernel's hidden shape scalar.
    assert "a_shape_0" in src


def test_dynamic_cta_rejects_implicit_entry() -> None:
    """A dynamic-CTA kernel has no compile-time grid, so the implicit
    auto-inserted host entry cannot derive one — it must error loudly rather
    than guess a CTA count."""
    mod = Module(
        name="m", functions=(dyn_double,), entry="dyn_double"
    )
    with pytest.raises(Exception, match=r"cannot derive its grid"):
        tilefoundry.compile(mod, target="cuda")


def test_dynamic_thread_topology_rejected() -> None:
    """Only a ``cta`` topology may carry a launch-provided (``None``) extent; a
    dynamic thread/warp extent has no launch source and is rejected at
    construction."""
    with pytest.raises(ValueError, match=r"only a 'cta' topology"):
        Topology("thread", None)


def _launch_entry_with_grid_x(extent):
    """A CPU host entry whose single launch uses *extent* as ``grid_x`` — for
    exercising the grid/block extent verifier on constructed IR."""
    from tilefoundry.ir.core import Constant, Var  # noqa: PLC0415
    from tilefoundry.ir.target import CpuTarget  # noqa: PLC0415
    from tilefoundry.ir.tir.launch import Launch  # noqa: PLC0415
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415
    from tilefoundry.ir.tir.stmts import Evaluate, Sequential  # noqa: PLC0415
    from tilefoundry.ir.tir.symbol_ref import SymbolRef  # noqa: PLC0415
    from tilefoundry.ir.types import CallableType, DType, TensorType, UnitType  # noqa: PLC0415

    t = TensorType(shape=(8,), dtype=DType.f32, layout=None, storage="gmem")
    a = Var(type=t, name="a")
    i64 = TensorType.scalar(DType.i64)
    one = Constant(type=i64, value=1)
    ref = SymbolRef(
        name="dev", type=CallableType(return_type=UnitType(), parameters=(t,))
    )
    args = (ref, extent, one, one, one, one, one, a)
    body = Sequential(body=(Evaluate(callable=Launch(), args=args),))
    return PrimFunction(name="host", params=(a,), body=body, target=CpuTarget())


def test_launch_extent_rejects_raw_dimvar() -> None:
    """A grid/block extent slot must be an Expr; a raw ``DimVar`` Op (which is
    not an Expr) is rejected by verify (tir.md §3.6)."""
    from tilefoundry.ir.core import VerifyError  # noqa: PLC0415
    from tilefoundry.ir.tir.verify import verify_prim_function  # noqa: PLC0415

    entry = _launch_entry_with_grid_x(DimVar("S", 1, 8))
    with pytest.raises(VerifyError, match="extent"):
        verify_prim_function(entry)


def test_launch_extent_rejects_external_shapeof() -> None:
    """A grid/block ``ShapeOf`` extent must reference a forwarded / entry
    parameter; a ShapeOf of an unrelated Var is rejected by verify."""
    from tilefoundry.ir.core import Var, VerifyError  # noqa: PLC0415
    from tilefoundry.ir.tir.shape import ShapeOf  # noqa: PLC0415
    from tilefoundry.ir.tir.verify import verify_prim_function  # noqa: PLC0415
    from tilefoundry.ir.types import DType, TensorType  # noqa: PLC0415

    external = Var(
        type=TensorType(shape=(8,), dtype=DType.f32, layout=None, storage="gmem"),
        name="y_external",
    )
    bad = ShapeOf(type=TensorType.scalar(DType.i32), param=external, axis=0)
    entry = _launch_entry_with_grid_x(bad)
    with pytest.raises(VerifyError, match="not a forwarded"):
        verify_prim_function(entry)
