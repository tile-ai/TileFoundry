"""Cover HirToTirPass rejection, storage decisions, and traversal reachability.

Tests target diagnostics and decisions not observable from successful kernel
execution, without pinning an interchangeable statement sequence.

See [passes §7.1](docs/spec/passes.md#71-hirtotirpass).
"""

from __future__ import annotations

import pytest

from tests.fixtures.shapes.window_programs import moved_tile_window_add, tile_window_add
from tilefoundry.dsl import (
    Mesh,
    ReduceKind,
    Tensor,
    Topology,
    func,
    tf,
)
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.nn.relu import ReLU as HirReLU
from tilefoundry.ir.tir.arith import Binary as TirBinary
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimAdd, DimMul
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import ShardLayout as SL
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.passes.transforms import HirToTirPass
from tilefoundry.passes.transforms.hir_to_tir import (
    _collect_hir_callee_names,
    _derive_meshes_from_body,
)


def test_umat_param_rejected_at_lowering() -> None:
    """Test umat param rejected at lowering.

    An unmaterialized value must not reach TIR: a function param carrying
    `StorageKind.UMAT` (e.g. an explicit `Tensor[..., StorageKind.UMAT]`
    annotation or programmatic IR) is rejected at the HIR->TIR boundary, since
    a kernel param has no memory space for the launch ABI / placement.
    """

    @func
    def f(x: Tensor[(8,), "f32", None, StorageKind.UMAT]) -> Tensor[(8,), "f32"]:
        return x

    fn = f
    module = Module(name="t", functions=(fn,), entry=fn.name)
    with pytest.raises(ValueError, match="unmaterialized"):
        HirToTirPass().run(module)


def test_binary_dst_storage_follows_hir_output_not_operand_order() -> None:
    """Test binary dst storage follows hir output not operand order.

    A value literal lowers/materializes to a register buffer, but the TIR
    Binary destination follows the HIR-resolved output residency (the gmem
    tensor operand), independent of which side the literal is on — no
    operand-order dependence reintroduced at lowering.
    """

    @func
    def _lit_rhs(x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 8), "f32"]:
        return tf.add(x, 1.0)

    @func
    def _lit_lhs(x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 8), "f32"]:
        return tf.add(1.0, x)

    def _binary_dst_storage(fn) -> StorageKind:
        module = Module(name="t", functions=(fn,), entry=fn.name)
        pf = HirToTirPass().run(module).functions[0]
        found = []

        def walk(s):
            if isinstance(s, Evaluate) and isinstance(s.callable, TirBinary):
                found.append(s.args[2].type.storage)
            for attr in ("body", "stmts"):
                v = getattr(s, attr, None)
                if isinstance(v, (list, tuple)):
                    for sub in v:
                        walk(sub)
                elif v is not None and hasattr(v, "__dict__"):
                    walk(v)

        walk(pf.body)
        assert len(found) == 1, f"expected one TIR Binary, got {len(found)}"
        return found[0]

    assert _binary_dst_storage(_lit_rhs) == StorageKind.GMEM
    assert _binary_dst_storage(_lit_lhs) == StorageKind.GMEM


def test_hir_reduce_no_workspace_when_only_thread_topology_split() -> None:
    """Test hir reduce no workspace when only thread topology split.

    When every Split on the reduce axis sits on a ``thread``
    topology, ``__shfl_xor_sync`` covers the cross-lane fold and
    no workspace is needed; lowering must emit the 2-arg
    ``Reduce(src, dst)`` form.
    """

    @func(topologies=(Topology("thread", 32),))
    def f(a: Tensor[(1, 256), DType.f32]):
        with Mesh(("thread",), (32,), ("t",)) as m:
            a_reg = tf.reshard(a, (1, 32 @ m.t, 8), "rmem")
            return tf.reduce(a_reg, (-1,), True, ReduceKind.SUM)

    pf = HirToTirPass().run(f).functions[0]

    def _find_reduce(s):
        if isinstance(s, Evaluate) and isinstance(s.callable, TirReduce):
            return s
        for attr in ("body", "stmts"):
            v = getattr(s, attr, None)
            if isinstance(v, (list, tuple)):
                for sub in v:
                    r = _find_reduce(sub)
                    if r is not None:
                        return r
            elif hasattr(v, "__dict__"):
                r = _find_reduce(v)
                if r is not None:
                    return r
        return None

    reduce_call = _find_reduce(pf.body)
    assert reduce_call is not None
    assert len(reduce_call.args) == 2, (
        f"intra-warp reduce should have 2 args (src, dst), got {len(reduce_call.args)}"
    )


def test_the_hir_walks_reach_every_child_of_a_grid_region() -> None:
    """A ``GridRegionExpr`` has children a hand-rolled Tuple/Call walk does not know about.

    A ``GridRegionExpr`` has children a hand-rolled Tuple/Call walk does not
    know about, and both of the pass's own walks depend on reaching them.

    A mesh referenced only inside the region's body must still be derived --
    missing it lowers a sharded kernel as if it had no mesh. A callee reachable
    only through ``yield_values`` must still be collected -- missing it drops an
    inter-group dependency edge in dispatch-group ordering. Neither failure is
    visible in the pass's output shape; both are wrong programs.
    """
    mesh = Mesh(
        topologies=(Topology("cta", 4),),
        layout=Layout(shape=(4,), strides=(1,)),
    )
    sl = SL(layout=Layout(shape=(4,), strides=(1,)), attrs=(Split(0),), mesh=mesh)
    body_ty = TensorType(shape=(4,), dtype=DType.f32, layout=sl, storage=StorageKind.GMEM)
    iv = Var(
        type=TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM), name="i"
    )
    in_body = GridRegionExpr(
        type=body_ty,
        induction_var=iv,
        carried_args=(),
        init_args=(),
        body=Call(type=body_ty, target=HirReLU(), args=()),
        yield_values=(),
        extent=4,
        step=1,
    )

    cta_mesh, thread_mesh = _derive_meshes_from_body(in_body)
    assert cta_mesh is mesh
    assert thread_mesh is None

    scalar_ty = TensorType.scalar(dtype=DType.f32, storage=StorageKind.RMEM)
    callee = HirFunction.build(
        name="callee_fn",
        params=(),
        body=Constant(value=0.0, type=scalar_ty),
        return_type=scalar_ty,
    )
    in_yield = GridRegionExpr(
        type=scalar_ty,
        induction_var=iv,
        carried_args=(),
        init_args=(),
        body=Constant(value=0.0, type=scalar_ty),
        yield_values=(Call(type=scalar_ty, target=callee, args=()),),
        extent=1,
        step=1,
    )

    assert _collect_hir_callee_names(in_yield) == {"callee_fn"}


def test_grid_output_ordinal_lowers_to_an_absolute_element_start() -> None:
    """Preserve grid output placement under absolute TensorView coordinates."""
    body_ty = TensorType(
        shape=(2,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM
    )
    grid_ty = TensorType(
        shape=(4, 2), dtype=DType.f32, layout=None, storage=StorageKind.GMEM
    )
    x = Var(type=body_ty, name="x")
    iv = Var(
        type=TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM), name="i"
    )
    grid = GridRegionExpr(
        type=grid_ty,
        induction_var=iv,
        carried_args=(),
        init_args=(),
        body=Call(type=body_ty, target=HirReLU(), args=(x,)),
        yield_values=(),
        extent=4,
        step=1,
    )
    fn = HirFunction.build(
        name="grid_output",
        params=(x,),
        body=grid,
        return_type=grid_ty,
    )
    pf = HirToTirPass().run(Module(name="t", functions=(fn,), entry=fn.name)).functions[0]

    views = []

    def walk(stmt):
        if (
            isinstance(stmt, LetStmt)
            and isinstance(stmt.value, Call)
            and isinstance(stmt.value.target, TensorView)
            and len(stmt.value.args) == 2
        ):
            views.append(stmt.value)
        for attr in ("body", "stmts"):
            child = getattr(stmt, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    walk(item)
            elif child is not None and hasattr(child, "__dict__"):
                walk(child)

    walk(pf.body)
    assert len(views) == 1
    coordinate = views[0].args[1]
    assert isinstance(coordinate, Call) and isinstance(coordinate.target, DimMul)
    assert coordinate.args[0] is iv
    assert isinstance(coordinate.args[1], Constant) and coordinate.args[1].value == 2


def test_a_moved_window_lowers_to_the_moved_address() -> None:
    """A read window moved by a compile-time offset is an address, not a value.

    The offset reaches the coordinate as arithmetic over the induction variable,
    so nothing is materialized to hold it.
    """
    f = moved_tile_window_add
    pf = HirToTirPass().run(Module(name="t", functions=(f,), entry=f.name)).lookup(f.name)
    coordinates = []

    def walk(stmt):
        if (
            isinstance(stmt, LetStmt)
            and isinstance(stmt.value, Call)
            and isinstance(stmt.value.target, TensorView)
            and len(stmt.value.args) == 3
        ):
            coordinates.append(stmt.value.args[1])
        for attr in ("body", "stmts"):
            child = getattr(stmt, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    walk(item)
            elif child is not None and hasattr(child, "__dict__"):
                walk(child)

    walk(pf.body)
    moved = [c for c in coordinates if isinstance(c, Call) and isinstance(c.target, DimAdd)]
    assert len(moved) == 1, coordinates
    assert isinstance(moved[0].args[0], Var)
    assert isinstance(moved[0].args[1], Constant) and moved[0].args[1].value == 6


def test_a_computed_window_coordinate_fails_closed() -> None:
    """An offset an op computes is refused, not materialized into a buffer.

    A write coordinate spelled as `i + C` reaches lowering as an ordinary scalar
    add rather than as an address, and a buffer holding its value is not a scalar
    index. Lowering says so instead of emitting code that does not compile.
    """

    @func
    def f(x: Tensor[(8, 4), "f32"], seed: Tensor[(2, 4), "f32"]):
        acc = tf.add(seed, seed)
        out = x
        for row in tile(4, 2):
            out = tf.insert_slice(out, acc, (row + 4, 0))
        return out

    with pytest.raises(NotImplementedError, match="coordinate computed by Binary"):
        HirToTirPass().run(Module(name="t", functions=(f,), entry=f.name))


def test_non_divisible_tile_window_lowering_fails_closed() -> None:
    f = tile_window_add
    with pytest.raises(NotImplementedError, match="requires handwritten tail lowering"):
        HirToTirPass().run(Module(name="t", functions=(f,), entry=f.name))
