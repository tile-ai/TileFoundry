"""What HirToTirPass refuses, what it decides about storage, and where its walk
has to reach.

That the pass produces a well-formed ``PrimFunction`` for a real program is
settled by every test that compiles one and runs it: a malformed lowering does not
produce CUDA that computes the right answer. Pinning the emitted statement
sequence here as well would fix the shape of a lowering that is free to change
while it keeps producing the same kernel.

What a runtime witness cannot say is why a program was rejected, which residency a
value ended up in when two readings were available, and whether a walk reached a
child that only some inputs have. Those are here.
"""

from __future__ import annotations

import pytest

# DSL surface imported at module scope so ``@func`` closure
# resolution sees ``Tensor`` / ``Mesh`` / ... when the tests below
# build inline @func definitions.
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
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.types import DType, TensorType
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
    """An unmaterialized value must not reach TIR: a function param carrying
    `StorageKind.UMAT` (e.g. an explicit `Tensor[..., StorageKind.UMAT]`
    annotation or programmatic IR) is rejected at the HIR->TIR boundary, since
    a kernel param has no memory space for the launch ABI / placement."""

    @func
    def f(x: Tensor[(8,), "f32", None, StorageKind.UMAT]) -> Tensor[(8,), "f32"]:
        return x

    fn = f
    module = Module(name="t", functions=(fn,), entry=fn.name)
    with pytest.raises(ValueError, match="unmaterialized"):
        HirToTirPass().run(module)


def test_binary_dst_storage_follows_hir_output_not_operand_order() -> None:
    """A value literal lowers/materializes to a register buffer, but the TIR
    Binary destination follows the HIR-resolved output residency (the gmem
    tensor operand), independent of which side the literal is on — no
    operand-order dependence reintroduced at lowering."""

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
    """When every Split on the reduce axis sits on a ``thread``
    topology, ``__shfl_xor_sync`` covers the cross-lane fold and
    no workspace is needed; lowering must emit the 2-arg
    ``Reduce(src, dst)`` form."""

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


# ── ExprVisitor-based HIR walks reach every child ([visitor-mutator §1](docs/spec/visitor-mutator.md#1-role))


def test_the_hir_walks_reach_every_child_of_a_grid_region() -> None:
    """A ``GridRegionExpr`` has children a hand-rolled Tuple/Call walk does not
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
    iv = Var(type=TensorType.scalar(dtype=DType.i32), name="i")
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

    scalar_ty = TensorType.scalar(dtype=DType.f32)
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
