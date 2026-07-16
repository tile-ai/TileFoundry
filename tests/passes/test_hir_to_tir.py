"""HirToTirPass produces a well-formed tir.PrimFunction for the demo IR."""

from __future__ import annotations

import textwrap

import pytest

from tests.models.demo.demo_ir import build_demo

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
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.tir.arith import Binary as TirBinary
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, MeshScope, Return, Sequential
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout as SL
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.passes.transforms import HirToTirPass


def _run() -> PrimFunction:
    fn, _, _ = build_demo()
    module = Module(name="t", functions=(fn,), entry=fn.name)
    new_module = HirToTirPass().run(module)
    [pf] = new_module.functions
    assert isinstance(pf, PrimFunction)
    return pf


def _flatten(seq: Sequential):
    """Flatten a LetStmt-nested Sequential into a list of its dataflow
    operations in emission order (LetStmts become (_Bind, var, op_type);
    plain Stmts pass through; ``Evaluate(SomeOp, ...)`` is rendered
    as the Op's class name — effect-ful Ops live in
    Stmt position via ``Evaluate(op, args)``)."""
    out = []
    for s in seq:
        if isinstance(s, LetStmt):
            assert isinstance(s.value, Call)
            out.append(("Let", s.var.name, type(s.value.target).__name__))
            out.extend(_flatten(s.body))
        elif isinstance(s, Evaluate):
            out.append(type(s.callable).__name__)
        else:
            out.append(type(s).__name__)
    return out


def test_demo_lowers_to_prim_function():
    pf = _run()
    assert pf.name == "demo"
    # HirToTirPass adds explicit out param.
    assert len(pf.params) == 2
    assert pf.params[0].name == "a"
    assert pf.params[1].name == "out"
    assert pf.params[1].type.shape == (1, 1536)
    assert pf.params[1].type.storage == StorageKind.GMEM
    assert pf.params[1].type.layout is None
    assert isinstance(pf.body, Sequential)
    assert len(pf.body) == 2
    outer, ret = pf.body
    assert isinstance(outer, MeshScope)
    assert isinstance(ret, Return)
    assert isinstance(outer.body, Sequential)
    assert len(outer.body) == 1
    inner = outer.body[0]
    assert isinstance(inner, MeshScope)
    assert isinstance(inner.body, Sequential)

    flat = _flatten(inner.body)
    # function-end sink writes the result into the `out` param
    # directly, not into a fresh global AllocTensor — one fewer LetStmt.
    expected = [
        ("Let", "sv1", "TensorView"),   # shard view of param 'a'
        ("Let", "t2", "AllocTensor"),   # b = alloc shared (plain)
        "Copy",                         # shard_view(a) → shared
        ("Let", "ptr3", "PtrOf"),       # ptr to shared buffer (sharded source)
        ("Let", "sv4", "TensorView"),   # shard view of shared result
        ("Let", "t5", "AllocTensor"),   # d0 = alloc reg (plain)
        "Copy",                         # shard_view(shared) → reg
        ("Let", "r6", "AllocTensor"),   # d1 = ReLU(d0): allocate output...
        "ReLU",                         # ...then effect-stmt pointwise
        ("Let", "sv7", "TensorView"),   # shard view of out param
        "Copy",                         # shard_view(reg) → out param
    ]
    assert flat == expected


def test_lowered_copy_storage_chain():
    pf = _run()
    inner_body = pf.body[0].body[0].body  # cta -> thread -> Sequential

    # Walk LetStmt chain to collect Copy calls in order. Copy is
    # an Op invoked as Evaluate(Copy, (source, destination)).
    copies = []

    def walk(seq):
        for s in seq:
            if isinstance(s, LetStmt):
                walk(s.body)
            elif isinstance(s, Evaluate) and isinstance(s.callable, Copy):
                copies.append(s)

    walk(inner_body)
    storages = [c.args[1].type.storage for c in copies]
    # Output scatter: Copy writes to TensorView(out, shard_layout), storage=GMEM
    assert storages == [StorageKind.SMEM, StorageKind.RMEM, StorageKind.GMEM]
    # The function-end Copy writes via a shard view of the out param,
    # not directly to the out Var (plain→shard scatter).
    last_dest = copies[-1].args[1]
    assert isinstance(last_dest.type.layout, SL)
    assert last_dest.type.storage == StorageKind.GMEM


# ── lowering does not fabricate mesh structure ─────────────────────


def test_lower_cta_only_kernel_skips_thread_mesh_scope() -> None:
    """A function whose ShardLayouts only reference a ``cta``-topology mesh
    lowers to a single outer ``MeshScope`` — no synthetic inner ``thread``
    scope: lowering must not invent meshes beyond what the body
    actually uses."""


    src = textwrap.dedent("""
    from tilefoundry import func
    from tilefoundry.dsl.tf import *
    from tilefoundry.dsl import Tensor
    from tilefoundry.ir.types import DType
    from tilefoundry.ir.types.shard.mesh import Mesh, Topology

    @func(topologies=(Topology("cta", 128),))
    def f(x: Tensor[(1, 2048), DType.f32]) -> Tensor[(1, 2048), DType.f32]:
        with Mesh(topology="cta", layout=(128,)) as cta:
            y = reshard(x, layout=(1, 2048 @ cta), storage="smem")
            z = relu(y)
            return reshard(z, layout=(1, 2048 @ cta), storage="gmem")
    """).lstrip()

    fn = parse_script(src)
    module = Module(name="t", functions=(fn,), entry=fn.name)
    new_module = HirToTirPass().run(module)
    [pf] = new_module.functions

    assert isinstance(pf.body, Sequential)
    outer = pf.body[0]
    assert isinstance(outer, MeshScope)
    assert outer.mesh.topology.name == "cta"
    # No inner thread MeshScope — body is the CTA-level Sequential directly.
    inner_first = outer.body[0]
    assert not isinstance(inner_first, MeshScope), (
        f"expected no synthetic thread MeshScope; got {type(inner_first).__name__}"
    )


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

    fn = f
    module = Module(name="t", functions=(fn,), entry=fn.name)
    pf = HirToTirPass().run(module).functions[0]

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
