"""Every subject ``tests/parser`` refuses, in one table.

A row is one subject and the diagnostic it must raise. ``subject`` is DSL source
to import, or a builder for the ones with no source to feed: a definition that
must fail while it is decorated, a hand-forged TIR node fed to the verifier, an
operand check that never reaches the parser. ``test_refused_programs.py`` is the
only entry point, so a new refusal is a new row rather than a new file. The
subjects the rows share live here too, and the surviving test files import them
from here instead of keeping a copy each.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tests._source import import_dsl
from tests.fixtures.logical.hir_composition import Expert
from tests.fixtures.shapes.window_programs import tile_window_add
from tilefoundry import func, module, prim_func
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.dsl import ConstTensor, DimVar, T, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op names used by the @func bodies
from tilefoundry.evaluator import EvalError, evaluate
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.ir.tir.sync import Sync, classify
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimAdd, simplify_dim
from tilefoundry.ir.types.shard import Layout, Mesh, P, Topology
from tilefoundry.ir.types.shard.layout import ComposedLayout
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.module import _DECLARING
from tilefoundry.parser.sugar import parse_shard_layout_sugar
from tilefoundry.target import CudaTarget


@dataclass(frozen=True)
class ParseErrorCase:
    """One refused subject: what is fed in, and the diagnostic it must raise."""

    id: str
    subject: str | Callable[[], object]
    """DSL source to import, or a builder for the ones with no source to feed."""
    raises: type[BaseException]
    match: str
    """The diagnostic, tight enough that no other row in the table satisfies it.

    A pattern loose enough to accept a neighbour's message stops asserting
    which thing went wrong, and a row can then drift off the check it was
    written for without anything noticing.
    """


def run_parse_error_case(case: ParseErrorCase) -> None:
    """Run one ``ParseErrorCase``: the subject is refused with that diagnostic."""
    with pytest.raises(case.raises, match=case.match):
        if callable(case.subject):
            case.subject()
        else:
            import_dsl(textwrap.dedent(case.subject).lstrip("\n"))


HIR_PRELUDE = """from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor
"""

_ONE_TENSOR = 'x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]'


def hir_source(*body: str, signature: str = _ONE_TENSOR, prelude: str = HIR_PRELUDE) -> str:
    """A one-``@func`` script.

    *signature* closes the parameter list and states the return annotation,
    *body* lines carry their own nesting, *prelude* holds module-level bindings
    the parser resolves through ``fn.__globals__``.
    """
    lines = "\n".join(f"    {line}" for line in body)
    return f"{prelude}\n@func\ndef f({signature}:\n{lines}\n"


_WHERE_PRELUDE = (
    "from tilefoundry.ir.types.shard import Layout, Mesh, Topology\n\n"
    'cta_mesh = Mesh((Topology("cta", 8),), Layout((8,), (1,)))'
)


def where_source(body: str, preamble: str = _WHERE_PRELUDE, ret: str = '(8, 16), "bf16"') -> str:
    """A one-``@func`` script carrying ``where(...)`` annotations on its body."""
    return f"""from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

{preamble}

@func
def candidate(x: Tensor[(8, 16), "bf16"]) -> Tensor[{ret}]:
{body}
"""


@module(entry="entry")
class Callee:
    """A child Module whose entry calls a sibling by its bare class-body binding."""

    @func
    def helper(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return tf.mul(x, x)

    @func
    def entry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return helper(x)  # noqa: F821 — sibling binding in the class body


@module
class NoEntry:
    """A Module with no callable entry, so calling it has nothing to reach."""

    @func
    def only(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return tf.add(x, x)


@module(entry="device")
class PrimEntry:
    """A Module whose entry is a device ``@prim_func`` rather than an HIR Function."""

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def device(x: Tensor[(8,), "f32"]) -> None:  # noqa: ARG001
        with Mesh((Topology("thread", 8),), Layout(shape=(8,), strides=(1,))) as m:
            T.sync(m)


def _arg_type_mismatch() -> None:
    """A nested call whose argument dtype does not match the callee's parameter."""

    @func
    def _inner_double(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return add(x, x)  # noqa: F821

    @func
    def _bad_dtype(x: Tensor[(8, 64), "bf16"]) -> Tensor[(8, 64), "f32"]:
        return _inner_double(x)  # noqa: F841


def _reach_a_child_entry_by_name() -> None:
    @module(entry="reach")
    class _ReachEntry:
        leaf = Callee

        @func
        def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return leaf.entry(x)  # noqa: F821


def _reach_a_child_helper_by_name() -> None:
    @module(entry="reach")
    class _ReachHelper:
        leaf = Callee

        @func
        def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return leaf.helper(x)  # noqa: F821


def _reach_a_module_member_by_class() -> None:
    @func
    def _reach_by_class(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return Callee.entry(x)


def _call_a_module_with_no_entry() -> None:
    @module(entry="reach")
    class _CallsEntryless:
        leaf = NoEntry

        @func
        def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return leaf(x)  # noqa: F821


def _call_a_module_whose_entry_is_a_prim_func() -> None:
    @module(entry="reach")
    class _CallsPrim:
        leaf = PrimEntry

        @func
        def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return leaf(x)  # noqa: F821


def _bare_decorator_leaves_the_child_unattached() -> None:
    @module
    class _BareDecorated:
        leaf = Callee

        @func
        def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return leaf(x)  # noqa: F821


def _module_call_with_no_binding() -> None:
    """A Module named in a body no class-body binding attaches.

    This row and the list-bound one below own the only arcs in ``module.py``
    that nothing else under ``tests/parser`` reaches. A third subject reaching
    the same message from a ``@run.converter`` body owned none, so it is not a
    row.
    """

    @module(entry="reach")
    class _Unattached:
        @func
        def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return Callee(x)


def _module_call_bound_only_inside_a_list() -> None:
    @module(entry="reach")
    class _ListAttached:
        kids = [Callee]

        @func
        def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return Callee(x)


def _declaration_left_open_by_a_failed_class_body() -> None:
    """A class body that raises leaves its declaration open, resolving nothing.

    The two halves cannot be separate rows: the second only means anything while
    the first has left a declaration on the stack, and the stack has to be
    restored either way. So the setup asserts its own ``RuntimeError`` here and
    the row pins what the leaked declaration must *not* do. The standalone
    ``@func`` below is also the whole of a subject that used to be its own test
    and owned no arcs of its own, so that one is not a row.
    """
    open_declarations = len(_DECLARING)
    try:
        with pytest.raises(RuntimeError, match="boom"):

            @module(entry="never")
            class _Boom:
                raise RuntimeError("boom")

        @func
        def _after(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return Callee(x)
    finally:
        del _DECLARING[open_declarations:]


def _direct_call_of_the_wrong_arity() -> None:
    """A sibling ``@func`` called with fewer arguments than it declares.

    A standalone ``@func`` reaching the same site with the shorter ``arity
    mismatch`` message owned no arcs of its own, and that message is a substring
    of this one, so it is not a row.
    """

    @module(entry="root", target=CudaTarget("nvidia.h200_sxm"))
    class _Direct:
        @func
        def leaf(x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return tf.matmul(x, w)

        @func
        def root(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return leaf(x)  # noqa: F821


def _child_call_of_the_wrong_width() -> None:
    @module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
    class _TooMany:
        mlp = Expert

        @func
        def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return mlp(x, x)  # noqa: F821


CTX_LEN = DimVar("CTX_LEN", 1, 4097)


def _evaluate_a_non_divisible_tile_window() -> None:
    evaluate(tile_window_add, torch.ones((10, 4)), torch.ones((4, 4)), device="cpu")


_M_MULTI = Mesh((Topology("thread", 6 * 32),), Layout((6, 32), (32, 1)), names=("w", "t"))
_M_STATE = Mesh(
    (Topology("thread", 4 * 2 * 16),), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)
_S_DYN = DimVar("seq_len", 1, 4)
_MESH_DIM_W = DimVar("W", 1, 8)


def _multi_axis_split_not_divisible() -> None:
    """A dim must be divisible by the product of the mesh extents; ``100 @`` is rejected."""

    @func
    def _bad(
        a: Tensor[(1, 100), "f32", (1, 100 @ (_M_MULTI.w, _M_MULTI.t)), "smem"],
    ) -> Tensor[(1, 100), "f32"]:
        return a


def _value_state_not_final() -> None:
    """The ``{...}`` value-state set is valid only as the last outer item."""

    @func
    def _bad(
        a: Tensor[
            (4, 64),
            "f32",
            ((4 @ _M_STATE.l, 64), {_M_STATE.t @ P("sum")}, (64, 1)),
            "smem",
        ],
    ) -> Tensor[(4, 64), "f32"]:
        return a


def _value_state_bare_p() -> None:
    """``P(...)`` in the value-state set requires its reduction argument."""

    @func
    def _bad(
        a: Tensor[(4, 64), "f32", ((4 @ _M_STATE.l, 64), {_M_STATE.t @ P()}), "smem"],
    ) -> Tensor[(4, 64), "f32"]:
        return a


def _mesh_coordinate_slices_a_placed_tensor() -> None:
    @func(topologies=(Topology("cta", 8),))
    def _bad(x: Tensor[(8,), "i64"]) -> Tensor[(4,), "i64"]:
        with Mesh(("cta",), layout=(8,), names=("w",)) as cta:
            placed = reshard(x, (8 @ cta.w,), "rmem")  # noqa: F405, F821
            return placed[cta.w : cta.w + 4]


def _unresolved_dynamic_split_axis() -> None:
    """Different symbolic extents cannot establish divisibility before binding."""
    other = DimVar("other", 1, 4)
    cta = Mesh((Topology("cta", other),), Layout((other,), (1,)), names=("cta",))
    node = ast.parse("(1, S @ cta, 32, 128)", mode="eval").body
    parse_shard_layout_sugar(node, lambda n: cta if n == "cta" else None, closure={"S": _S_DYN})


def mesh_dims_reshard_func(warps, lanes):
    """A reshard whose mesh shape comes from ``layout=(warps, lanes)``.

    The dims may be integer literals or closure Names — a closure int must
    resolve like the literal, and a dynamic ``DimVar`` in that static-extent
    position must be rejected.
    """

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(("thread",), layout=(warps, lanes), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F405, F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F405, F821

    return _f


def literal_reshard_func():
    """The all-literal twin of ``mesh_dims_reshard_func``.

    ``layout=(4, 32)`` mesh dims spelled out, so the closure-resolved builder
    above has something to print equal to.
    """

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(("thread",), layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F405, F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F405, F821

    return _f


def _symbolic_mesh_extent() -> None:
    """A symbolic mesh extent is valid even when a later split is undecidable."""
    mesh_dims_reshard_func(_MESH_DIM_W, 32)


def _bool_split_extent_single_axis() -> None:
    """A ``bool`` split extent in the single-axis form (``True @ m.w``) is rejected."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(("thread",), layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, True @ m.w), "rmem")  # noqa: F405, F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F405, F821


def _float_split_extent_single_axis() -> None:
    """A non-``bool`` split extent that is not a shape dimension names its own type.

    ``bool`` earns a separate diagnostic because it *is* an int; every other
    wrong type gets the plain one, and this row is what keeps that plain one
    from going unwritten.
    """

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(("thread",), layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 1.5 @ m.w), "rmem")  # noqa: F405, F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F405, F821


def _duplicate_topology_name() -> None:
    @func(topologies=(Topology("cta", 128), Topology("cta", 64)))
    def _dup(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
        return a


def _mesh_on_an_undeclared_topology() -> None:
    @func(topologies=(Topology("cta", 128),))
    def _unk(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
        with Mesh(("nonexistent",), layout=Layout(shape=(128,), strides=(1,))) as m:  # noqa: F841
            return a


def _mesh_topology_source(mesh_source: str) -> str:
    return (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Mesh, Tensor\n"
        "from tilefoundry.ir.types.shard import Topology\n\n"
        "@func(topologies=(Topology('cta', 128),))\n"
        "def f(a: Tensor[(128,), 'f32']):\n"
        f"    with {mesh_source} as cta:\n"
        "        return a\n"
    )


_DTYPE_HEADER = """
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *
"""

_BAD_CALL_DTYPE = (
    _DTYPE_HEADER
    + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="float32")
"""
)

_BAD_ANNOTATION_DTYPE = (
    _DTYPE_HEADER
    + """
@func
def f(x: Tensor[(8,), "float32"]) -> Tensor[(8,), "f32"]:
    return cast(x, dtype="f32")
"""
)

_BAD_REDUCE_KIND = (
    _DTYPE_HEADER
    + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="plus")
"""
)


def alloc_frag_kernel(topology, mesh_layout, names=()):
    """A kernel that allocs a fragment via ``atom.A`` inside the given scope."""

    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
        with Mesh((topology,), mesh_layout, names=names) as warp:  # noqa: F841
            frag = T.alloc_tensor(  # noqa: F841
                TensorType(
                    shape=(16, 16), dtype=DType.bf16, layout=atom.A, storage=StorageKind.RMEM
                )
            )

    return kernel


def _mma_scope(topology, layout) -> Callable[[], object]:
    """A builder that parses ``alloc_frag_kernel`` under one candidate thread scope."""
    return lambda: prim_func(target=CudaTarget("nvidia.h200_sxm"))(
        alloc_frag_kernel(topology, layout)
    )


def _atom_outside_a_mesh_scope() -> None:
    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
        frag = T.alloc_tensor(  # noqa: F841
            TensorType(shape=(16, 16), dtype=DType.bf16, layout=atom.A, storage=StorageKind.RMEM)
        )

    prim_func(target=CudaTarget("nvidia.h200_sxm"))(kernel)


def thread_mesh() -> Mesh:
    """A 128-thread block viewed as (4 warps, 32 lanes)."""
    return Mesh((Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t"))


def cta_mesh() -> Mesh:
    """A 128-CTA grid."""
    return Mesh(topologies=(Topology("cta", 128),), layout=Layout(shape=(128,), strides=(1,)))


def _binding(name: str = "m") -> Var:
    return Var(type=TensorType.scalar(DType.i64, storage=StorageKind.RMEM), name=name)


def scoped_sync(mesh: Mesh, sync_mesh: Mesh) -> PrimFunction:
    """A ``PrimFunction`` syncing on *sync_mesh* inside a mesh scope over *mesh*.

    The three rows built on this that share the ``enclosing`` diagnostic — no
    enclosing mesh at all, a forged sub-box exceeding its parent, a forged
    topology tuple — each own arcs in ``ir/tir/sync.py`` no other subject under
    ``tests/parser`` reaches. They are three judgements in front of one raise,
    so none of them stands in for another.
    """
    return PrimFunction(
        name="fn",
        params=(),
        body=Sequential(
            body=(
                MeshScope(
                    mesh=mesh,
                    binding=_binding(),
                    body=Sequential(
                        body=(Evaluate(callable=Sync(mesh=sync_mesh), args=()), Return())
                    ),
                ),
            )
        ),
    )


def _sync_argument_is_not_a_mesh() -> None:
    def kernel(a: Tensor[(128,), "f32"]):  # noqa: ARG001
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as m:  # noqa: F841
            T.sync(a)

    prim_func(target=CudaTarget("nvidia.h200_sxm"))(kernel)


def _sync_with_no_enclosing_mesh() -> None:
    verify_prim_function(
        PrimFunction(
            name="fn",
            params=(),
            body=Sequential(body=(Evaluate(callable=Sync(mesh=cta_mesh()), args=()), Return())),
        )
    )


def _sync_on_a_non_contiguous_slice() -> None:
    """A lane subset across warps (``m[:, 1:3]``) is not a contiguous thread interval."""
    m = thread_mesh()
    verify_prim_function(scoped_sync(m, m[:, 1:3]))


def _sync_on_a_mesh_the_scope_does_not_bind() -> None:
    """An un-sliced mesh the enclosing scope simply does not bind.

    This is the middle of the three ways to miss an enclosing mesh: a scope is
    in force, and the synced mesh is a whole mesh rather than a sub-box, so what
    is wrong is the identity of the mesh and not the shape of a slice.
    """
    enclosing = thread_mesh()
    other = Mesh((Topology("thread", 64),), Layout(shape=(64,), strides=(1,)))
    verify_prim_function(scoped_sync(enclosing, other))


def _sync_on_a_forged_subbox_exceeding_its_parent() -> None:
    """A (1, 64) sub-box of a (4, 32) parent is not constructible by ``Mesh.__getitem__``."""
    enclosing = thread_mesh()
    forged = Mesh(
        topologies=enclosing.topologies,
        layout=ComposedLayout(inner=None, offset=0, outer=Layout((1, 64), (32, 1))),
        names=enclosing.names,
    )
    verify_prim_function(scoped_sync(enclosing, forged))


def _sync_on_a_forged_topology_mismatch() -> None:
    """A forged sync mesh sharing the primary topology but not the full topology tuple."""
    enclosing = Mesh(
        topologies=(Topology("warp", 4), Topology("thread", 32)),
        layout=Layout(shape=(4, 32), strides=(32, 1)),
    )
    forged = Mesh(
        topologies=(Topology("warp", 4),),
        layout=ComposedLayout(inner=None, offset=0, outer=Layout((2, 32), (32, 1))),
        names=enclosing.names,
    )
    verify_prim_function(scoped_sync(enclosing, forged))


def _sync_on_a_cross_warp_unaligned_slice() -> None:
    """A contiguous but cross-warp-unaligned range (lanes 16..47)."""
    m = Mesh((Topology("thread", 64),), Layout(shape=(64,), strides=(1,)))
    verify_prim_function(scoped_sync(m, m[16:48]))


def _classify_a_partial_cta_slice() -> None:
    """A cta slice is a subset of CTAs, and no barrier covers that."""
    classify(cta_mesh()[0:64])


def _exhaust_the_named_barriers() -> None:
    ctx = CodegenContext()
    ctx.reset_barrier_ids()
    for _ in range(15):
        ctx.alloc_barrier_id()
    ctx.alloc_barrier_id()


_HIR_BODY_STATEMENTS: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="yield-in-hir-body",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl.tf import *
            from tilefoundry.dsl import Tensor

            @func
            def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                yield x
        """,
        raises=VerifyError,
        match=r"`yield` is not an HIR statement",
    ),
    ParseErrorCase(
        id="lambda-in-hir-body",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl.tf import *
            from tilefoundry.dsl import Tensor

            @func
            def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                g = lambda y: y
                return g(x)
        """,
        raises=VerifyError,
        match="Lambda",
    ),
    ParseErrorCase(
        id="augassign-in-grid-body",
        subject=hir_source("o = relu(x)", "for i in range(8):", "    o += x", "return o"),
        raises=VerifyError,
        match="augmented assignment",
    ),
    ParseErrorCase(
        id="return-in-grid-body",
        subject=hir_source("for i in range(8):", "    return x", "return x"),
        raises=VerifyError,
        match="must not contain `return`",
    ),
)


_OP_CALL_SURFACE: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="unknown-op-name",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl.tf import *
            from tilefoundry.dsl import Tensor

            @func
            def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return totally_undefined_op(x)
        """,
        raises=VerifyError,
        match=r"unknown HIR callable|unknown Op name",
    ),
    ParseErrorCase(
        id="tuple-input-for-a-plain-tensor-param",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl.tf import *
            from tilefoundry.dsl import Tensor

            @func
            def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return relu((x, x))
        """,
        raises=VerifyError,
        match=r"unsupported AST node in expression: Tuple",
    ),
    ParseErrorCase(
        id="integer-literal-keeps-its-own-dtype",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl import Tensor
            from tilefoundry.dsl.tf import *

            @func
            def f(x: Tensor[(1, 8), 'f32']) -> Tensor[(1, 8), 'f32']:
                return mul(x, 2)
        """,
        raises=VerifyError,
        match=r"Binary MUL: dtype mismatch \(f32 vs i64\)",
    ),
    ParseErrorCase(
        id="unknown-dtype-string-at-a-call",
        subject=_BAD_CALL_DTYPE,
        raises=VerifyError,
        match=r"DType: unknown value 'float32'",
    ),
    ParseErrorCase(
        id="unknown-dtype-string-in-an-annotation",
        subject=_BAD_ANNOTATION_DTYPE,
        raises=ValueError,
        match=r"DType: unknown value 'float32'",
    ),
    ParseErrorCase(
        id="unknown-reduce-kind-string",
        subject=_BAD_REDUCE_KIND,
        raises=VerifyError,
        match=r"ReduceKind: unknown value 'plus'",
    ),
)


_DIM_OPERANDS: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="dim-plus-bool",
        subject=lambda: CTX_LEN + True,
        raises=TypeError,
        match=r"dim arithmetic: bool operand True is not a dimension",
    ),
    ParseErrorCase(
        id="dim-plus-object",
        subject=lambda: CTX_LEN + object(),
        raises=TypeError,
        match=r"unsupported operand type\(s\) for \+: 'DimVar' and 'object'",
    ),
    ParseErrorCase(
        id="simplify-dim-bool-on-the-left",
        subject=lambda: simplify_dim(DimAdd, (True, CTX_LEN)),
        raises=TypeError,
        match=r"simplify_dim: bool operand True is not a ShapeDim",
    ),
    ParseErrorCase(
        id="simplify-dim-bool-on-the-right",
        subject=lambda: simplify_dim(DimAdd, (CTX_LEN, False)),
        raises=TypeError,
        match=r"simplify_dim: bool operand False is not a ShapeDim",
    ),
)


_CALL_BOUNDARY: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="nested-call-arg-type-mismatch",
        subject=_arg_type_mismatch,
        raises=VerifyError,
        match=r"arg 0 shape/dtype mismatch",
    ),
    ParseErrorCase(
        id="direct-call-arity-mismatch",
        subject=_direct_call_of_the_wrong_arity,
        raises=VerifyError,
        match="nested @func call arity mismatch",
    ),
    ParseErrorCase(
        id="child-call-of-the-wrong-width",
        subject=_child_call_of_the_wrong_width,
        raises=VerifyError,
        match="takes 1 activation",
    ),
    ParseErrorCase(
        id="reach-a-child-entry-by-name",
        subject=_reach_a_child_entry_by_name,
        raises=VerifyError,
        match=r"'leaf.entry': a Module is called through its bare binding",
    ),
    ParseErrorCase(
        id="reach-a-child-helper-by-name",
        subject=_reach_a_child_helper_by_name,
        raises=VerifyError,
        match=r"'leaf.helper': a Module is called through its bare binding",
    ),
    ParseErrorCase(
        id="reach-a-module-member-by-class",
        subject=_reach_a_module_member_by_class,
        raises=VerifyError,
        match=r"'Callee.entry': a Module is called through its bare binding",
    ),
    ParseErrorCase(
        id="call-a-module-with-no-entry",
        subject=_call_a_module_with_no_entry,
        raises=VerifyError,
        match=r"Module 'NoEntry' declares no entry",
    ),
    ParseErrorCase(
        id="call-a-module-whose-entry-is-a-prim-func",
        subject=_call_a_module_whose_entry_is_a_prim_func,
        raises=VerifyError,
        match="rather than an hir Function",
    ),
    ParseErrorCase(
        id="bare-decorator-leaves-the-child-unattached",
        subject=_bare_decorator_leaves_the_child_unattached,
        raises=VerifyError,
        match=r"'leaf': a Module is called only from a function authored in a @module class body",
    ),
    ParseErrorCase(
        id="declaration-left-open-by-a-failed-class-body",
        subject=_declaration_left_open_by_a_failed_class_body,
        raises=VerifyError,
        match=r"'Callee': a Module is called only from a function authored in a @module class body",
    ),
    ParseErrorCase(
        id="module-call-with-no-binding",
        subject=_module_call_with_no_binding,
        raises=ValueError,
        match=r"@module '_Unattached': call\(s\) to Module\(s\) \['Callee'\] that no class-body binding attaches",
    ),
    ParseErrorCase(
        id="module-call-bound-only-inside-a-list",
        subject=_module_call_bound_only_inside_a_list,
        raises=ValueError,
        match=r"@module '_ListAttached': call\(s\) to Module\(s\) \['Callee'\] that no class-body binding attaches",
    ),
)


_FUNCTION_BINDINGS: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="kernel-binding-underscore",
        subject="""
            from tilefoundry import func, module
            from tilefoundry.dsl import Tensor

            @module()
            class _KernelUnderscore:
                @func
                def _(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                    return x
        """,
        raises=ValueError,
        match=r"@module '_KernelUnderscore': a kernel binding may not be named '_'",
    ),
    ParseErrorCase(
        id="variant-binding-underscore",
        subject="""
            from tilefoundry import func, module
            from tilefoundry.dsl import DimVar, DimVarRangePat, Tensor

            _N = DimVar("N", 1, 8)

            @module()
            class _VariantUnderscore:
                @func
                def dispatch(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
                    pass

                @dispatch.specialize(DimVarRangePat("N", 1, 4))
                def _(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
                    return x
        """,
        raises=ValueError,
        match=(
            r"@module '_VariantUnderscore' base 'dispatch': "
            r"a variant binding may not be named '_'"
        ),
    ),
    ParseErrorCase(
        id="duplicate-kernel-binding",
        subject="""
            from tilefoundry import func, module
            from tilefoundry.dsl import Tensor

            @module()
            class _DuplicateKernel:
                @func
                def run(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                    return x

                @func
                def run(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                    return x
        """,
        raises=ValueError,
        match=r"@module '_DuplicateKernel': duplicate kernel binding 'run'",
    ),
)


_SUBSCRIPTS: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="tile-with-too-many-args",
        subject=hir_source(
            "for i in tile(1, 2, 3):",
            "    y = relu(x)",
        ),
        raises=VerifyError,
        match="tile.. takes 2 arguments",
    ),
    ParseErrorCase(
        id="runtime-slice-stride",
        subject=hir_source(
            "return x[0:8:step, :]",
            signature='x: Tensor[(8, 4), "f32"], step: Tensor[(), "i64"]) -> Tensor[(4, 4), "f32"]',
        ),
        raises=VerifyError,
        match=r"tensor subscript axis 0: slice stride must be a compile-time dimension",
    ),
    ParseErrorCase(
        id="runtime-start-with-an-unrelated-stop",
        subject=hir_source(
            "return x[start:8, :]",
            signature=(
                'x: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]) -> Tensor[(4, 4), "f32"]'
            ),
        ),
        raises=VerifyError,
        match=r"tensor subscript axis 0: a run-time start needs the stop endpoint",
    ),
    ParseErrorCase(
        id="non-divisible-tile-window-at-evaluate-time",
        subject=_evaluate_a_non_divisible_tile_window,
        raises=EvalError,
        match="Slice window exceeds axis 0",
    ),
    ParseErrorCase(
        id="window-moved-by-a-runtime-offset",
        subject=hir_source(
            "o = relu(x[:, 0:2])",
            "for n in tile(4, 2):",
            "    o = relu(x[:, n + k])",
            "return o",
            signature='x: Tensor[(1, 8), "f32"], k: Tensor[(), "i64"]) -> Tensor[(1, 2), "f32"]',
        ),
        raises=VerifyError,
        match="moves by a compile-time integer",
    ),
    ParseErrorCase(
        id="window-reversed-instead-of-moved",
        subject=hir_source(
            "o = relu(x[:, 0:2])",
            "for n in tile(4, 2):",
            "    o = relu(x[:, 4 - n])",
            "return o",
            signature='x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        ),
        raises=VerifyError,
        match="reverses the window",
    ),
    ParseErrorCase(
        id="window-moved-off-the-end",
        subject=hir_source(
            "o = relu(x[:, 0:2])",
            "for n in tile(4, 2):",
            "    o = relu(x[:, n + 5])",
            "return o",
            signature='x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        ),
        raises=VerifyError,
        match=r"reads \[7, 9\).*axis is 8 long",
    ),
    ParseErrorCase(
        id="window-moved-before-the-front",
        subject=hir_source(
            "o = relu(x[:, 0:2])",
            "for n in tile(4, 2):",
            "    o = relu(x[:, n - 2])",
            "return o",
            signature='x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        ),
        raises=VerifyError,
        match="begin before the axis",
    ),
    ParseErrorCase(
        id="subscript-rank-mismatch",
        subject=hir_source(
            "o = relu(x)",
            "for ok in tile(2048, 512):",
            "    o = relu(x[ok])",
            "return o",
            signature='x: Tensor[(1, 2048), "f32"]) -> Tensor[(1, 2048), "f32"]',
        ),
        raises=VerifyError,
        match="rank 1 != tensor rank 2",
    ),
    ParseErrorCase(
        id="runtime-tuple-index",
        subject=hir_source(
            "out = quant(x)",
            "return out[i]",
            signature=(
                'x: Tensor[(1, 1536), "bf16"], i: Tensor[(), "i64"])'
                ' -> Tensor[(1, 1536), "fp8e4m3"]'
            ),
        ),
        raises=VerifyError,
        match="integer constant index",
    ),
    ParseErrorCase(
        id="window-scaled-instead-of-moved",
        subject=hir_source(
            "o = relu(x[:, 0:2])",
            "for n in tile(4, 2):",
            "    o = relu(x[:, n * 2])",
            "return o",
            signature='x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        ),
        raises=VerifyError,
        match="unsupported indexer",
    ),
)


_GRID_LOOPS: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="single-argument-tile",
        subject=hir_source("for i in tile(8):", "    y = relu(x)"),
        raises=VerifyError,
        match=r"use range\(extent\)",
    ),
    ParseErrorCase(
        id="tile-with-a-keyword-step",
        subject=hir_source("for i in tile(8, step=2):", "    y = relu(x)"),
        raises=VerifyError,
        match=r"tile\(\) does not accept keyword args",
    ),
    ParseErrorCase(
        id="range-with-a-keyword-stop",
        subject=hir_source("for i in range(stop=8):", "    y = relu(x)"),
        raises=VerifyError,
        match=r"range\(\) does not accept keyword args",
    ),
    ParseErrorCase(
        id="range-over-a-non-dim-expr",
        subject=hir_source("for i in range(x):", "    y = relu(x)"),
        raises=VerifyError,
        match=r"and extent=x \(Var\) is not one",
    ),
    ParseErrorCase(
        id="range-over-a-float-extent",
        subject=hir_source("for i in range(8.5):", "    y = relu(x)"),
        raises=VerifyError,
        match=r"and extent=float is not one",
    ),
)


_SHARD_SUGAR: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="multi-axis-split-not-divisible",
        subject=_multi_axis_split_not_divisible,
        raises=ValueError,
        match="not divisible",
    ),
    ParseErrorCase(
        id="value-state-not-the-last-outer-item",
        subject=_value_state_not_final,
        raises=ValueError,
        match="last outer item",
    ),
    ParseErrorCase(
        id="value-state-with-a-bare-p",
        subject=_value_state_bare_p,
        raises=ValueError,
        match="reduction argument",
    ),
    ParseErrorCase(
        id="mesh-coordinate-slices-a-placed-tensor",
        subject=_mesh_coordinate_slices_a_placed_tensor,
        raises=VerifyError,
        match="data-dependent mesh ownership",
    ),
    ParseErrorCase(
        id="unresolved-dynamic-split-axis",
        subject=_unresolved_dynamic_split_axis,
        raises=ValueError,
        match=r"split layout dim DimVar\(name='seq_len'.*mesh extent DimVar\(name='other'",
    ),
    ParseErrorCase(
        id="symbolic-mesh-extent",
        subject=_symbolic_mesh_extent,
        raises=ValueError,
        match=r"split layout dim 128 and mesh extent DimVar\(name='W'.*at axis position 0",
    ),
    ParseErrorCase(
        id="bool-layout-extent",
        subject=_bool_split_extent_single_axis,
        raises=ValueError,
        match=r"bool True is not one; bool is an int subclass",
    ),
    ParseErrorCase(
        id="float-layout-extent",
        subject=_float_split_extent_single_axis,
        raises=ValueError,
        match=r"layout dim must be a shape dimension \(int / DimVar / dim-op Expr\), got float",
    ),
)


_TOPOLOGY: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="duplicate-topology-name",
        subject=_duplicate_topology_name,
        raises=VerifyError,
        match="duplicate topology name",
    ),
    ParseErrorCase(
        id="mesh-on-an-undeclared-topology",
        subject=_mesh_on_an_undeclared_topology,
        raises=VerifyError,
        match="topology.*not declared",
    ),
    ParseErrorCase(
        id="mesh-with-a-bare-topology-name",
        subject=_mesh_topology_source('Mesh("cta", layout=(128,))'),
        raises=VerifyError,
        match="tuple of declared topology names",
    ),
    ParseErrorCase(
        id="mesh-with-a-keyword-topology-name",
        subject=_mesh_topology_source('Mesh(topology="cta", layout=(128,))'),
        raises=VerifyError,
        match="tuple of declared topology names",
    ),
)


_TUPLE_UNPACK: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="unpack-a-non-tuple-rhs",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl.tf import *
            from tilefoundry.dsl import Tensor

            @func
            def bad_rhs(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
                p, q = add(a, b)
                return p
        """,
        raises=VerifyError,
        match="tuple unpack requires RHS of TupleType",
    ),
    ParseErrorCase(
        id="unpack-of-the-wrong-arity",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl.tf import *
            from tilefoundry.dsl import Tensor

            @func
            def bad_targets(x: Tensor[(1, 1536), "bf16"]) -> Tensor[(1, 1536), "fp8e4m3"]:
                a, b, c = quant(x)
                return a
        """,
        raises=VerifyError,
        match="tuple unpack arity mismatch",
    ),
    ParseErrorCase(
        id="unpack-into-a-nested-target",
        subject="""
            from tilefoundry import func
            from tilefoundry.dsl.tf import *
            from tilefoundry.dsl import Tensor

            @func
            def bad_targets(x: Tensor[(1, 1536), "bf16"]) -> Tensor[(1, 1536), "fp8e4m3"]:
                (a, b), c = quant(x)
                return a
        """,
        raises=VerifyError,
        match="targets must all be plain names",
    ),
)


_CONSTRAINTS: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="where-with-no-kwargs",
        subject=where_source("    y: where() = tf.add(x, x)\n    return y"),
        raises=VerifyError,
        match=r"where\(\.\.\.\) cannot be empty",
    ),
    ParseErrorCase(
        id="where-with-an-empty-layout",
        subject=where_source("    y: where(layout=()) = tf.add(x, x)\n    return y"),
        raises=VerifyError,
        match="layout constraint cannot be empty",
    ),
    ParseErrorCase(
        id="where-binding-one-topology-twice",
        subject=where_source(
            '    y: where(layout=((_, 16 @ cta), {cta @ P("sum")})) = tf.add(x, x)\n    return y'
        ),
        raises=VerifyError,
        match="layout constraint cannot bind one topology more than once",
    ),
    ParseErrorCase(
        id="where-with-two-binding-sets",
        subject=where_source(
            '    y: where(layout=((_, 16), {cta @ P("sum")}, {cta @ B()})) = tf.add(x, x)\n'
            "    return y"
        ),
        raises=VerifyError,
        match="layout constraint accepts one binding set",
    ),
    ParseErrorCase(
        id="where-with-an-unknown-kwarg",
        subject=where_source('    y: where(partial=P("sum")) = tf.add(x, x)\n    return y'),
        raises=VerifyError,
        match=r"where\(\.\.\.\) has unknown field 'partial'",
    ),
    ParseErrorCase(
        id="where-with-a-non-int-extent",
        subject=where_source("    y: where(layout=(1.5,)) = tf.add(x, x)\n    return y"),
        raises=VerifyError,
        match="layout dimensions must use `_`, an integer, or a symbolic extent",
    ),
    ParseErrorCase(
        id="where-annotated-twice",
        subject=where_source(
            '    y: where(storage="gmem") = tf.add(x, x)\n'
            '    y: where(storage="gmem")\n'
            "    return y"
        ),
        raises=VerifyError,
        match="duplicate where annotation for Expr 'y'",
    ),
    ParseErrorCase(
        id="where-on-a-subscript-lvalue",
        subject=where_source(
            """    value = tf.add(x, x)
    value[0]: where(storage="gmem")
    return value"""
        ),
        raises=VerifyError,
        match="bound plain Name|annotation lvalue",
    ),
    ParseErrorCase(
        id="where-on-a-whole-tuple-binding",
        subject=where_source(
            """    pair = tf.topk(x, k=4, axis=-1)
    pair: where(storage="gmem")
    return x""",
            preamble="",
        ),
        raises=VerifyError,
        match="tensor-valued",
    ),
    ParseErrorCase(
        id="where-layout-extent-name-undefined",
        subject=where_source("    y: where(layout=(_, N @ cta)) = tf.add(x, x)\n    return y"),
        raises=VerifyError,
        match=r"where layout extent 'N' could not be resolved: undefined name 'N'",
    ),
    ParseErrorCase(
        id="where-layout-extent-name-is-a-string",
        subject=where_source(
            "    y: where(layout=(_, N @ cta)) = tf.add(x, x)\n    return y",
            preamble=_WHERE_PRELUDE + '\nN = "not-an-int"',
        ),
        raises=VerifyError,
        match="must resolve to an int or DimVar",
    ),
)


_MMA_SCOPES: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="mma-flat-32-lanes",
        subject=_mma_scope(Topology("thread", 32), Layout(shape=(32,), strides=(1,))),
        raises=VerifyError,
        match=r"shape \(32,\) strides \(1,\).*thread-value decomposition",
    ),
    ParseErrorCase(
        id="mma-wrong-lane-order",
        subject=_mma_scope(Topology("thread", 32), Layout(shape=(4, 8), strides=(8, 1))),
        raises=VerifyError,
        match=r"shape \(4, 8\) strides \(8, 1\).*thread-value decomposition",
    ),
    ParseErrorCase(
        id="mma-cta-not-thread",
        subject=_mma_scope(Topology("cta", 32), Layout(shape=(4, 8), strides=(1, 4))),
        raises=VerifyError,
        match="the scope is a cta scope and the atom needs a thread one",
    ),
    ParseErrorCase(
        id="mma-wrong-lane-count",
        subject=_mma_scope(Topology("thread", 64), Layout(shape=(8, 8), strides=(1, 8))),
        raises=VerifyError,
        match=r"shape \(8, 8\) strides \(1, 8\).*64 lanes and the atom needs 32",
    ),
    ParseErrorCase(
        id="mma-inconsistent-mesh",
        subject=_mma_scope(Topology("thread", 64), Layout(shape=(4, 8), strides=(1, 4))),
        raises=VerifyError,
        match=r"thread\(64\) viewed as shape \(4, 8\).*64 lanes and the atom needs 32",
    ),
    ParseErrorCase(
        id="mma-atom-outside-a-mesh-scope",
        subject=_atom_outside_a_mesh_scope,
        raises=VerifyError,
        match="must be used inside a `with Mesh",
    ),
)


_SYNC_SCOPES: tuple[ParseErrorCase, ...] = (
    ParseErrorCase(
        id="sync-argument-is-not-a-mesh",
        subject=_sync_argument_is_not_a_mesh,
        raises=VerifyError,
        match=r"T.sync expects a Mesh argument \(m or a slice m\[\.\.\.\]\), got Var",
    ),
    ParseErrorCase(
        id="sync-with-no-enclosing-mesh",
        subject=_sync_with_no_enclosing_mesh,
        raises=VerifyError,
        match="no enclosing mesh scope",
    ),
    ParseErrorCase(
        id="sync-on-a-mesh-the-scope-does-not-bind",
        subject=_sync_on_a_mesh_the_scope_does_not_bind,
        raises=VerifyError,
        match="no enclosing scope binds that mesh",
    ),
    ParseErrorCase(
        id="sync-on-a-forged-subbox-exceeding-its-parent",
        subject=_sync_on_a_forged_subbox_exceeding_its_parent,
        raises=VerifyError,
        match=r"T\.sync\(\(thread\(128\)\)\[1, 64\]\): that sub-box is not a slice",
    ),
    ParseErrorCase(
        id="sync-on-a-forged-topology-mismatch",
        subject=_sync_on_a_forged_topology_mismatch,
        raises=VerifyError,
        match=r"T\.sync\(\(warp\(4\)\)\[2, 32\]\): that sub-box is not a slice",
    ),
    ParseErrorCase(
        id="sync-on-a-non-contiguous-slice",
        subject=_sync_on_a_non_contiguous_slice,
        raises=VerifyError,
        match="contiguous",
    ),
    ParseErrorCase(
        id="sync-on-a-cross-warp-unaligned-slice",
        subject=_sync_on_a_cross_warp_unaligned_slice,
        raises=VerifyError,
        match=r"a cross-warp subset must be warp-aligned",
    ),
    ParseErrorCase(
        id="classify-a-partial-cta-slice",
        subject=_classify_a_partial_cta_slice,
        raises=VerifyError,
        match="partial grid",
    ),
    ParseErrorCase(
        id="named-barriers-exhausted",
        subject=_exhaust_the_named_barriers,
        raises=ValueError,
        match="too many distinct named barriers",
    ),
)


ERROR_CASES: tuple[ParseErrorCase, ...] = (
    *_HIR_BODY_STATEMENTS,
    *_OP_CALL_SURFACE,
    *_DIM_OPERANDS,
    *_CALL_BOUNDARY,
    *_FUNCTION_BINDINGS,
    *_SUBSCRIPTS,
    *_GRID_LOOPS,
    *_SHARD_SUGAR,
    *_TOPOLOGY,
    *_TUPLE_UNPACK,
    *_CONSTRAINTS,
    *_MMA_SCOPES,
    *_SYNC_SCOPES,
)
