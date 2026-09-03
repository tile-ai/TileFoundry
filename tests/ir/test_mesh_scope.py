"""The execution domain as a region: what it composes, and what it refuses."""

from __future__ import annotations

import pytest

from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.mesh_scope import MeshScope
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.hir.specialize import is_concrete
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.ir.types.shard.mesh import Mesh, Topology, check_topology, composed
from tilefoundry.ir.types.shard.scope_match import covered_by_scope, storage_reaches
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprCloner, collect_exprs, expr_children
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_TY = TensorType(shape=(8,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)


def _mesh(level: str, size: int, name: str) -> Mesh:
    return Mesh((Topology(level, size),), Layout((size,), (1,)), (name,))


CTA, WARP, THREAD = _mesh("cta", 2, "c"), _mesh("warp", 8, "w"), _mesh("thread", 4, "t")


def _scoped(*meshes: Mesh) -> tuple[MeshScope, Var]:
    """A stack of regions, outermost first, around one variable."""
    leaf = Var(name="x", type=_TY)
    expr = leaf
    for mesh in reversed(meshes):
        expr = MeshScope(mesh=mesh, body=expr, type=_TY)
    return expr, leaf


class _ScopeAtLeaf(TypeInferVisitor):
    """Report the scope in force where the walk bottoms out."""

    def __init__(self) -> None:
        super().__init__()
        self.at_leaf: object = None

    def visit_leaf_Var(self, var: Var, _operands, ctx: TypeInferContext):
        self.at_leaf = ctx.mesh_scope
        return var.type


def test_a_region_is_an_expression_the_generic_walks_already_know() -> None:
    """Not a node every traversal has to be taught about one at a time.

    A rewrite that does not mention regions must still return an equal tree
    rather than dropping the region or refusing to walk it.
    """
    outer, leaf = _scoped(CTA, THREAD)

    (inner,) = expr_children(outer)
    assert isinstance(inner, MeshScope) and inner.mesh == THREAD
    assert leaf in collect_exprs(outer)

    rebuilt = ExprCloner().visit(outer, None)
    assert rebuilt is outer


class _SwapLeaf(ExprCloner):
    """Rewrite the one variable, so every region above it must be rebuilt."""

    def __init__(self, replacement: Var) -> None:
        super().__init__()
        self.replacement = replacement

    def visit_leaf_Var(self, _var: Var, _operands, _ctx) -> Var:
        return self.replacement


def test_rewriting_what_a_region_holds_rebuilds_the_region_around_it() -> None:
    """A region carries its body, so replacing the body replaces the regions.

    What the regions themselves say is not the rewrite's business: the mesh each
    one names comes through untouched, and so does the type, which is the body's
    -- who runs the work says nothing about the shape of what it produced.
    """
    outer, _leaf = _scoped(CTA, THREAD)
    fresh = Var(name="y", type=_TY)

    rebuilt = _SwapLeaf(fresh).visit(outer, None)

    assert rebuilt is not outer
    assert isinstance(rebuilt, MeshScope) and rebuilt.mesh == CTA
    inner = rebuilt.body
    assert isinstance(inner, MeshScope) and inner.mesh == THREAD
    assert inner.body is fresh
    assert rebuilt.type == rebuilt.body.type == inner.body.type


def test_the_scope_inside_a_region_is_the_whole_nesting() -> None:
    """A lane is a lane of a CTA, so the scope in force is the pair.

    The composed mesh is what a mesh naming both levels itself would say, and
    the caller's own scope survives the recursion unchanged.
    """
    outer, _leaf = _scoped(CTA, THREAD)
    ctx = TypeInferContext()

    walk = _ScopeAtLeaf()
    walk.visit(outer, ctx)

    assert [t.name for t in walk.at_leaf.topologies] == ["cta", "thread"]
    assert walk.at_leaf.layout == Layout(shape=(2, 4), strides=(4, 1))
    assert ctx.mesh_scope is None


def test_a_region_used_as_call_argument_keeps_its_own_scope() -> None:
    """Call operands are values and must not re-compose their execution region."""
    outer = composed((CTA, THREAD))
    value = Var(name="x", type=_TY)
    argument_region = MeshScope(mesh=CTA, body=value, type=_TY)
    call = Call(type=_TY, target=ReLU(), args=(argument_region,))

    assert TypeInferVisitor().visit(call, TypeInferContext(mesh_scope=outer)) == _TY


def test_call_operand_subgraphs_keep_visibility_checks() -> None:
    """A region-valued sibling must not hide another operand from checking."""
    fine_type = TensorType(
        shape=(8,),
        dtype=DType.f32,
        layout=ShardLayout(Layout((8,), (1,)), (Split(0),), THREAD),
        storage=StorageKind.RMEM,
    )
    fine = Var(name="fine", type=fine_type)
    region = MeshScope(mesh=CTA, body=Var(name="value", type=_TY), type=_TY)
    call = Call(
        type=_TY,
        target=Binary(kind=BinaryKind.ADD),
        args=(region, fine),
    )

    with pytest.raises(VerifyError, match="input 1 is laid out more finely"):
        TypeInferVisitor().visit(call, TypeInferContext(mesh_scope=CTA))


def test_scope_coverage_compares_level_positions_without_an_order_relation() -> None:
    """An inner-only layout is covered by a scope that also names an outer level."""
    unit_cta = _mesh("cta", 1, "c")
    current = composed((unit_cta, THREAD))
    implicit_c_order = Mesh(THREAD.topologies, Layout((4,)), THREAD.names)
    different_positions = Mesh(THREAD.topologies, Layout((4,), (2,)), THREAD.names)

    assert covered_by_scope(THREAD, current)
    assert covered_by_scope(implicit_c_order, current)
    assert covered_by_scope(THREAD[1:3], THREAD[1:3])
    assert not covered_by_scope(THREAD, unit_cta)
    assert not covered_by_scope(different_positions, current)
    assert storage_reaches(StorageKind.RMEM, THREAD, current)
    assert not storage_reaches(StorageKind.RMEM, unit_cta, current)
    assert storage_reaches(StorageKind.SMEM, unit_cta, current)
    assert storage_reaches(StorageKind.GMEM, unit_cta, current)
    assert not storage_reaches(StorageKind.HOST, unit_cta, current)


def test_symbolic_dimension_visitor_walks_hir_mesh_scope() -> None:
    """Concrete-dimension analysis descends through HIR scope regions."""
    param = Var(name="x", type=_TY)
    scope = MeshScope(mesh=CTA, body=param, type=_TY)
    fn = Function.build(name="scoped", params=(param,), body=scope, return_type=_TY)
    assert is_concrete(fn)


def test_evaluator_walks_hir_mesh_scope() -> None:
    """The compile-time mesh descriptor does not alter body evaluation."""
    scalar = TensorType(shape=(), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
    value = Constant(type=scalar, value=3.0)
    scope = MeshScope(mesh=CTA, body=value, type=scalar)
    fn = Function.build(name="scoped", params=(), body=scope, return_type=scalar)
    assert evaluate(fn).item() == 3.0


def test_a_third_region_composes_onto_the_two_above_it() -> None:
    """Entering a scope folds it onto a chain, not onto one level of it.

    The chain reaching a third region already names two levels, so composing
    has to accept what it produced itself.
    """
    outer, _leaf = _scoped(CTA, WARP, THREAD)

    walk = _ScopeAtLeaf()
    walk.visit(outer, TypeInferContext())

    assert [t.name for t in walk.at_leaf.topologies] == ["cta", "warp", "thread"]
    assert walk.at_leaf.layout == Layout(shape=(2, 8, 4), strides=(32, 4, 1))


def test_an_inner_region_replaces_all_levels_in_force_or_rejects_partial_overlap() -> None:
    """Inlining may put a callee's reading of one level inside the caller's."""
    inner = Mesh(CTA.topologies, Layout((1, 2), (2, 1)), ("row", "col"))
    nested, _ = _scoped(CTA, inner)

    walk = _ScopeAtLeaf()
    walk.visit(nested, TypeInferContext())

    assert walk.at_leaf == inner

    distant, _ = _scoped(CTA, WARP, inner)
    with pytest.raises(
        ValueError,
        match=(
            r"\['cta'\] named again while \['warp'\] is not; a scope either "
            r"replaces the levels in force or adds levels below them"
        ),
    ):
        walk.visit(distant, TypeInferContext())


def test_check_topology_rejects_one_scope_with_too_many_positions() -> None:
    """A scope cannot address more units than its declared level provides."""
    oversized = Mesh((Topology("cta", 128),), Layout((256,), (1,)), ("cta",))

    with pytest.raises(
        ValueError,
        match="mesh level 'cta' has 256 positions, exceeding declared extent 128",
    ):
        check_topology(oversized)


def test_composing_slices_scales_strides_and_offsets_by_the_level_below() -> None:
    """A sliced outer level keeps its coordinates when a finer level is added."""
    cta = Mesh((Topology("cta", 8),), Layout((8,), (1,)), ("cta",))
    thread = Mesh((Topology("thread", 128),), Layout((128,), (1,)), ("thread",))

    lower_half = composed((cta[0:4], thread))
    assert lower_half.layout == ComposedLayout(
        inner=None,
        offset=0,
        outer=Layout(shape=(4, 128), strides=(128, 1)),
    )

    upper_half = composed((cta[4:8], thread))
    assert upper_half.layout == ComposedLayout(
        inner=None,
        offset=512,
        outer=Layout(shape=(4, 128), strides=(128, 1)),
    )

    opposite_halves = composed((cta[4:8], thread[64:128]))
    assert opposite_halves.layout == ComposedLayout(
        inner=None,
        offset=576,
        outer=Layout(shape=(4, 64), strides=(128, 1)),
    )
