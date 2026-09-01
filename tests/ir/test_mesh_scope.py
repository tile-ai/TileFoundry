"""The execution domain as a region: what it composes, and what it refuses."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.mesh_scope import MeshScope
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh, Topology
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
    assert ctx.mesh_scope == ()


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


def test_two_regions_on_one_level_are_refused() -> None:
    """One level is divided once, so nesting a second scope on it says nothing.

    It is refused where the second region is entered, whether the level it
    repeats is the one just above it or one further out.
    """
    adjacent, _ = _scoped(CTA, _mesh("cta", 2, "c2"))
    with pytest.raises(ValueError, match="both name topology 'cta'"):
        _ScopeAtLeaf().visit(adjacent, TypeInferContext())

    distant, _ = _scoped(CTA, WARP, _mesh("cta", 2, "c2"))
    with pytest.raises(ValueError, match="both name topology 'cta'"):
        _ScopeAtLeaf().visit(distant, TypeInferContext())
