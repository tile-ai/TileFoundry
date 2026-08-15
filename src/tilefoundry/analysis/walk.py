"""Provide target-independent readings of authored IR for every analysis.

Traversal order, function values, tensor leaves, and type byte sizes live here
so analysis families measure the same program consistently.
"""

from __future__ import annotations

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    IRMetadata,
    SourceSpanMetadata,
    Tuple,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.module import owning_module as _owning_module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.shard import Mesh, shard_layout_of
from tilefoundry.ir.types.shard.layout_algebra import size
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprWalker, _expr_children

from .errors import AnalysisError


def children(expr: Expr) -> tuple[Expr, ...]:
    """The operand expressions of *expr*, in authored order."""
    if isinstance(expr, (Function, ShapeOf)):
        return ()
    return _expr_children(expr)


def enclosing_trips(root: Expr | None) -> dict[int, int]:
    """Return production counts for values repeated by enclosing loops.

    A value repeats only when it depends on a loop induction variable or carried
    argument; invariant values reached by a body remain single computations.
    Counts multiply through true nesting and are keyed by object identity for
    the SSA DAG. Values absent from the returned map have a count of one.
    """
    if root is None:
        return {}
    trips: dict[int, int] = {}
    for loop in postorder(root):
        if not isinstance(loop, GridRegionExpr):
            continue
        count = loop_trip_count(loop)
        if count == 1:
            continue
        for expr_id in loop_repeated_values(loop):
            trips[expr_id] = trips.get(expr_id, 1) * count
    return trips


def loop_repeated_values(loop: GridRegionExpr) -> set[int]:
    """Return ids of values *loop* recomputes on every trip.

    Values depending on the induction variable or a carried argument change each
    trip; all others remain invariant. Walk the body and every yielded value in
    definition order so dependencies are known before consumers and every carried
    chain contributes its repeated work.
    """
    seeds = {id(loop.induction_var), *(id(carried) for carried in loop.carried_args)}
    repeated: set[int] = set()
    for root in (loop.body, *loop.yield_values):
        for expr in postorder(root):
            if id(expr) in seeds or any(
                id(child) in repeated or id(child) in seeds for child in children(expr)
            ):
                repeated.add(id(expr))
    return repeated


def loop_trip_count(loop: GridRegionExpr) -> int:
    """How many times *loop* runs, or one when its bounds are not numbers.

    One rather than a guess: an extent still carrying a range has no trip count, and
    inventing one would report a cost for a program nobody asked about. Asking about
    a stated size is what resolves it.
    """
    start, extent, step = loop.start, loop.extent, loop.step
    if not all(isinstance(value, int) for value in (start, extent, step)):
        return 1
    if step <= 0 or extent <= start:
        return 1
    return -(-(extent - start) // step)


def postorder(root: Expr | None) -> tuple[Expr, ...]:
    """Every value reachable from *root*, operands before their consumer.

    The body is an SSA DAG rather than a tree, so a value shared by two
    consumers is visited once. That is what makes this order usable as a
    definition order: a value appears exactly where it is defined.
    """
    if root is None:
        return ()
    result: list[Expr] = []

    class _Postorder(ExprWalker[None]):
        def _record(self, expr: Expr) -> None:
            result.append(expr)

        def visit_Call(self, expr: Call) -> None:
            self.visit_operands(expr)
            self._record(expr)

        def visit_Var(self, expr: Var) -> None:
            self._record(expr)

        def visit_Constant(self, expr: Constant) -> None:
            self._record(expr)

        def visit_Tuple(self, expr: Tuple) -> None:
            self.visit_operands(expr)
            self._record(expr)

        def visit_GridRegionExpr(self, expr: GridRegionExpr) -> None:
            self.visit_operands(expr)
            self._record(expr)

        def visit_Function(self, expr: Function) -> None:
            self.visit_operands(expr)
            self._record(expr)

        def visit_SymbolRef(self, expr) -> None:
            self._record(expr)

        def visit_ShapeOf(self, expr) -> None:
            self._record(expr)

    if root is not None:
        _Postorder().visit(root)
    return tuple(result)


def values_of(fn: Function) -> tuple[Expr, ...]:
    """Every expression an analysis may annotate on *fn*.

    The Function object itself is included, because a whole-function record has
    nowhere else to live: it is not a property of any single value in the body.
    """
    return (fn, *fn.params, *postorder(fn.body))


def reachable_functions(root: Function) -> tuple[Function, ...]:
    """*root* and every Function it calls, callers before callees.

    Callee-last order is what lets a caller's totals be assembled from results
    already computed for its callees, by walking the tuple in reverse.
    """
    result: list[Function] = []
    seen: set[int] = set()

    def visit(fn: Function) -> None:
        if id(fn) in seen:
            return
        seen.add(id(fn))
        result.append(fn)
        for expr in postorder(fn.body):
            if isinstance(expr, Call) and isinstance(expr.target, Function):
                visit(expr.target)

    visit(root)
    return tuple(result)


def called_functions(fn: Function) -> tuple[Function, ...]:
    """Every Function *fn* calls directly, in the order its body reaches them."""
    found: list[Function] = []
    for expr in postorder(fn.body):
        if isinstance(expr, Call) and isinstance(expr.target, Function):
            found.append(expr.target)
    return tuple(found)


def owning_module(root: Module, fn: Function) -> Module:
    """The one node of *root*'s tree that owns *fn*.

    Asked within a supplied tree, by identity and recorded origin rather than
    by a name a copy keeps. No owner, or more than one, is refused rather than
    assigned to the root.
    """
    owner = _owning_module(root, fn)
    if owner is None:
        raise AnalysisError(
            f"function {fn.name!r} is owned by no single node of module "
            f"{root.name!r}; analysis answers ownership within the tree it was "
            f"given, never by name"
        )
    return owner


def entry_function(module: Module) -> Function:
    """The HIR Function *module* is entered through."""
    entry = module.entry_function()
    if not isinstance(entry, Function):
        raise AnalysisError(
            f"module {module.name!r}: analysis accepts HIR functions only, and the "
            f"entry is a {type(entry).__name__}"
        )
    return entry


def tensor_types(type_: Type) -> tuple[TensorType, ...]:
    """The tensor leaves of *type_*, flattened out of any tuple nesting."""
    match type_:
        case TensorType():
            return (type_,)
        case TupleType(fields=fields):
            return tuple(field for item in fields for field in tensor_types(item))
        case _:
            return ()


def bytes_by_storage(
    type_: Type, *, umat_level: str | None = None
) -> dict[str, int]:
    """Logical bytes *type_* occupies, per storage level name.

    An unmaterialized leaf occupies nothing by default: it has no committed
    residency yet, so charging it to a level would report capacity for a
    placement no one has chosen. A consuming analysis may provide the level at
    which such a leaf is materialized for that operation.
    """
    result: dict[str, int] = {}
    for tensor in tensor_types(type_):
        if tensor.storage is StorageKind.UMAT:
            if umat_level is None:
                continue
            name = umat_level
        else:
            name = str(tensor.storage)
        result[name] = result.get(name, 0) + tensor_bytes(tensor)
    return result


def _mesh_position_count(mesh: Mesh) -> tuple[str, int]:
    """The one topology name and logical position count a mesh can state here."""
    names = tuple(topology.name for topology in mesh.topologies)
    if len(names) != 1:
        raise AnalysisError(
            "a per-topology position count requires one Mesh topology, got "
            f"{names}"
        )
    count = size(mesh.layout)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise AnalysisError("a mesh position count requires a positive static layout size")
    return names[0], count


def execution_domain(type_: Type) -> dict[str, int] | None:
    """The logical positions *type_*'s meshes say its value is spread over.

    ``None`` means the type carries no mesh at all, which is different from an
    empty domain: nothing was declared, rather than one instance declared.
    Conflicting extents for one topology name are rejected instead of
    reconciled, because there is no reading under which both are true.
    """
    domains: set[tuple[tuple[str, int], ...]] = set()
    for tensor in tensor_types(type_):
        layout = shard_layout_of(tensor.layout)
        if layout is None:
            continue
        domain: dict[str, int] = {}
        name, count = _mesh_position_count(layout.mesh)
        previous = domain.get(name)
        if previous is not None and previous != count:
            raise AnalysisError(
                f"one Mesh declares conflicting {name!r} extents {previous} and {count}"
            )
        domain[name] = count
        domains.add(tuple(sorted(domain.items())))
    if len(domains) > 1:
        raise AnalysisError(
            f"one value references conflicting execution domains {sorted(domains)}"
        )
    return dict(next(iter(domains))) if domains else None


def topology_extent(type_: Type, name: str) -> int | None:
    """The logical extent *type_*'s meshes state for topology *name*."""
    extents: set[int] = set()
    for tensor in tensor_types(type_):
        layout = shard_layout_of(tensor.layout)
        if layout is None:
            continue
        mesh_name, count = _mesh_position_count(layout.mesh)
        if mesh_name == name:
            extents.add(count)
    if len(extents) > 1:
        raise AnalysisError(
            f"one value references conflicting {name!r} extents {sorted(extents)}"
        )
    return next(iter(extents), None)


def describe(expr: Expr) -> str:
    """One diagnostic line locating *expr* in the authored source."""
    binding = binding_name(expr)
    span = get_metadata(expr, SourceSpanMetadata)
    prefix = f"{span.file}:{span.line}:{span.column}: " if span is not None else ""
    op = type(expr.target).__name__ if isinstance(expr, Call) else type(expr).__name__
    return f"{prefix}binding={binding or '<unnamed>'} op={op}"


def attach(expr: Expr, value: IRMetadata) -> None:
    """Attach *value* to *expr*, replacing any record of the same type.

    The update is in place because an analysis annotates the program the caller
    holds. Rebuilding the expression would hand back a copy and leave the
    caller's IR unmeasured.
    """
    kept = tuple(item for item in expr.metadata if type(item) is not type(value))
    object.__setattr__(expr, "metadata", (*kept, value))


def detach(expr: Expr, metadata_type: type[IRMetadata]) -> None:
    """Remove any *metadata_type* record from *expr*, in place."""
    kept = tuple(item for item in expr.metadata if type(item) is not metadata_type)
    object.__setattr__(expr, "metadata", kept)


__all__ = [
    "attach",
    "bytes_by_storage",
    "called_functions",
    "children",
    "describe",
    "detach",
    "enclosing_trips",
    "entry_function",
    "execution_domain",
    "loop_repeated_values",
    "loop_trip_count",
    "owning_module",
    "postorder",
    "reachable_functions",
    "tensor_types",
    "topology_extent",
    "values_of",
]
