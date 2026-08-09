"""How every analysis reads the authored program.

An analysis family decides what a number means; none of them should decide what
"the values of this function" or "the bytes of this type" means. Those readings
live here once, so two families cannot disagree about the program they measured.

Nothing here consults a Target. The traversal order, the tensor leaves of a
type, and a type's byte size are properties of the authored IR alone.
"""

from __future__ import annotations

from tilefoundry.ir.core import (
    Call,
    Expr,
    IRMetadata,
    SourceSpanMetadata,
    Tuple,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.shard import Mesh, ShardLayout
from tilefoundry.ir.types.shard.layout_algebra import size
from tilefoundry.ir.types.storage import StorageKind

from .errors import AnalysisError


def children(expr: Expr) -> tuple[Expr, ...]:
    """The operand expressions of *expr*, in authored order."""
    match expr:
        case Call(args=args):
            return args
        case Tuple(elements=elements):
            return elements
        case GridRegionExpr(init_args=init_args, body=body, yield_values=yield_values):
            return (*init_args, body, *yield_values)
        case _:
            return ()


def enclosing_trips(root: Expr | None) -> dict[int, int]:
    """How many times each value under *root* is produced, by loop nesting.

    A `GridRegionExpr` is one Expr standing for a loop, so a walk of the body visits
    each of its calls once while the program runs them once per trip. Charging them
    once is charging a tiled kernel for one tile: a K-loop over twenty-four blocks
    and a column loop over a hundred and forty come back as a kernel doing a
    two-thousandth of its arithmetic, which is not a small error in a compiler whose
    subject is tiling.

    Which values a loop repeats is not "everything the body reaches". The body reads
    values defined before the loop -- a blocked weight, a normalised input -- and
    those are computed once however many trips run. What repeats is what changes
    between trips, and the IR says which those are: a value repeats exactly when it
    depends on the loop's induction variable or on one of its carried arguments.

    Getting that wrong is not a small overcount. Two loops in sequence are nested by
    data dependence -- the second body reads the first loop's result -- so charging
    everything the body reaches would multiply the first loop's arithmetic by the
    second loop's trips as well, and a tiled MLP comes back thirty-four times its own
    cost instead of a thousandth of it.

    Keyed by `id`, like the rest of this module, because the body is an SSA DAG and a
    value is the object that defines it. A value no loop repeats maps to one; the map
    holds only what repeats, so a caller uses `.get(id(expr), 1)`.
    """
    if root is None:
        return {}
    trips: dict[int, int] = {}
    for loop in postorder(root):
        if not isinstance(loop, GridRegionExpr):
            continue
        count = _trip_count(loop)
        if count == 1:
            continue
        for expr_id in _repeated_by(loop):
            trips[expr_id] = trips.get(expr_id, 1) * count
    return trips


def _repeated_by(loop: GridRegionExpr) -> set[int]:
    """The ids of the values *loop* recomputes on every trip.

    A value depending on the induction variable or on a carried argument is a
    different value each trip; one depending on neither is the same value every
    trip, whatever else it is an operand of. Walked in definition order so a
    dependence is known before its consumer is asked about.

    Every yielded value is walked as well as the body, because a loop carrying two
    accumulators keeps one chain under `body` and the other under `yield_values` --
    walking only the body finds one of the two matmuls a tiled MLP performs and
    charges the other once.
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


def _trip_count(loop: GridRegionExpr) -> int:
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
    seen: set[int] = set()
    result: list[Expr] = []

    def visit(expr: Expr) -> None:
        if id(expr) in seen:
            return
        seen.add(id(expr))
        for child in children(expr):
            visit(child)
        result.append(expr)

    visit(root)
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


def bytes_by_storage(type_: Type) -> dict[str, int]:
    """Logical bytes *type_* occupies, per storage level name.

    An unmaterialized leaf occupies nothing: it has no committed residency yet,
    so charging it to a level would report capacity for a placement no one has
    chosen.
    """
    result: dict[str, int] = {}
    for tensor in tensor_types(type_):
        if tensor.storage is StorageKind.UMAT or tensor.storage is None:
            continue
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
        if not isinstance(tensor.layout, ShardLayout):
            continue
        domain: dict[str, int] = {}
        name, count = _mesh_position_count(tensor.layout.mesh)
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
        if not isinstance(tensor.layout, ShardLayout):
            continue
        mesh_name, count = _mesh_position_count(tensor.layout.mesh)
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
    "children",
    "describe",
    "detach",
    "enclosing_trips",
    "entry_function",
    "execution_domain",
    "postorder",
    "reachable_functions",
    "tensor_types",
    "topology_extent",
    "values_of",
]
