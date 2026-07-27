"""How every analysis reads the authored program.

An analysis family decides what a number means; none of them should decide what
"the values of this function" or "the bytes of this type" means. Those readings
live here once, so two families cannot disagree about the program they measured.

Nothing here consults a Target. The traversal order, the tensor leaves of a
type, and a type's byte size are properties of the authored IR alone.
"""

from __future__ import annotations

import math

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
from tilefoundry.ir.types.shard import ShardLayout, Topology
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


def execution_domain(type_: Type) -> dict[str, int] | None:
    """The topology extents *type_*'s meshes say its value is spread over.

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
        for topology in tensor.layout.mesh.topologies:
            if not isinstance(topology.size, int) or topology.size <= 0:
                raise AnalysisError(
                    "an execution domain requires positive static topology extents"
                )
            previous = domain.get(topology.name)
            if previous is not None and previous != topology.size:
                raise AnalysisError(
                    f"one Mesh declares conflicting {topology.name!r} extents "
                    f"{previous} and {topology.size}"
                )
            domain[topology.name] = topology.size
        domains.add(tuple(sorted(domain.items())))
    if len(domains) > 1:
        raise AnalysisError(
            f"one value references conflicting execution domains {sorted(domains)}"
        )
    return dict(next(iter(domains))) if domains else None


def execution_count(
    call: Call, fn: Function, topologies: tuple[Topology, ...]
) -> int:
    """How many times *call* runs, as the product of its topology extents.

    The output's own domain wins when it has one, because the output is what
    the call produced; only when it carries no mesh do the inputs decide. The
    function's declared topologies are then folded in, and a value that
    contradicts the declaration is an error rather than an override.
    """
    domain = execution_domain(call.type)
    inputs = {
        tuple(sorted(value.items()))
        for arg in call.args
        if (value := execution_domain(arg.type)) is not None
    }
    if domain is None and len(inputs) > 1:
        raise AnalysisError(
            f"{describe(call)}: inputs reference conflicting execution domains "
            f"{sorted(inputs)}"
        )
    if domain is None:
        domain = dict(next(iter(inputs))) if inputs else {}
    for topology in topologies:
        if not isinstance(topology.size, int) or topology.size <= 0:
            raise AnalysisError(
                f"function {fn.name!r}: an execution count requires positive "
                "static topology extents"
            )
        previous = domain.get(topology.name)
        if previous is not None and previous != topology.size:
            raise AnalysisError(
                f"{describe(call)}: value Mesh declares {topology.name}={previous}, "
                f"but function {fn.name!r} declares {topology.name}={topology.size}"
            )
        domain[topology.name] = topology.size
    return math.prod(domain.values())


def topology_extent(type_: Type, name: str) -> int | None:
    """The extent *type_*'s meshes declare for the topology called *name*."""
    extents: set[int] = set()
    for tensor in tensor_types(type_):
        if not isinstance(tensor.layout, ShardLayout):
            continue
        for topology in tensor.layout.mesh.topologies:
            if topology.name != name:
                continue
            if not isinstance(topology.size, int) or topology.size <= 0:
                raise AnalysisError(
                    f"a {name!r} extent must be a positive static integer"
                )
            extents.add(topology.size)
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
    "entry_function",
    "execution_count",
    "execution_domain",
    "postorder",
    "reachable_functions",
    "tensor_types",
    "topology_extent",
    "values_of",
]
