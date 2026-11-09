"""The public Analyze operation.

One call selects one or more root analyses. Analyze resolves their transitive
dependency closure, orders it, and runs each member exactly once, so an
analysis that several roots depend on is computed once rather than per root.

The result is semantic. Human text, JSON, and annotated HIR are renderings of
it and of the Metadata left on the IR, not fields of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tilefoundry import dump, services, target
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr

from .check import check_analyses
from .errors import Error
from .iteration_scope import ScopeBuilder
from .metadata import ExecutionMeshMetadata
from .visitor import AnalyzeContext


@dataclass(frozen=True)
class Result:
    """Describe what one public analysis call computed.

    ``executed`` records dependency order and ``metadata_types`` lists records
    actually written, excluding analyses that matched nothing. ``function`` is
    the checked root that carries aggregate records; ``module`` remains the call's
    execution domain.
    """

    module: core.Module
    function: core.Function
    analyses: tuple[str, ...]
    topology_level: str | None
    executed: tuple[str, ...]
    metadata_types: tuple[type[expr.IRMetadata], ...]
    functions: tuple[core.Function, ...]


def select_analyzer(tgt: target.Target, selector: str, *, root: str) -> target.Analyzer:
    """The service selected by the resolved Target for *selector*."""
    try:
        return tgt.get_analyzer(selector)
    except target.UnsupportedCapabilityError as error:
        if selector == root:
            raise Error(str(error)) from None
        raise Error(f"{root!r} depends on {selector!r}, which is not registered: {error}") from None


def normalize_roots(analysis: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one or more root selectors, preserving their first occurrence."""
    requested = (analysis,) if isinstance(analysis, str) else tuple(analysis)
    if not requested:
        raise Error("analyze: analysis must name at least one root")
    if any(not isinstance(item, str) or not item for item in requested):
        raise Error(f"analyze: every analysis must be a non-empty selector, got {analysis!r}")
    return tuple(dict.fromkeys(requested))


def analyzer_closure(tgt: target.Target, roots: tuple[str, ...]) -> tuple[target.Analyzer, ...]:
    """*roots* and everything they transitively need, dependencies first.

    The walk is depth-first over declared names, which both orders the closure
    and detects a cycle: meeting a selector that is still being visited means
    it depends on itself through some path.
    """
    ordered: list[target.Analyzer] = []
    done: set[str] = set()
    visiting: list[str] = []

    def visit(selector: str, root: str) -> None:
        if selector in done:
            return
        if selector in visiting:
            cycle = " -> ".join([*visiting[visiting.index(selector) :], selector])
            raise Error(f"analysis dependency cycle: {cycle}")
        visiting.append(selector)
        algorithm = select_analyzer(tgt, selector, root=root)
        for required in algorithm.requires:
            visit(required, root)
        visiting.pop()
        done.add(selector)
        ordered.append(algorithm)

    for root in roots:
        visit(root, root)
    return tuple(ordered)


def check_subject(module: core.Module, function: core.Function) -> None:
    """Require an HIR definition owned by the supplied checked program."""
    if not isinstance(module, core.Module):
        raise TypeError(
            f"analyze: expected a Module, got {type(module).__name__}. A Function "
            "carries no execution context; select the Module that owns it."
        )
    if not isinstance(function, core.Function):
        raise TypeError(f"analyze: expected an hir.Function, got {type(function).__name__}")
    if not module.owns(function, derived=True):
        raise Error(f"analyze: {function.name!r} is not a function of module {module.name!r}")


def build_context(
    module: core.Module,
    function: core.Function,
    *,
    topology_level: str | None = None,
    options: object | None = None,
) -> tuple[AnalyzeContext, ...]:
    """Build one local scope tree per definition in a previously checked program."""
    check_subject(module, function)
    topologies = module.effective_topologies()
    if topology_level is None:
        topology_level = topologies[0].name if topologies else None

    contexts, definitions = [], {}
    frames = tuple(valuation.definitions(valuation.Context(module, function), postorder=True))
    for frame in frames:
        context = AnalyzeContext(
            module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies,
            owner=frame.function,
            topology_level=topology_level,
            options=options,
            entry=function,
            definitions=definitions,
        )
        builder = ScopeBuilder(
            frame.module,
            frame.function,
            ctx=context,
            is_root=frame.function is function,
        )
        builder.build()
        definitions[id(frame.function)] = context.scope
        contexts.append(context)
    return tuple(contexts)


def analyze(
    module: core.Module,
    function: core.Function,
    *,
    analysis: str | Sequence[str],
    topology_level: str | None = None,
    level: str | None = None,
    options: object | None = None,
) -> Result:
    """Run the requested analyses' union dependency closure over *function*.

    The caller supplies the program returned by ``check_program``. Analyses
    write records onto that same program; dimensions are bound by the caller.
    The selected topology defaults to the module's coarsest level.
    """
    if topology_level is not None and level is not None and topology_level != level:
        raise ValueError("analyze: topology_level and level disagree")
    topology_level = topology_level if topology_level is not None else level
    check_subject(module, function)
    if expr.get_metadata(function, ExecutionMeshMetadata) is None:
        raise Error("analyze needs the program returned by check_program")
    roots = normalize_roots(analysis)
    result_module = module
    closure = analyzer_closure(module.resolve_target(), roots)
    if topology_level is None:
        topologies = module.effective_topologies()
        topology_level = topologies[0].name if topologies else None
    check_analyses(module, function, closure, topology_level=topology_level)
    contexts = build_context(
        module,
        function,
        topology_level=topology_level,
        options=options,
    )
    context = contexts[-1]
    functions = tuple(frame.function for frame in valuation.definitions(context))

    order: list[type[expr.IRMetadata]] = []
    written_records: set[tuple] = set()
    for algorithm in closure:
        before = metadata_snapshot(functions)
        try:
            for local in contexts:
                algorithm.run(local.function, local)
        except target.UnsupportedCapabilityError as error:
            raise Error(f"{algorithm.selector}: {error}") from None
        after = metadata_snapshot(functions)
        records = require_owned_writes(algorithm, before, after)
        written_records |= records
        written = {key[-1] for key in records}
        for metadata_type in algorithm.produces:
            if metadata_type in written and metadata_type not in order:
                order.append(metadata_type)

    final = metadata_snapshot(functions)
    surviving = {
        metadata_type for key in written_records & final.keys() for metadata_type in (key[-1],)
    }

    result = Result(
        module=result_module,
        function=function,
        analyses=roots,
        topology_level=topology_level,
        executed=tuple(algorithm.selector for algorithm in closure),
        metadata_types=tuple(item for item in order if item in surviving),
        functions=functions,
    )
    dump.dump("analysis.json", lambda: services.report_json(result), dump.Flags.ANALYSIS)
    return result


def metadata_snapshot(
    functions: tuple[core.Function, ...],
) -> dict[tuple[int, type], int]:
    """Which Metadata object sits on which expression, by identity.

    Keying on the object's identity rather than its value is what lets the
    caller tell a replacement from an untouched entry: an algorithm that
    rewrites another's Metadata to an equal value has still overwritten it.

    The Function objects are part of the walk, not just their bodies: a
    whole-function record hangs on the Function itself, and a snapshot that
    skipped it would leave those records outside ownership entirely.
    """
    snapshot: dict[tuple[int, type], int] = {}
    for fn in functions:
        for node in core.function_values(fn):
            for item in node.metadata:
                snapshot[(id(node), type(item))] = id(item)
    return snapshot


def metadata_delta(
    before: dict[tuple[int, type], int],
    after: dict[tuple[int, type], int],
) -> tuple[set[tuple[int, type]], set[tuple[int, type]]]:
    """What this step touched, as ``(changed, written)`` record keys.

    ``changed`` spans the union of both snapshots, because removing a record
    changes the IR just as much as overwriting it and would otherwise be
    invisible: the key is simply absent afterwards. ``written`` keeps only the
    keys that still carry a record, so a record the step deleted is not counted
    as something a reader can go and find.
    """
    changed: set[tuple[int, type]] = set()
    written: set[tuple[int, type]] = set()
    for key in before.keys() | after.keys():
        if before.get(key) == after.get(key):
            continue
        changed.add(key)
        if key in after:
            written.add(key)
    return changed, written


def require_owned_writes(
    algorithm: target.Analyzer,
    before: dict[tuple[int, type], int],
    after: dict[tuple[int, type], int],
) -> set[tuple[int, type]]:
    """Require that *algorithm* only touched Metadata types it declares.

    Ownership is checked against what actually landed on the IR rather than
    against what the algorithm reports, so an analysis cannot quietly overwrite
    or delete a dependency's results. Returns the records it left behind, keyed
    per expression so a later member deleting one of them can be accounted for.
    """
    owned = set(algorithm.produces)
    changed, written = metadata_delta(before, after)
    trespassed = {key[-1] for key in changed} - owned
    if trespassed:
        raise Error(
            f"{algorithm.selector!r} changed Metadata it does not declare: "
            f"{sorted(item.__name__ for item in trespassed)}; it declares "
            f"{sorted(item.__name__ for item in owned)}"
        )
    return written


__all__ = ["Result", "analyze", "build_context"]
