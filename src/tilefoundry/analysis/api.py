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

from tilefoundry.analysis.check import check_analyses
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.iteration_scope import ScopeBuilder
from tilefoundry.analysis.metadata import ExecutionMeshMetadata
from tilefoundry.analysis.registry import Analyzer
from tilefoundry.analysis.report import render_json, report_data
from tilefoundry.analysis.visitor import AnalyzeContext, CallContext, VisitContext
from tilefoundry.dump import DumpFlags, dump
from tilefoundry.ir.core import IRMetadata, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.visitor import function_values
from tilefoundry.target import Target, UnsupportedCapabilityError
from tilefoundry.visitor_registry.call_frame import CallFrame, CallFrameVisitor


@dataclass(frozen=True)
class AnalysisResult:
    """Describe what one public analysis call computed.

    ``executed`` records dependency order and ``metadata_types`` lists records
    actually written, excluding analyses that matched nothing. ``function`` is
    the checked root that carries aggregate records; ``module`` remains the call's
    execution domain.
    """

    module: Module
    function: Function
    analyses: tuple[str, ...]
    topology_level: str | None
    executed: tuple[str, ...]
    metadata_types: tuple[type[IRMetadata], ...]
    functions: tuple[Function, ...]

    @property
    def level(self) -> str | None:
        """The selected topology level."""
        return self.topology_level


def _algorithm(target: Target, selector: str, *, root: str) -> Analyzer:
    """The service selected by the resolved Target for *selector*."""
    try:
        return target.get_analyzer(selector)
    except UnsupportedCapabilityError as error:
        if selector == root:
            raise AnalysisError(str(error)) from None
        raise AnalysisError(
            f"{root!r} depends on {selector!r}, which is not registered: {error}"
        ) from None


def _roots(analysis: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one or more root selectors, preserving their first occurrence."""
    requested = (analysis,) if isinstance(analysis, str) else tuple(analysis)
    if not requested:
        raise AnalysisError("analyze: analysis must name at least one root")
    if any(not isinstance(item, str) or not item for item in requested):
        raise AnalysisError(
            f"analyze: every analysis must be a non-empty selector, got {analysis!r}"
        )
    return tuple(dict.fromkeys(requested))


def _closure(target: Target, roots: tuple[str, ...]) -> tuple[Analyzer, ...]:
    """*roots* and everything they transitively need, dependencies first.

    The walk is depth-first over declared names, which both orders the closure
    and detects a cycle: meeting a selector that is still being visited means
    it depends on itself through some path.
    """
    ordered: list[Analyzer] = []
    done: set[str] = set()
    visiting: list[str] = []

    def visit(selector: str, root: str) -> None:
        if selector in done:
            return
        if selector in visiting:
            cycle = " -> ".join([*visiting[visiting.index(selector) :], selector])
            raise AnalysisError(f"analysis dependency cycle: {cycle}")
        visiting.append(selector)
        algorithm = _algorithm(target, selector, root=root)
        for required in algorithm.requires:
            visit(required, root)
        visiting.pop()
        done.add(selector)
        ordered.append(algorithm)

    for root in roots:
        visit(root, root)
    return tuple(ordered)


def _check_subject(module: Module, function: Function) -> None:
    """Require an HIR definition owned by the supplied checked program."""
    if not isinstance(module, Module):
        raise TypeError(
            f"analyze: expected a Module, got {type(module).__name__}. A Function "
            "carries no execution context; select the Module that owns it."
        )
    if not isinstance(function, Function):
        raise TypeError(f"analyze: expected an hir.Function, got {type(function).__name__}")
    if not module.owns(function, derived=True):
        raise AnalysisError(
            f"analyze: {function.name!r} is not a function of module {module.name!r}"
        )


def _build_context(
    module: Module,
    function: Function,
    *,
    topology_level: str | None = None,
    options: object | None = None,
) -> AnalyzeContext:
    """Build one local scope tree per definition in a previously checked program."""
    _check_subject(module, function)
    topologies = module.effective_topologies()
    if topology_level is None:
        topology_level = topologies[0].name if topologies else None

    definitions = {}
    frames = tuple(CallFrameVisitor.walk_frames(CallFrame(module, function), postorder=True))
    for frame in frames:
        context = AnalyzeContext(
            frame.module,
            frame.target,
            topology_level,
            options,
            frame=frame,
            definitions=definitions,
        )
        builder = ScopeBuilder(
            frame.module, frame.function, ctx=context, is_root=frame.function is function
        )
        context.root = context.current = builder.build()
        definitions[id(frame.function)] = context
    return definitions[id(function)]


def _check_valuations(closure: tuple[Analyzer, ...]) -> None:
    """Validate declared record requirements before a family can write metadata."""
    read_types = {}
    for algorithm in closure:
        records = set(algorithm.produces)
        for dependency in algorithm.requires:
            records.update(read_types[dependency])
        read_types[algorithm.selector] = records
        for record in sorted(records, key=lambda item: item.__name__):
            for valuation in record.requires:
                for frame in (CallContext, VisitContext):
                    if not issubclass(frame, valuation):
                        raise AnalysisError(
                            f"{algorithm.selector}: {frame.__name__} does not "
                            f"provide {valuation.__name__} for {record.__name__}"
                        )


def analyze(
    module: Module,
    function: Function,
    *,
    analysis: str | Sequence[str],
    topology_level: str | None = None,
    level: str | None = None,
    options: object | None = None,
) -> AnalysisResult:
    """Run the requested analyses' union dependency closure over *function*.

    The caller supplies the program returned by ``check_program``. Analyses
    write records onto that same program; dimensions are bound by the caller.
    The selected topology defaults to the module's coarsest level.
    """
    if topology_level is not None and level is not None and topology_level != level:
        raise ValueError("analyze: topology_level and level disagree")
    topology_level = topology_level if topology_level is not None else level
    _check_subject(module, function)
    if get_metadata(function, ExecutionMeshMetadata) is None:
        raise AnalysisError("analyze needs the program returned by check_program")
    roots = _roots(analysis)
    result_module = module
    closure = _closure(module.resolve_target(), roots)
    if topology_level is None:
        topologies = module.effective_topologies()
        topology_level = topologies[0].name if topologies else None
    _check_valuations(closure)
    check_analyses(module, function, closure, topology_level=topology_level)
    context = _build_context(
        module,
        function,
        topology_level=topology_level,
        options=options,
    )
    module = context.module
    topology_level = context.topology_level
    function = context.root.owner
    functions = tuple(frame.function for frame in CallFrameVisitor.walk_frames(context.frame))

    order: list[type[IRMetadata]] = []
    written_records: set[tuple] = set()
    for algorithm in closure:
        before = _metadata_snapshot(functions)
        try:
            algorithm.run(function, context)
        except UnsupportedCapabilityError as error:
            raise AnalysisError(f"{algorithm.selector}: {error}") from None
        after = _metadata_snapshot(functions)
        records = _require_owned_writes(algorithm, before, after)
        written_records |= records
        written = {key[-1] for key in records}
        for metadata_type in algorithm.produces:
            if metadata_type in written and metadata_type not in order:
                order.append(metadata_type)

    final = _metadata_snapshot(functions)
    surviving = {
        metadata_type for key in written_records & final.keys() for metadata_type in (key[-1],)
    }

    result = AnalysisResult(
        module=result_module,
        function=function,
        analyses=roots,
        topology_level=topology_level,
        executed=tuple(algorithm.selector for algorithm in closure),
        metadata_types=tuple(item for item in order if item in surviving),
        functions=functions,
    )
    dump(
        "analysis.json",
        lambda: render_json(
            report_data(
                module=result.module,
                function=result.function,
                analyses=result.analyses,
                topology_level=result.topology_level,
                executed=result.executed,
                metadata_types=result.metadata_types,
                functions=result.functions,
            )
        ),
        DumpFlags.ANALYSIS,
    )
    return result


def _metadata_snapshot(
    functions: tuple[Function, ...],
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
        for expr in function_values(fn):
            for item in expr.metadata:
                snapshot[(id(expr), type(item))] = id(item)
    return snapshot


def _metadata_delta(
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


def _require_owned_writes(
    algorithm: Analyzer,
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
    changed, written = _metadata_delta(before, after)
    trespassed = {key[-1] for key in changed} - owned
    if trespassed:
        raise AnalysisError(
            f"{algorithm.selector!r} changed Metadata it does not declare: "
            f"{sorted(item.__name__ for item in trespassed)}; it declares "
            f"{sorted(item.__name__ for item in owned)}"
        )
    return written


__all__ = ["AnalysisResult", "analyze"]
