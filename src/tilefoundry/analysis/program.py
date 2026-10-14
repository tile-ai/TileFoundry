"""Retained analysis definitions, invocation frames, and frame-local records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Constant, Expr, Var, attach_metadata, get_metadata
from tilefoundry.ir.core.metadata import BindingMetadata
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import (
    InstantiateDimensionCloner,
    SpecializationError,
    _DimVarCollector,
    bound_dims_of,
    dim_vars_reached,
)
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types.callable_type import callable_type_for
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.types.substitute import dim_vars_by_name, substitute_shape_dim
from tilefoundry.ir.visitor import (
    BindingSubstitutionCloner,
    ExprWalker,
    collect_exprs,
    expr_children,
    function_values,
)
from tilefoundry.visitor_registry.contexts import CallFrame, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

from .errors import AnalysisError


def invocation_key(frame):
    """The invocation key, with None representing the root frame."""
    return None if frame is None else frame.key


@dataclass(frozen=True)
class CallSite:
    """A retained call and the one derived body implementing its definition."""

    call: Call
    definition: Function
    body: Function
    frame: CallFrame
    key: str | None
    activations: tuple[Var, ...]
    weights: tuple[Var, ...]
    index: int | Expr | None
    label: str


@dataclass
class Program:
    """One typed derived body per definition, with retained call-site edges."""

    module: Module
    root: Function
    sites: dict[int, CallSite] = field(default_factory=dict)
    frames: dict[int, CallFrame] = field(default_factory=dict)
    definitions: dict[int, Function] = field(default_factory=dict)

    def bodies(self) -> tuple[Function, ...]:
        """Definitions in first-reached order, including the root."""
        return tuple(self.definitions.values())

    def callee_first(self) -> tuple[Function, ...]:
        """Order derived definitions after their dependencies."""
        ordered, seen = [], set()

        def visit(body):
            if id(body) in seen:
                return
            seen.add(id(body))
            for value in collect_exprs(body.body):
                if site := self.sites.get(id(value)):
                    visit(site.body)
            ordered.append(body)

        visit(self.root)
        return tuple(ordered)


class _BodyCloner(BindingSubstitutionCloner):
    def __init__(self, metadata):
        super().__init__(metadata=metadata)
        self.number = 0

    def visit_Call(self, expr, ctx):
        rebuilt = super().visit_Call(expr, ctx)
        name = f"v{self.number}"
        self.number += 1
        return replace(rebuilt, metadata=(*rebuilt.metadata, BindingMetadata(name)))


def build_program(module, fn, dims=None) -> Program:
    """Bind one reachable program, then infer one body per definition and signature."""
    from .check import _required_frame, _view_metadata  # noqa: PLC0415 -- cycle

    if dims is not None:
        if not isinstance(dims, Mapping) or not dims:
            raise SpecializationError(
                f"specialising {fn.name!r} needs a non-empty mapping of dimension names to extents, got {dims!r}"
            )
        for name, extent in dims.items():
            if not isinstance(name, str) or not name:
                raise SpecializationError(f"specialising {fn.name!r}: {name!r} is not a dimension name")
            if isinstance(extent, bool) or not isinstance(extent, int):
                raise SpecializationError(
                    f"specialising {fn.name!r}: dimension {name!r} takes an integer extent, got {extent!r}"
                )
    supplied = {} if dims is None else dict(dims)
    original_frame = _required_frame(module, fn)
    declared = {}
    collector = _DimVarCollector(declared, supplied)
    collector.visit_function(fn, original_frame)
    if collector.cycles:
        raise AnalysisError(f"recursive call path through {collector.cycles[0]!r}")
    if collector.missing_bodies:
        name, path = collector.missing_bodies[0]
        raise AnalysisError(f"cannot analyze Function {name!r} without a body at {path!r}")
    unknown = sorted(set(supplied) - declared.keys())
    if unknown:
        raise SpecializationError(f"{fn.name!r} has no dimension named {unknown!r}; it states {sorted(declared)}")
    for name, extent in supplied.items():
        substitute_shape_dim(declared[name], {name: extent})
    cloner = InstantiateDimensionCloner(dims)
    for function, _frame in collector.functions:
        cloner.function(function)
    bound_root = cloner.function(fn)
    bound_module = cloner.module(module)
    initial = CallFrame(bound_module, bound_root)
    residual = set(dim_vars_reached(bound_root, initial))
    for owner in cloner.modules.values():
        residual.update(dim_vars_by_name(owner.topologies))
    if residual:
        raise SpecializationError(
            f"{fn.name!r} still states {sorted(residual)} as ranges after binding {sorted(supplied)}; every dimension has to be given an extent"
        )
    program = Program(bound_module, bound_root)
    signatures = {}
    ctx = TypeInferContext()

    def derive(definition, frame, types, mesh, label):
        identity = id(definition)
        signature = (types, mesh)
        if identity in signatures:
            previous, path = signatures[identity]
            if previous != signature:
                raise AnalysisError(
                    f"callee {definition.name!r} has incompatible signatures at {path!r} and "
                    f"{label!r}: {previous!r} versus {signature!r}"
                )
            return program.definitions[identity]
        signatures[identity] = signature, label
        inner = replace(ctx, frame=frame, current_mesh=mesh, memo={})
        chosen = definition
        params = tuple(
            replace(p, type=t, metadata=_view_metadata(p))
            for p, t in zip(chosen.params, types, strict=True)
        )
        body = _BodyCloner(_view_metadata).visit(
            chosen.body, {id(p): q for p, q in zip(chosen.params, params, strict=True)}
        )
        derived = replace(
            chosen,
            params=params,
            body=body,
            variants=(),
            converters=(),
            metadata=_view_metadata(chosen),
        )
        derived._specialized_from = definition._specialized_from
        derived._specialized_dims = bound_dims_of(definition)
        program.definitions[identity] = derived
        program.frames[id(derived)] = replace(frame, function=derived)
        TypeInferVisitor(ranges=True).visit(
            derived, replace(inner, frame=program.frames[id(derived)])
        )
        class Sites(ExprWalker):
            def __init__(self):
                super().__init__()
                self.calls = []

            def visit_MeshRegion(self, region, current_mesh=None):
                for arg in region.args:
                    self.visit(arg, current_mesh)
                self.visit(region.body, make_mesh(current_mesh, region.mesh) if current_mesh else region.mesh)

            def visit_Call(self, expr, current_mesh=None):
                self.visit_operands(expr, current_mesh)
                target = expr.target
                callee = target if isinstance(target, Function) else None
                child_frame = frame
                slot = target if isinstance(target, ModuleList) else (
                    target.args[0] if isinstance(target, Call) and isinstance(target.target, TupleGetItem)
                    and isinstance(target.args[0], ModuleList) else None
                )
                key, index = None, None
                if slot is not None:
                    key = slot.key
                    slot = frame.module.children[key]
                    index = target.args[1] if isinstance(target, Call) else None
                    index = index.value if isinstance(index, Constant) else index
                    child = slot.elements[index if isinstance(index, int) else 0]
                    callee = child.entry_function()
                    child_frame = CallFrame(
                        child, callee, child.target if child.target is not None else frame.target,
                        child.topologies if child.topologies is not None else frame.topologies,
                    )
                if callee is not None:
                    stem = callee.name if key is None else key
                    if index is not None:
                        stem += f"[{index if isinstance(index, int) else getattr(index, 'name', None) or 'i'}]"
                    self.calls.append((expr, callee, child_frame, key, index, current_mesh, stem))

        sites = Sites()
        sites.visit(body, mesh)
        stems = {}
        for expr, _callee, _frame, _key, _index, _mesh, stem in sites.calls:
            stems.setdefault(stem, []).append(expr)
        for expr, callee, child_frame, key, index, current_mesh, stem in sites.calls:
            siblings = stems[stem]
            if len(siblings) > 1:
                stem += f"#{next(i for i, value in enumerate(siblings) if value is expr)}"
            given = iter(expr.args)
            types = tuple(
                param.type if key is not None and param.is_const else next(given).type
                for param in callee.params
            )
            child = derive(callee, child_frame, types, current_mesh, f"{label}/{stem}")
            activations = tuple(p for p in child.params if key is None or not p.is_const)
            weights = tuple(p for p in child.params if key is not None and p.is_const)
            program.sites[id(expr)] = CallSite(
                expr, callee, child, replace(child_frame, function=child), key,
                activations, weights, index, stem,
            )
        derived.return_type = derived.body.type
        derived.type = callable_type_for(derived.params, derived.return_type)
        return derived

    program.root = derive(bound_root, initial, tuple(p.type for p in bound_root.params), None, fn.name)
    return program


class FrameRecords:
    """Root metadata lives on IR; callee metadata lives in an invocation side table."""

    def __init__(self):
        self.table = {}
        self.frames: list = []

    def get(self, frame, node, kind):
        if frame is None or frame.key is None:
            return get_metadata(node, kind)
        return self.table.get((frame, id(node), kind))

    def attach(self, frame, node, record):
        if frame is None or frame.key is None:
            attach_metadata(node, record)
        else:
            self.table[frame, id(node), type(record)] = record

    def reader(self, frame) -> Callable:
        """Return a metadata lookup bound to one invocation."""
        return lambda node, kind: self.get(frame, node, kind)

    def definition_reader(self, body):
        frame = next(frame for frame in self.frames if frame.function is body)
        return self.reader(frame)

    def snapshot(self, program):
        snapshot = dict((key, id(value)) for key, value in self.table.items())
        for body in program.bodies():
            for node in function_values(body):
                for record in node.metadata:
                    snapshot[None, id(node), type(record)] = id(record)
        return snapshot


class InvocationWalker:
    """Step into retained call frames while swapping the visitor's DAG memo."""

    def attach(self, node, record, ctx):
        if ctx.records is None:
            attach_metadata(node, record)
        else:
            ctx.records.attach(ctx.current.invocation, node, record)

    def record(self, node, kind, ctx):
        return (
            get_metadata(node, kind)
            if ctx.records is None
            else ctx.records.get(ctx.current.invocation, node, kind)
        )

    def visit_Call(self, expr, ctx):
        frame = getattr(ctx.current, "invocation", None)
        child = None if frame is None else frame.callees.get(id(expr))
        if child is None:
            return self.default_visit_leaf(expr, self.visit_operands(expr, ctx), ctx)
        for arg in expr.args:
            self.visit(arg, ctx)
        if isinstance(child.site.index, Expr):
            self.visit(child.site.index, ctx)
        memo, self._memo = self._memo, {}
        try:
            self.visit(child.function.body, replace(ctx, current=child, module=child.frame.module, target=child.frame.target))
        finally:
            self._memo = memo
        self.call_result(expr, ctx)
        return None

    def call_result(self, expr, ctx):
        """A retained call contributes no extra operation-level work."""


def invocation_values(frame):
    """Walk values in operand-first order, entering callees at their call sites."""
    seen = set()

    def visit(expr):
        if id(expr) in seen:
            return
        seen.add(id(expr))
        child = frame.callees.get(id(expr))
        if child is not None:
            for arg in expr.args:
                yield from visit(arg)
            if isinstance(child.site.index, Expr):
                yield from visit(child.site.index)
            yield from invocation_values(child)
        else:
            for arg in expr_children(expr):
                yield from visit(arg)
        yield frame, expr

    yield from visit(frame.function.body)
