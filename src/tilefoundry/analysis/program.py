"""Retained analysis definitions, invocation frames, and frame-local records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Expr, Var, attach_metadata, get_metadata
from tilefoundry.ir.core.metadata import BindingMetadata
from tilefoundry.ir.core.module import Module, instance_slot, resolve_callee
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.specialize import (
    InstantiateDimensionCloner,
    SpecializationError,
    bound_dims_of,
    dim_vars_reached,
    variant_for,
)
from tilefoundry.ir.types.callable_type import callable_type_for
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.types.substitute import dim_vars_by_name, substitute_shape_dim
from tilefoundry.ir.visitor import (
    BindingSubstitutionCloner,
    collect_exprs,
    expr_children,
    function_values,
)
from tilefoundry.visitor_registry.contexts import FunctionScope, ModuleInstance, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

from .errors import AnalysisError


def frame_key(frame):
    """The invocation key, with None representing the root frame."""
    return None if frame is None else frame.key


def _call_stem(definition, instance):
    if instance is None:
        return definition.name
    stem = instance.key
    if instance.index is not None:
        index = instance.index
        stem += f"[{index if isinstance(index, int) else getattr(index, 'name', None) or 'i'}]"
    return stem


@dataclass(frozen=True)
class CallSite:
    """A retained call and the one derived body implementing its definition."""

    call: Call
    definition: Function
    body: Function
    instance: ModuleInstance | None
    activations: tuple[Var, ...]
    weights: tuple[Var, ...]
    index: Expr | None
    label: str


@dataclass
class Program:
    """One typed derived body per definition, with retained call-site edges."""

    module: Module
    root: Function
    sites: dict[int, CallSite] = field(default_factory=dict)
    scopes: dict[int, FunctionScope] = field(default_factory=dict)
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
    from .check import _required_scope, _view_metadata, program_dim_vars  # noqa: PLC0415 -- cycle

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
    declared = program_dim_vars(module, fn)
    supplied = {} if dims is None else dict(dims)
    reached, checked = [], set()

    def preflight(function, scope, path, active):
        if id(function) in active:
            raise AnalysisError(f"recursive call path through {function.name!r}: {path}")
        key = (id(function), id(scope.module))
        if key in checked:
            return
        chosen = variant_for(function, supplied)
        if chosen.body is None:
            raise AnalysisError(f"cannot analyze Function {function.name!r} without a body")
        reached.append(function)
        for expr in collect_exprs(chosen.body):
            if not isinstance(expr, Call):
                continue
            callee, inner = resolve_callee(expr, scope)
            if callee is not None:
                preflight(callee, inner, f"{path}/{callee.name}", active | {id(function)})
        checked.add(key)

    original_scope = _required_scope(module, fn)
    preflight(fn, original_scope, fn.name, set())
    unknown = sorted(set(supplied) - declared.keys())
    if unknown:
        raise SpecializationError(f"{fn.name!r} has no dimension named {unknown!r}; it states {sorted(declared)}")
    for name, extent in supplied.items():
        substitute_shape_dim(declared[name], {name: extent})
    cloner = InstantiateDimensionCloner(dims)
    for function in reached:
        cloner.function(function)
    bound_root = cloner.function(fn)
    bound_module = cloner.module(module)
    instances = tuple(replace(frame, module=cloner.module(frame.module)) for frame in original_scope.instances)
    initial = FunctionScope(cloner.module(original_scope.module), bound_root, instances)
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

    def derive(definition, scope, types, mesh, label):
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
        inner = replace(ctx, scope=scope, current_mesh=mesh, memo={})
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
        program.scopes[id(derived)] = replace(scope, function=derived)
        TypeInferVisitor(ranges=True).visit(
            derived, replace(inner, scope=program.scopes[id(derived)])
        )
        stems = {}
        for value in collect_exprs(body):
            if not isinstance(value, Call):
                continue
            callee, entered = resolve_callee(value, scope)
            if callee is not None:
                instance = entered.instances[-1] if instance_slot(value.target) else None
                stems.setdefault(_call_stem(callee, instance), []).append(value)

        def discover(expr, current_mesh):
            if isinstance(expr, MeshRegion):
                for arg in expr.args:
                    discover(arg, current_mesh)
                discover(
                    expr.body, make_mesh(current_mesh, expr.mesh) if current_mesh else expr.mesh
                )
                return
            for operand in expr_children(expr):
                discover(operand, current_mesh)
            if not isinstance(expr, Call) or id(expr) in program.sites:
                return
            callee, callee_scope = resolve_callee(expr, scope)
            if callee is None:
                return
            located = instance_slot(expr.target)
            instance = callee_scope.instances[-1] if located else None
            index = expr.target.args[1] if located and isinstance(expr.target, Call) else None
            activations = tuple(p for p in callee.params if not (located and p.is_const))
            weights = tuple(p for p in callee.params if located and p.is_const)
            supplied = iter(expr.args)
            arg_types = tuple(
                p.type if located and p.is_const else next(supplied).type
                for p in callee.params
            )
            stem = _call_stem(callee, instance)
            siblings = stems[stem]
            if len(siblings) > 1:
                stem += f"#{next(i for i, value in enumerate(siblings) if value is expr)}"
            path = f"{label}/{stem}"
            child = derive(callee, callee_scope, arg_types, current_mesh, path)
            child_activations = tuple(
                p for p, old in zip(child.params, callee.params, strict=True) if any(old is activation for activation in activations)
            )
            child_weights = tuple(
                p for p, old in zip(child.params, callee.params, strict=True) if any(old is weight for weight in weights)
            )
            program.sites[id(expr)] = CallSite(
                expr, callee, child, instance, child_activations, child_weights, index, stem
            )

        discover(body, mesh)
        derived.return_type = derived.body.type
        derived.type = callable_type_for(derived.params, derived.return_type)
        return derived

    program.root = derive(bound_root, initial, tuple(p.type for p in bound_root.params), None, fn.name)
    return program


@dataclass(eq=False)
class CallFrame:
    """One invocation path; shared nodes acquire records only through this frame."""

    parent: CallFrame | None
    call: Call | None
    site: CallSite | None
    body: Function
    instances: tuple[ModuleInstance, ...]
    scope: object = None
    callees: dict[int, CallFrame] = field(default_factory=dict)
    variance: dict[int, frozenset[int]] = field(default_factory=dict)
    tables: dict[str, object] = field(default_factory=dict)
    records: object | None = None
    result_tied: bool = False
    label: str = ""

    @property
    def key(self):
        """Keep root-only record keys backward compatible."""
        return None if self.parent is None else self

    def resource_key(self, weight):
        """The instance path addressing an implicit constant."""
        segments = []
        for frame in self.instances[1:]:
            segments.append(frame.key)
            if frame.index is not None:
                segments.append(str(frame.index) if isinstance(frame.index, int) else "{i}")
        return ".".join((*segments, weight.name))


class FrameRecords:
    """Root metadata lives on IR; callee metadata lives in an invocation side table."""

    def __init__(self):
        self.table = {}
        self.frames: list[CallFrame] = []

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
        frame = next(frame for frame in self.frames if frame.body is body)
        return self.reader(frame)

    def snapshot(self, program):
        snapshot = dict((key, id(value)) for key, value in self.table.items())
        for body in program.bodies():
            for node in function_values(body):
                for record in node.metadata:
                    snapshot[None, id(node), type(record)] = id(record)
        return snapshot


class FrameWalker:
    """Step into retained call frames while swapping the visitor's DAG memo."""

    def attach(self, node, record, ctx):
        if ctx.records is None:
            attach_metadata(node, record)
        else:
            ctx.records.attach(ctx.current.frame, node, record)

    def record(self, node, kind, ctx):
        return (
            get_metadata(node, kind)
            if ctx.records is None
            else ctx.records.get(ctx.current.frame, node, kind)
        )

    def visit_Call(self, expr, ctx):
        frame = getattr(ctx.current, "frame", None)
        child = None if frame is None else frame.callees.get(id(expr))
        if child is None:
            return self.default_visit_leaf(expr, self.visit_operands(expr, ctx), ctx)
        for arg in expr.args:
            self.visit(arg, ctx)
        if child.site.index is not None:
            self.visit(child.site.index, ctx)
        memo, self._memo = self._memo, {}
        try:
            self.visit(child.body.body, replace(ctx, current=child.scope))
        finally:
            self._memo = memo
        self.call_result(expr, ctx)
        return None

    def call_result(self, expr, ctx):
        """A retained call contributes no extra operation-level work."""


def frame_values(frame):
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
            if child.site.index is not None:
                yield from visit(child.site.index)
            yield from frame_values(child)
        else:
            for arg in expr_children(expr):
                yield from visit(arg)
        yield frame, expr

    yield from visit(frame.body.body)
