"""Canonical Python DSL printer for HIR Functions.

Converts a ``hir.Function`` to executable Python source using the
``@func`` DSL. Placement sugar names only explicit mesh-scope bindings;
otherwise the verbose ``ShardLayout(...)`` form is used.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass

from tilefoundry.ir.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    StorageConstraint,
    constraint_metadata,
)
from tilefoundry.ir.constraints.layout import is_layout_wildcard
from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    IRMetadata,
    Tuple,
    Var,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.specialize import (
    canonical_specialization_signature,
    display_name,
    origin_of,
)
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice, window_base
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
)
from tilefoundry.ir.visitor import expr_children
from tilefoundry.utils.python_source import PythonExpr

from .print_context import HirPrintContext
from .printer_base import PythonPrinter
from .tir_printer import _function_block as _tir_function_block
from .tir_printer import tir_function_to_python, tir_module_to_python
from .values import PARTS, render_comment


class HirPrinter(PythonPrinter):
    """Canonical HIR expression, type, and function printer."""

    def __init__(self) -> None:
        super().__init__()
        self._names: dict[int, str] = {}
        self._param_alias: dict[int, Expr] = {}
        self._child_entries: dict[int, str] = {}
        self._moved_window = lambda start, size, stride: None

    def print(self, fn: HirFunction, *, options=None) -> str:
        return self.render(fn, options=options).source

    def render(self, fn: HirFunction, *, options=None) -> _PythonRendering:
        return _render_hir_function(fn, options=options)

    def bind_def(
        self,
        names: dict[int, str],
        param_alias: dict[int, Expr],
        child_entries: dict[int, str],
        moved_window,
    ) -> None:
        """Install the value-naming environment for one function definition."""
        self._names = names
        self._param_alias = param_alias
        self._child_entries = child_entries
        self._moved_window = moved_window

    def tuple_reference(self, elements) -> str:
        inner = ", ".join(
            repr(element.value)
            if isinstance(element, Constant)
            else self.reference(element)
            for element in elements
        )
        return f"({inner}{',' if len(elements) == 1 else ''})"

    def reference(self, expr: Expr) -> str:
        """Return the binding that denotes one value in the current definition."""
        if isinstance(expr, Tuple):
            return self.tuple_reference(expr.elements)
        if id(expr) in self._param_alias:
            return self.reference(self._param_alias[id(expr)])
        projection = _region_projection(expr)
        if isinstance(projection, LoopRegion):
            return self._names[id(projection.carried_args[expr.target.index])]
        if isinstance(expr, LoopRegion):
            carried = tuple(self._names[id(carry)] for carry in expr.carried_args)
            return carried[0] if len(carried) == 1 else "(" + ", ".join(carried) + ")"
        if isinstance(projection, MeshRegion):
            return self.reference(projection.body.elements[expr.target.index])
        if isinstance(expr, MeshRegion):
            return self.reference(expr.body)
        return self._names[id(expr)]

    def _slice_start(self, start, size, stride) -> str:
        moved = self._moved_window(start, size, stride)
        if moved is None:
            return repr(start.value) if isinstance(start, Constant) else self.reference(start)
        window, offset = moved
        return _moved_window_ref(self.reference(window), offset)

    def visit_program_call(self, expr: Call, ctx=None) -> str:
        """Render one HIR call after expression-level dispatch selected it."""
        target = expr.target
        args_text = ", ".join(self.reference(arg) for arg in expr.args)
        if isinstance(target, Reshard):
            if ctx is not None:
                ctx.imports.add("from tilefoundry.dsl.tf import *")
            layout_kw = ""
            if target.layout is not None:
                with self.type_surface(indent=self._indent + "    "):
                    layout_kw = ", layout=" + self.visit(target.layout, ctx)
            storage = ""
            if target.storage is not None:
                storage_name = target.storage.name.lower()
                if ctx is not None:
                    ctx.use(
                        PythonExpr(
                            (f"from tilefoundry.dsl.storage import {storage_name}",),
                            storage_name,
                        )
                    )
                storage = f", storage={storage_name}"
            return f"reshard({args_text}{layout_kw}{storage})"
        if isinstance(target, HirFunction):
            binding = _module_callee_binding(target, self._child_entries)
            return f"{binding or target.name}({args_text})"
        if isinstance(target, Slice):
            starts = expr.args[1]
            if not isinstance(starts, Tuple):
                raise ValueError("canonical_source: Slice starts must be a Tuple")
            indexers: list[str] = []
            runtime_starts = False
            for axis, (start, size, stride) in enumerate(
                zip(starts.elements, target.sizes, target.strides)
            ):
                if self._moved_window(start, size, stride) is not None:
                    indexers.append(self._slice_start(start, size, stride))
                    continue
                dim = expr.args[0].type.shape[axis]
                if (
                    isinstance(start, Constant)
                    and start.value == 0
                    and size == dim
                    and stride == 1
                ):
                    indexers.append(":")
                    continue
                if not (
                    isinstance(start, Constant)
                    and isinstance(start.value, int)
                    and isinstance(size, int)
                    and isinstance(stride, int)
                ):
                    runtime_starts = True
                    break
                begin = int(start.value)
                stop = begin + size * stride
                indexers.append(
                    f"{begin}:{stop}" if stride == 1 else f"{begin}:{stop}:{stride}"
                )
            if runtime_starts:
                if ctx is not None:
                    ctx.imports.add("from tilefoundry.dsl.tf import *")
                start_refs = ", ".join(
                    self._slice_start(start, size, stride)
                    for start, size, stride in zip(
                        starts.elements, target.sizes, target.strides
                    )
                )
                if len(starts.elements) == 1:
                    start_refs += ","
                return (
                    f"slice({self.reference(expr.args[0])}, ({start_refs}), "
                    f"sizes={_attr_tuple_str(target.sizes, self, ctx)}, "
                    f"strides={_attr_tuple_str(target.strides, self, ctx)})"
                )
            return f"{self.reference(expr.args[0])}[{', '.join(indexers)}]"

        if ctx is not None:
            ctx.imports.add("from tilefoundry.dsl.tf import *")
        alias_name = _kinded_alias_name(target)
        suppressed = {"kind"} if alias_name is not None else set()
        attrs: list[str] = []
        for param in type(target).params():
            if param.kind != "attribute":
                continue
            value = getattr(target, param.name, None)
            if value is None or param.name in suppressed or param.name == "layout":
                continue
            if isinstance(value, str):
                attrs.append(f'{param.name}="{value}"')
            elif isinstance(value, DType):
                attrs.append(f'{param.name}="{value.name}"')
            elif isinstance(value, enum.Enum) and isinstance(value.value, str):
                attrs.append(f'{param.name}="{value.value}"')
            elif isinstance(value, float):
                if math.isinf(value):
                    literal = "-1e999" if value < 0 else "1e999"
                elif math.isnan(value):
                    literal = "(1e999 - 1e999)"
                else:
                    literal = repr(value)
                attrs.append(f"{param.name}={literal}")
            elif isinstance(value, (ShardLayout, TensorType)):
                with self.type_surface(indent=self._indent + "        "):
                    rendered = self.visit(value, ctx)
                if isinstance(value, TensorType):
                    rendered = " ".join(rendered.split())
                attrs.append(f"{param.name}={rendered}")
            elif isinstance(value, tuple):
                attrs.append(f"{param.name}={_attr_tuple_str(value, self, ctx)}")
            else:
                attrs.append(f"{param.name}={value}")
        return f"{_op_name(target)}({', '.join([*(self.reference(arg) for arg in expr.args), *attrs])})"


@dataclass(frozen=True)
class PythonPrintOptions:
    """Optional non-canonical annotations for inspection output."""

    show_types: bool = False
    comment_metadata_types: tuple[type[IRMetadata], ...] = ()
    comment_opt_in: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _PrintedStatement:
    """The left-hand side and physical start line of one emitted Call."""

    value: str
    line: int


@dataclass(frozen=True)
class _PythonRendering:
    """Rendered source and the Call statements produced in that same pass."""

    source: str
    statements: dict[int, _PrintedStatement]


def _physical_line_count(lines: list[str]) -> int:
    return sum(line.count("\n") + 1 for line in lines)


def _compact_type(ty: object, printer: PythonPrinter, ctx) -> str:
    """One physical-line, DSL-shaped type annotation for inspection output."""
    if isinstance(ty, (TensorType, TupleType)):
        with printer.type_surface():
            rendered = printer.visit(ty, ctx)
        return " ".join(rendered.split())
    return repr(ty)


def _comments(expr: Expr, options: PythonPrintOptions, printer: PythonPrinter, ctx) -> str:
    """Return same-line annotations for one printed statement.

    Omit the binding because the left-hand side already carries its emitted name
    and importing recovers it there. Emit types as annotation fragments without
    a redundant ``type=`` key.

    Part zero is the value's own type, which carries no key: it is not a
    measurement of the value, it is the value, and it is DSL text that can be
    pasted back. Every later part is what a record measured, and ``PARTS`` is
    the boundary between those two languages.
    """
    comments: list[str] = []
    if options.show_types:
        comments.append(_compact_type(expr.type, printer, ctx))
    for metadata_type in options.comment_metadata_types:
        metadata = get_metadata(expr, metadata_type)
        if metadata is None:
            continue
        comment = render_comment(metadata, opt_in=options.comment_opt_in)
        if comment is not None:
            comments.append(comment)
    return f"  # {PARTS.join(comments)}" if comments else ""


def _moved_window_ref(name: str, offset: int) -> str:
    """A tile-window indexer, carrying the compile-time offset that moves it."""
    if offset == 0:
        return name
    return f"{name} + {offset}" if offset > 0 else f"{name} - {-offset}"


def _attr_tuple_str(value: tuple, printer: PythonPrinter, ctx) -> str:
    """Render an attribute's tuple value as a Python tuple literal.

    A shape-valued attribute -- `new_shape`, a tile's extents -- can hold a
    `DimVar` or dim arithmetic, and the tuple's own `str` would render those as
    dataclass reprs. Printing them the way the annotations do keeps one program
    described one way, and keeps the printed source importable: the declaration
    the header emits binds the name, not the repr.
    """
    rendered = tuple(
        printer.visit(entry, ctx) if _is_dim_entry(entry) else repr(entry)
        for entry in value
    )
    if len(rendered) == 1:
        return f"({rendered[0]},)"
    return "(" + ", ".join(rendered) + ")"


def _is_dim_entry(entry: object) -> bool:
    """Whether *entry* is a dimension rather than a plain attribute value."""
    return isinstance(entry, (DimVar, Var, Constant, Call))


def _tensor_import_names(fn: HirFunction) -> str:
    """``"Tensor"`` or ``"ConstTensor, Tensor"``.

    ``"Tensor"`` or ``"ConstTensor, Tensor"`` — whichever the printed
    signature (base plus every variant) actually references.
    """
    if any(p.is_const for f in (fn, *fn.variants) for p in f.params):
        return "ConstTensor, Tensor"
    return "Tensor"


def _op_name(target) -> str:
    """Return the Python DSL function name for an operation.

    Prefer surface aliases for kinded binary and unary operations, then the
    registered operation schema name, then a snake-case class-name fallback.
    Surface aliases keep emitted source importable without enum names in scope.
    """
    if isinstance(target, HirFunction):
        return target.name
    alias_name = _kinded_alias_name(target)
    if alias_name is not None:
        return alias_name
    schema = getattr(target, "_op_schema", None)
    if schema is not None:
        return schema.name
    cls_name = type(target).__name__
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cls_name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1).lower()
    for suffix in ("_op", "_expr", "_stmt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _kinded_alias_name(target) -> str | None:
    """Return the surface alias name for a kinded ``Binary`` / ``Unary`` instance, else ``None``.

    Return the surface alias name (``add`` / ``neg`` / ...) for a
    kinded ``Binary`` / ``Unary`` instance, else ``None``.

    Per-name HIR math classes are gone; the IR instance is
    ``Binary(kind=...)`` / ``Unary(kind=...)``. Round-trip printing
    must emit the alias surface name so importing the regenerated DSL source
    uses the same alias schema.
    """
    if isinstance(target, Binary):
        kind = getattr(target, "kind", None)
        return _BINARY_KIND_TO_ALIAS.get(kind)
    if isinstance(target, Unary):
        kind = getattr(target, "kind", None)
        return _UNARY_KIND_TO_ALIAS.get(kind)
    return None


def _build_kinded_alias_maps():
    return (
        {
            BinaryKind.ADD: "add", BinaryKind.SUB: "sub", BinaryKind.MUL: "mul",
            BinaryKind.DIV: "div", BinaryKind.FLOOR_DIV: "floor_div",
            BinaryKind.MOD: "mod", BinaryKind.MIN: "min", BinaryKind.MAX: "max",
            BinaryKind.EQ: "cmp_eq", BinaryKind.NE: "cmp_ne",
            BinaryKind.LT: "cmp_lt", BinaryKind.LE: "cmp_le",
            BinaryKind.GT: "cmp_gt", BinaryKind.GE: "cmp_ge",
            BinaryKind.AND: "logical_and", BinaryKind.OR: "logical_or",
        },
        {
            UnaryKind.NEG: "neg", UnaryKind.ABS: "abs", UnaryKind.NOT: "logical_not",
            UnaryKind.EXP: "exp", UnaryKind.LOG: "log",
            UnaryKind.CEIL: "ceil", UnaryKind.ROUND: "round",
            UnaryKind.EXP2: "exp2", UnaryKind.LOG2: "log2",
        },
    )


_BINARY_KIND_TO_ALIAS, _UNARY_KIND_TO_ALIAS = _build_kinded_alias_maps()


def _op_display_name(target) -> str:
    """Display-only op name for DOT / viewer graph labels.

    Display-only op name for DOT / viewer graph labels: the target's class
    name with a trailing ``Op`` / ``Expr`` / ``Stmt`` suffix stripped
    (``MatMul``, ``TupleGetItem``, ...). Distinct from ``_op_name``, which
    renders the round-trippable DSL callable name — this one is shared by
    ``dot.py`` and ``viewer/builder.py`` for human-facing labels only.
    """
    cls = type(target).__name__
    for suffix in ("Op", "Expr", "Stmt"):
        if cls.endswith(suffix) and cls != suffix:
            cls = cls[: -len(suffix)]
    return cls


def _sanitize_name(name: str) -> str:
    """Make a Python-safe identifier from a loc string."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "v"


def _constraint_value_str(value: object) -> str:
    if is_layout_wildcard(value):
        return "_"
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value):
        return value
    return repr(value)


def _layout_constraint_str(constraint: LayoutConstraint) -> str:
    split_bindings = {
        attr.axis: (topology, attr)
        for topology, attr in constraint.bindings
        if isinstance(attr, Split)
    }
    dims: list[str] = []
    for index, extent in enumerate(constraint.layout.shape):
        if index in split_bindings:
            topology, _ = split_bindings[index]
            dims.append(
                f"{_constraint_value_str(extent)} @ "
                f"{_constraint_value_str(topology)}"
            )
        else:
            dims.append(_constraint_value_str(extent))
    dims_str = "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"
    bindings = [
        (topology, attr)
        for topology, attr in constraint.bindings
        if not isinstance(attr, Split)
    ]
    if not bindings:
        return dims_str
    binding_str = []
    for topology, attr in bindings:
        if isinstance(attr, Broadcast):
            binding_str.append(f"{_constraint_value_str(topology)} @ B()")
        elif isinstance(attr, Partial):
            binding_str.append(
                f'{_constraint_value_str(topology)} @ P("{attr.reduction}")'
            )
        else:  # pragma: no cover - LayoutConstraint validates this type
            raise TypeError(f"unsupported layout binding {type(attr).__name__}")
    return f"({dims_str}, {{{', '.join(binding_str)}}})"


def _where_str(metadata: ScheduleConstraintMetadata) -> str:
    layout = next(
        (item for item in metadata.constraints if isinstance(item, LayoutConstraint)),
        None,
    )
    fields: list[str] = []
    if layout is not None:
        fields.append(f"layout={_layout_constraint_str(layout)}")
    for item in metadata.constraints:
        if isinstance(item, MeshConstraint):
            fields.append(f"mesh={PythonPrinter().visit(item.mesh, HirPrintContext())}")
        elif isinstance(item, StorageConstraint):
            fields.append(f'storage="{item.storage.name.lower()}"')
    return "where(" + ", ".join(fields) + ")"


def _constraint_line(expr: Expr, indent: str, name: str) -> str | None:
    metadata = constraint_metadata(expr)
    if metadata is None:
        return None
    return f"{indent}{name}: {_where_str(metadata)}"


def iter_exprs(root: Expr | None, seen: set[int] | None = None) -> Iterator[Expr]:
    """Iter exprs.

    Post-order traversal of *root* and its descendants via
    ``tilefoundry.ir.visitor.expr_children`` (which, unlike the hand-rolled
    walkers this replaces, descends into ``LoopRegion``). Each node is
    yielded exactly once by object identity; *seen* lets callers share dedup
    state across repeated calls (e.g. one per function param).
    """
    if root is None:
        return
    if seen is None:
        seen = set()
    key = id(root)
    if key in seen:
        return
    seen.add(key)
    for child in expr_children(root):
        yield from iter_exprs(child, seen)
    yield root


def _region_projection(expr: Expr) -> LoopRegion | MeshRegion | None:
    """Return the region projected by a one-argument ``TupleGetItem``."""
    if not (
        isinstance(expr, Call)
        and isinstance(expr.target, TupleGetItem)
        and len(expr.args) == 1
    ):
        return None
    region = expr.args[0]
    if isinstance(region, LoopRegion):
        return region
    if isinstance(region, MeshRegion) and isinstance(region.body, Tuple):
        return region
    return None


def _module_callee_binding(target: HirFunction, child_entries: dict[int, str]) -> str | None:
    """The attribute a call on *target* was written through, if a child's entry.

    Follows the whole chain a rebuilt function records, so a target elaborated
    for its call site is still recognised. The table is keyed by the attached
    entry's identity, never by a name, which is what keeps two attached copies
    of one source Module apart.
    """
    candidate: object = target
    seen: set[int] = set()
    while isinstance(candidate, HirFunction) and id(candidate) not in seen:
        seen.add(id(candidate))
        if id(candidate) in child_entries:
            return child_entries[id(candidate)]
        candidate = origin_of(candidate)
    return None


def _emit_def(
    fn: HirFunction, def_name: str, ctx: HirPrintContext, indent: str,
    options: PythonPrintOptions, child_entries: dict[int, str] | None = None,
    *,
    line_offset: int = 0,
    statements: dict[int, _PrintedStatement] | None = None,
) -> list[str]:
    """Render one function ``def`` block: signature + body (or ``pass`` for a prototype).

    Render one function ``def`` block: signature + body (or ``pass`` for a
    prototype). The caller prepends the decorator line(s). Each call builds its
    own SSA name scope, so a base and its variants do not share names.
    *child_entries* names the attached children this body may call.
    """
    child_entries = {} if child_entries is None else child_entries
    lines: list[str] = []
    printer = HirPrinter()
    root_mesh = (
        fn.body.mesh
        if isinstance(fn.body, MeshRegion) and not fn.specializations
        else None
    )
    if root_mesh is not None:
        ctx.push_mesh(root_mesh, "mesh")

    _counter = [0]
    _names: dict[int, str] = {}



    _seen: set[int] = set()
    _order: list[Expr] = list(iter_exprs(fn.body, _seen))
    for p in fn.params:
        _order.extend(iter_exprs(p, _seen))
    for scope in tuple(expr for expr in _order if isinstance(expr, MeshRegion)):
        for param in scope.params:
            _order.extend(iter_exprs(param, _seen))
    _param_alias = {
        id(param): arg
        for scope in tuple(expr for expr in _order if isinstance(expr, MeshRegion))
        for param, arg in zip(scope.params, scope.args, strict=True)
    }


    _op_names_set: set[str] = set()
    for expr in _order:
        if isinstance(expr, Call):
            _op_names_set.add(_op_name(expr.target))

    _forced_names: dict[int, str] = {}
    _tile_window_steps: dict[int, object] = {}
    collapsed_slice_ids = {
        id(expr.args[0])
        for expr in _order
        if isinstance(expr, Call)
        and isinstance(expr.target, Reshape)
        and len(expr.args) == 1
        and isinstance(expr.args[0], Call)
        and isinstance(expr.args[0].target, Slice)
        and isinstance(expr.args[0].type, TensorType)
        and isinstance(expr.type, TensorType)
        and len(expr.type.shape) < len(expr.args[0].type.shape)
    }

    _grid_internal_ids: set[int] = set()
    _mesh_region_internal_ids: set[int] = set()
    _nested_grid_ids: set[int] = set()
    for expr in _order:
        if not isinstance(expr, LoopRegion):
            continue
        if (
            expr.start == 0
            and any(
                isinstance(candidate, Call)
                and isinstance(candidate.target, Slice)
                and id(candidate) not in collapsed_slice_ids
                and len(candidate.args) == 2
                and isinstance(candidate.args[1], Tuple)
                and any(
                    window_base(start)[0] is expr.induction_var
                    and size == expr.step
                    and stride == 1
                    for start, size, stride in zip(
                        candidate.args[1].elements,
                        candidate.target.sizes,
                        candidate.target.strides,
                    )
                )
                for candidate in _order
            )
        ):
            _tile_window_steps[id(expr.induction_var)] = expr.step
        for carry, init, value in zip(
            expr.carried_args, expr.init_args, expr.yield_values
        ):
            _forced_names[id(carry)] = _sanitize_name(carry.name)
            _forced_names[id(init)] = _sanitize_name(carry.name)
        for _ in iter_exprs(expr.body, _grid_internal_ids):
            pass
        for value in expr.yield_values:
            for _ in iter_exprs(value, _grid_internal_ids):
                pass
        for nested in iter_exprs(expr.body, set()):
            if isinstance(nested, LoopRegion) and nested is not expr:
                _nested_grid_ids.add(id(nested))
        for value in expr.yield_values:
            for nested in iter_exprs(value, set()):
                if isinstance(nested, LoopRegion) and nested is not expr:
                    _nested_grid_ids.add(id(nested))

    _root_grid_init_ids: set[int] = set()
    for expr in _order:
        if not isinstance(expr, LoopRegion) or id(expr) in _nested_grid_ids:
            continue
        for init in expr.init_args:
            for _ in iter_exprs(init, _root_grid_init_ids):
                pass
    _grid_internal_ids.difference_update(_root_grid_init_ids)

    for expr in _order:
        if isinstance(expr, MeshRegion):
            for _ in iter_exprs(expr.body, _mesh_region_internal_ids):
                pass
    def _moved_window(start, size, stride):
        """The tile window and offset *start* moves it by, else ``None``."""
        window, offset = window_base(start)
        if (
            isinstance(window, Var)
            and stride == 1
            and _tile_window_steps.get(id(window)) == size
        ):
            return window, offset
        return None


    _inlined_start_ids = {
        id(start)
        for expr in _order
        if isinstance(expr, Call)
        and isinstance(expr.target, Slice)
        and len(expr.args) == 2
        and isinstance(expr.args[1], Tuple)
        for start, size, stride in zip(
            expr.args[1].elements, expr.target.sizes, expr.target.strides
        )
        if _moved_window(start, size, stride) is not None
    }

    def _assign_name(expr: Expr) -> str:
        key = id(expr)
        if key in _names:
            return _names[key]
        if key in _forced_names:
            name = _forced_names[key]
        elif isinstance(expr, Var):
            name = _sanitize_name(expr.name)
        elif isinstance(expr, Call) and (authored_name := binding_name(expr)):
            name = _sanitize_name(authored_name)
        else:
            name = f"v{_counter[0]}"
            _counter[0] += 1
        if key in _forced_names and name in _names.values():
            _names[key] = name
            return name

        if name in _op_names_set:
            name = f"{name}_out"
        base = name
        n = 2
        while name in _names.values():
            name = f"{base}_{n}"
            n += 1
        _names[key] = name
        return name


    for expr in _order:
        _assign_name(expr)
    for expr in _order:
        if isinstance(expr, LoopRegion):
            for carry in expr.carried_args:
                _assign_name(carry)
    printer.bind_def(_names, _param_alias, child_entries, _moved_window)

    return_type = fn.return_type
    arrow = ""
    if isinstance(return_type, TensorType):
        with printer.type_surface(indent=indent):
            arrow = " -> " + printer.visit(return_type, ctx)
    elif not isinstance(return_type, TupleType):
        arrow = " -> None"

    lines.append(f"def {def_name}(")
    params: list[str] = []
    for param in fn.params:
        name = _names[id(param)]
        if isinstance(param.type, TensorType):
            with printer.type_surface(indent=indent, const=param.is_const):
                annotation = printer.visit(param.type, ctx)
            params.append(f"{indent}{name}: {annotation}")
        else:
            params.append(f"{indent}{name}")
    for index, text in enumerate(params):
        suffix = "," if index < len(params) - 1 else ""
        lines.extend((text + suffix).split("\n"))
    lines.append(f"){arrow}:")

    for param in fn.params:
        line = _constraint_line(param, indent, _names[id(param)])
        if line is not None:
            lines.append(line)
    if fn.body is None:
        lines.append(f"{indent}pass")
        if root_mesh is not None:
            ctx.pop_mesh()
        return lines

    printed: set[int] = {id(param) for param in fn.params}

    def _emit_inline_call(expr: Call, level: str) -> None:
        name = _names[id(expr)]
        if statements is not None:
            statements[id(expr)] = _PrintedStatement(
                value=name,
                line=line_offset + _physical_line_count(lines) + 1,
            )
        with printer.type_surface(indent=level):
            rendered = printer.visit(expr, ctx)
        lines.append(
            f"{level}{name} = {rendered}"
            f"{_comments(expr, options, printer, ctx)}"
        )
        printed.add(id(expr))

    def _emit_expr(expr: Expr, level: str) -> None:
        key = id(expr)
        if key in printed:
            return
        if key in _inlined_start_ids:
            printed.add(key)
            return
        if isinstance(expr, Var):
            printed.add(key)
            return
        if isinstance(expr, Constant):
            lines.append(
                f"{level}{_names[id(expr)]} = {repr(expr.value)}"
                f"{_comments(expr, options, printer, ctx)}"
            )
            printed.add(key)
            return
        if isinstance(expr, Tuple):
            for element in expr.elements:
                if not isinstance(element, Constant):
                    _emit_expr(element, level)
            printed.add(key)
            return
        if isinstance(expr, LoopRegion):
            _emit_loop_region(expr, level)
            return
        if isinstance(expr, MeshRegion):
            _emit_mesh_region(expr, level)
            return
        if isinstance(expr, Call):
            projection = _region_projection(expr)
            if isinstance(projection, LoopRegion):
                _emit_loop_region(projection, level)
                printed.add(key)
                return
            if isinstance(projection, MeshRegion):
                _emit_mesh_region(projection, level)
                printed.add(key)
                return
            for arg in expr.args:
                _emit_expr(arg, level)
            _emit_inline_call(expr, level)
            return
        printed.add(key)

    def _emit_loop_region(region: LoopRegion, level: str) -> None:
        key = id(region)
        if key in printed:
            return
        for init in region.init_args:
            _emit_expr(init, level)
        for carry in region.carried_args:
            printed.add(id(carry))
        extent = printer.visit(region.extent, ctx)
        step = printer.visit(region.step, ctx)
        start = printer.visit(region.start, ctx)
        if id(region.induction_var) in _tile_window_steps:
            ctx.imports.add("from tilefoundry.dsl.tf import *")
            loop = f"tile({extent}, {step})"
        elif region.start == 0 and region.step == 1:
            loop = f"range({extent})"
        else:
            loop = f"range({start}, {extent}, {step})"
        lines.append(f"{level}for {region.induction_var.name} in {loop}:{_comments(region, options, printer, ctx)}")
        printed.add(key)
        inner = level + "    "
        _emit_expr(region.body, inner)
        for value in region.yield_values:
            _emit_expr(value, inner)
        for carry, value in zip(region.carried_args, region.yield_values):
            lines.append(f"{inner}{_names[id(carry)]} = {printer.reference(value)}")

    def _emit_mesh_region(region: MeshRegion, level: str, *, terminal: bool = False) -> None:
        key = id(region)
        if key in printed:
            return
        for arg in region.args:
            _emit_expr(arg, level)
        if root_mesh is not None and region is fn.body:
            printed.add(key)
            _emit_expr(region.body, level)
            if terminal:
                lines.append(f"{level}return {printer.reference(region.body)}")
            return
        mesh_text = printer.visit(region.mesh, ctx)
        mesh_name = ctx.scope_name(region.mesh)
        lines.append(
            f"{level}with {mesh_text} as {mesh_name}:"
            f"{_comments(region, options, printer, ctx)}"
        )
        printed.add(key)
        inner = level + "    "
        ctx.push_mesh(region.mesh, mesh_name)
        try:
            if terminal and isinstance(region.body, MeshRegion):
                _emit_mesh_region(region.body, inner, terminal=True)
                return
            _emit_expr(region.body, inner)
            if terminal:
                lines.append(f"{inner}return {printer.reference(region.body)}")
        finally:
            ctx.pop_mesh()

    for expr in _order:
        if (
            isinstance(expr, Var)
            or id(expr) in _grid_internal_ids
            or id(expr) in _mesh_region_internal_ids
        ):
            continue
        if id(expr) in _inlined_start_ids:
            printed.add(id(expr))
            continue
        if isinstance(expr, LoopRegion):
            _emit_loop_region(expr, indent)
            continue
        if isinstance(expr, MeshRegion):
            _emit_mesh_region(expr, indent, terminal=expr is fn.body)
            continue
        if _region_projection(expr) is not None:
            printed.add(id(expr))
            continue
        if isinstance(expr, Constant):
            name = _names[id(expr)]
            lines.append(f"{indent}{name} = {repr(expr.value)}{_comments(expr, options, printer, ctx)}")
            line = _constraint_line(expr, indent, name)
            if line is not None:
                lines.append(line)
            printed.add(id(expr))
            continue
        if isinstance(expr, Tuple):




            continue
        if isinstance(expr, Call):
            name = _names[id(expr)]
            if statements is not None:
                statements[id(expr)] = _PrintedStatement(
                    value=name,
                    line=line_offset + _physical_line_count(lines) + 1,
                )
            with printer.type_surface(indent=indent):
                rendered = printer.visit(expr, ctx)
            lines.append(
                f"{indent}{name} = {rendered}"
                f"{_comments(expr, options, printer, ctx)}"
            )
            line = _constraint_line(expr, indent, name)
            if line is not None:
                lines.append(line)
            printed.add(id(expr))



    if not isinstance(fn.body, MeshRegion):
        if isinstance(fn.body, Tuple):
            lines.append(f"{indent}return {printer.tuple_reference(fn.body.elements)}")
        elif isinstance(fn.body, LoopRegion):
            values = tuple(_names[id(carry)] for carry in fn.body.carried_args)
            result = values[0] if len(values) == 1 else "(" + ", ".join(values) + ")"
            lines.append(f"{indent}return {result}")
        else:
            body_name = printer.reference(fn.body)
            lines.append(f"{indent}return {body_name}")
    if root_mesh is not None:
        ctx.pop_mesh()
    return lines



def _new_hir_context(*, for_module: bool = False, target=None) -> HirPrintContext:
    """Create a HIR context with imports owned by the surrounding file."""
    ctx = HirPrintContext()
    if for_module:
        ctx.imports.add("from tilefoundry.module import module")
    ctx.imports.add("from tilefoundry import func")
    if target is not None:
        rendered = target.to_python()
        ctx.imports.update(rendered.imports)
    return ctx

def _variant_binding_name(variant: HirFunction) -> str:
    """Return a valid source binding for a variant without display metadata."""
    label = display_name(variant)
    if label is not None:
        return label
    signature = canonical_specialization_signature(variant.specializations)
    return "variant_" + re.sub(r"[^0-9A-Za-z_]", "_", signature)


def _emit_decorated_defs(
    fn: HirFunction, ctx: HirPrintContext, indent: str, options: PythonPrintOptions,
    child_entries: dict[int, str] | None = None,
    *,
    line_offset: int = 0,
    statements: dict[int, _PrintedStatement] | None = None,
) -> list[str]:
    """Emit decorated defs.

    Emit a base ``@func`` definition followed by one specialization block per
    variant. Standalone and module output share this path so dispatch prototypes
    render identically.
    See [inspection §2.6](docs/spec/inspection.md#26-specialization-printing).
    """
    printer = HirPrinter()
    decorator = "@func"
    if isinstance(fn.body, MeshRegion) and not fn.specializations:
        root_ctx = HirPrintContext()
        with printer.type_surface(indent=indent):
            mesh_text = printer.visit(fn.body.mesh, root_ctx)
        ctx.imports.update(root_ctx.imports)
        decorator = f"@func(mesh={mesh_text})"
    lines: list[str] = [decorator]
    lines.extend(
        _emit_def(
            fn,
            fn.name,
            ctx,
            indent,
            options,
            child_entries,
            line_offset=line_offset + _physical_line_count(lines),
            statements=statements,
        )
    )


    for variant in fn.variants:
        lines.append("")
        lines.append(
            f"@{fn.name}.specialize({HirPrinter().render_pattern(variant.specializations[0], ctx)})"
        )
        lines.extend(
            _emit_def(
                variant, _variant_binding_name(variant), ctx, indent, options,
                child_entries,
                line_offset=line_offset + _physical_line_count(lines),
                statements=statements,
            )
        )
    return lines


def _render_hir_function(
    fn: HirFunction, *, options: PythonPrintOptions | None = None,
) -> _PythonRendering:
    """Render a HIR Function and locate every Call equation in the same pass.

    A normal function prints as a single ``@func``. A dispatch prototype
    (``variants != ()``) prints as a ``pass``-bodied ``@func`` base followed by
    one ``@<name>.specialize(pattern)`` block per variant. Placement sugar is
    emitted only where an explicit mesh-scope binding dominates its use.
    """
    indent = "    "
    ctx = _new_hir_context()
    statements: dict[int, _PrintedStatement] = {}
    lines = _emit_decorated_defs(
        fn,
        ctx,
        indent,
        options or PythonPrintOptions(),
        line_offset=0,
        statements=statements,
    )
    header = ctx.header()
    header_lines = _physical_line_count(header)
    statements = {
        identity: _PrintedStatement(value=item.value, line=item.line + header_lines)
        for identity, item in statements.items()
    }
    return _PythonRendering("\n".join(header + lines) + "\n", statements)


def hir_function_to_python(
    fn: HirFunction, *, options: PythonPrintOptions | None = None,
) -> str:
    """Convert a HIR Function to canonical Python DSL source."""
    return HirPrinter().print(fn, options=options)


def as_script(
    fn: HirFunction | PrimFunction | Module, *, module: str | None = None,
    options: PythonPrintOptions | None = None,
) -> str:
    """Convert an HIR function or module to Python DSL source.

    Without *module*, emit a standalone decorated function. With *module*, emit
    a named module-class wrapper with entry and mesh definitions. *options*
    controls canonical-source rendering.
    """
    if isinstance(fn, Module):
        if fn.functions and all(isinstance(item, PrimFunction) for item in fn.functions):
            return tir_module_to_python(fn, module, options=options)
        return _module_to_python(fn, module, options=options)
    if isinstance(fn, PrimFunction) and module is None:
        return tir_function_to_python(fn, options=options)
    if module is not None:
        if isinstance(fn, PrimFunction):
            return tir_module_to_python(Module(name=module, functions=(fn,), entry=fn.name), options=options)
        return _module_to_python(fn, module, options=options)
    return hir_function_to_python(fn, options=options)


def module_to_python(fn: HirFunction, module_name: str = "M") -> str:
    """Backward-compat alias for ``as_script(fn, module=module_name)``."""
    return as_script(fn, module=module_name)


def _module_hir_functions(mod: Module) -> tuple[HirFunction, ...]:
    """The Module's HIR functions."""
    return tuple(fn for fn in _emission_order(mod) if isinstance(fn, HirFunction))


def _emission_order(mod: Module) -> tuple:
    """A Module's functions in the order the printed class body binds them.

    The entry goes last: a body calling a sibling names the attribute the class
    body already bound, so every callee must be written before it. Mesh
    traversal reads the same order, so output does not depend on the order the
    authored source happened to use.
    """
    functions = mod.functions
    entry = mod.entry_function() if functions and mod.entry is not None else None
    ordered = tuple(fn for fn in functions if fn is not entry)
    return ordered + (entry,) if entry is not None else ordered


def _module_tree_functions(mod: Module) -> tuple[HirFunction, ...]:
    """Every HIR function owned by *mod* or any Module beneath it."""
    functions = list(_module_hir_functions(mod))
    for child in mod.modules:
        functions.extend(_module_tree_functions(child))
    return tuple(functions)


def _module_decorator_line(mod: Module, entry_name: str | None, ctx: HirPrintContext) -> str:
    """Render the context this Module declares as an ``@module(...)`` line.

    Always the called form. A bare decorator has not run while the class body
    is evaluated, so a body naming a child call could not resolve it.
    """
    kwargs: list[str] = [] if entry_name is None else [f'entry="{entry_name}"']
    printer = HirPrinter()
    if mod.target is not None:
        rendered: PythonExpr = mod.target.to_python()
        ctx.imports.update(rendered.imports)
        kwargs.append(f"target={rendered.text}")
    if mod.topologies is not None:
        ctx.imports.add("from tilefoundry.ir.types.shard import Topology")
        topo_strs = [
            f'Topology("{t.name}", {printer.visit(t.size, ctx)})'
            for t in mod.topologies
        ]
        rendered_topologies = f'({", ".join(topo_strs)},)' if topo_strs else "()"
        kwargs.append(f"topologies={rendered_topologies}")
    return f"@module({', '.join(kwargs)})"


def _emit_module_class(
    mod: Module, module_name: str, ctx: HirPrintContext, indent: str,
    options: PythonPrintOptions,
) -> list[str]:
    """One ``@module`` class block: its nested Modules, then its functions.

    Children first, because a body calling one names the attribute it is bound
    to and a class body binds in the order it is written.
    """
    lines = [_module_decorator_line(mod, mod.entry, ctx), f"class {module_name}:"]
    ordered = _emission_order(mod)
    child_entries = {
        id(child.entry_function()): child.name
        for child in mod.modules
        if child.entry is not None and isinstance(child.entry_function(), HirFunction)
    }
    blocks: list[list[str]] = [
        _emit_module_class(child, child.name, ctx, indent, options)
        for child in mod.modules
    ]
    for fn in ordered:
        if isinstance(fn, HirFunction):
            blocks.append(_emit_decorated_defs(fn, ctx, indent, options, child_entries))
        elif isinstance(fn, PrimFunction):
            tir_block = _tir_function_block(fn)
            ctx.imports.update(tir_block.imports)
            blocks.append(tir_block)
        else:
            raise TypeError(f"Python printer cannot serialize {type(fn).__name__}")
    for index, block in enumerate(blocks):
        if index:
            lines.append("")
        lines.extend(f"{indent}{ln}" if ln else ln for ln in block)
    return lines


def _module_to_python(
    fn_or_module: HirFunction | Module, module_name: str | None = None,
    *, options: PythonPrintOptions | None = None,
) -> str:
    """Render a function or a whole Module tree as ``@module`` source."""
    if isinstance(fn_or_module, Module):
        root = fn_or_module
        module_name = root.name if module_name is None else module_name
    else:
        root = Module(
            name="M" if module_name is None else module_name,
            functions=(fn_or_module,),
            entry=fn_or_module.name,
        )
        module_name = root.name
    functions = _module_tree_functions(root)
    if not functions:
        raise TypeError("HIR Module printer requires at least one HIR function")
    entry = root.entry_function() if root.entry is not None else None
    if entry is not None and not isinstance(entry, (HirFunction, PrimFunction)):
        raise TypeError("Module printer requires a function entry")


    indent4 = "    "
    ctx = _new_hir_context(for_module=True, target=root.target)
    lines = _emit_module_class(
        root, module_name, ctx, indent4, options or PythonPrintOptions(),
    )
    header = ctx.header()
    return "\n".join(header + lines) + "\n"


def _module_tree(root: Module) -> Iterator[Module]:
    yield root
    for child in root.modules:
        yield from _module_tree(child)
