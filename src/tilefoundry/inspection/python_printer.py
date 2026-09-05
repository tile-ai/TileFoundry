"""Canonical Python DSL printer for HIR Functions.

Converts a ``hir.Function`` to executable Python source using the
``@func`` DSL.  When meshes have named axes, compact sugar annotations
are emitted; otherwise the verbose ``ShardLayout(...)`` form is used.
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
    dim_vars_reached,
    display_name,
    origin_of,
)
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice, window_base
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimConst,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
)
from tilefoundry.ir.types.shard import try_c_order_strides
from tilefoundry.ir.types.shard.layout import Layout, LayoutBase
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.ir.types.substitute import dim_vars_by_name
from tilefoundry.ir.visitor import ExprFunctor, expr_children
from tilefoundry.utils.python_source import PythonExpr

from .print_context import HirPrintContext
from .printer_base import PythonPrinter
from .tir_printer import _function_block as _tir_function_block
from .tir_printer import tir_function_to_python, tir_module_to_python
from .values import PARTS, render_comment

_DIM_INFIX_OPS: dict[type, str] = {
    DimAdd: "+",
    DimSub: "-",
    DimMul: "*",
    DimFloorDiv: "//",
    DimMod: "%",
}


_DIM_FUNC_OPS: dict[type, str] = {
    DimMin: "min",
    DimMax: "max",
}


class HirPrinter(PythonPrinter):
    """HIR façade retaining the historical canonical rendering entry point."""

    def print(self, fn: HirFunction, *, options=None) -> str:
        return _render_hir_function(fn, options=options).source

    def dim_entry(self, value, ctx=None) -> str:
        return shape_entry_str(value)

    def shard_surface(self, value, ctx=None):
        mesh_map = getattr(ctx, "mesh_name_map", {}) if ctx is not None else {}
        mesh_name = ctx.mesh_alias(value.mesh) if ctx is not None else None
        if mesh_name is None or not value.mesh.names:
            return None
        return _shard_layout_surface_str(
            value, mesh_name=mesh_name, mesh_unique=len(mesh_map) == 1
        )


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


def _compact_type(ty: object, mesh_name_map: dict[int, str]) -> str:
    """One physical-line, DSL-shaped type annotation for inspection output.

    Takes the same mesh-name map the signature and mesh prelude are rendered
    from, so an annotated layout names the hoisted mesh instead of restating it.
    """
    if isinstance(ty, TensorType):
        rendered = _tensor_annotation(ty, mesh_name_map=mesh_name_map)
        return " ".join(rendered.split())
    if isinstance(ty, TupleType):
        fields = ", ".join(_compact_type(field, mesh_name_map) for field in ty.fields)
        return f"Tuple[{fields}]"
    return repr(ty)


def _comments(expr: Expr, options: PythonPrintOptions, mesh_name_map: dict[int, str]) -> str:
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
        comments.append(_compact_type(expr.type, mesh_name_map))
    for metadata_type in options.comment_metadata_types:
        metadata = get_metadata(expr, metadata_type)
        if metadata is None:
            continue
        comment = render_comment(metadata, opt_in=options.comment_opt_in)
        if comment is not None:
            comments.append(comment)
    return f"  # {PARTS.join(comments)}" if comments else ""


def shape_entry_str(entry: object) -> str:
    """Render one tensor shape entry in canonical human-readable form.

    Static integers remain literals, dimension variables use their names, and
    arithmetic expression trees use infix or function syntax. The printer and
    viewer share this rendering instead of exposing dataclass representations.
    See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions) and
    [inspection §2.3](docs/spec/inspection.md#23-dsl-text-forms).
    """
    return _shape_entry_str(entry, nested=False)


class _ShapeEntryVisitor(ExprFunctor[str]):
    def __init__(self, nested: bool) -> None:
        super().__init__()
        self.nested = nested

    def visit_DimVar(self, entry: DimVar, ctx=None) -> str:
        return entry.name

    def visit_Var(self, entry: Var, ctx=None) -> str:
        return entry.name

    def visit_Constant(self, entry: Constant, ctx=None) -> str:
        return str(entry.value)

    def visit_Call(self, entry: Call, ctx=None) -> str:
        ceildiv_args = _ceildiv_args(entry)
        if ceildiv_args is not None:
            a, b = ceildiv_args
            return f"ceildiv({self._render(a, False)}, {self._render(b, False)})"
        target = entry.target
        if isinstance(target, DimConst):
            return str(target.value)
        for op_cls, sym in _DIM_INFIX_OPS.items():
            if isinstance(target, op_cls):
                a, b = entry.args
                rendered = f"{self._render(a, True)} {sym} {self._render(b, True)}"
                return f"({rendered})" if self.nested else rendered
        for op_cls, fname in _DIM_FUNC_OPS.items():
            if isinstance(target, op_cls):
                rendered = ", ".join(self._render(arg, False) for arg in entry.args)
                return f"{fname}({rendered})"
        return repr(entry)

    def default_visit(self, entry, ctx=None) -> str:
        if isinstance(entry, bool):
            return repr(entry)
        if isinstance(entry, int):
            return str(entry)
        return repr(entry)

    def _render(self, entry, nested: bool) -> str:
        previous = self.nested
        self.nested = nested
        try:
            return self.visit(entry, None)
        finally:
            self.nested = previous


def _shape_entry_str(entry: object, *, nested: bool) -> str:
    return _ShapeEntryVisitor(nested).visit(entry)


def _ceildiv_args(entry: Call) -> tuple[object, object] | None:
    """Recover the public constructor from ceildiv's canonical arithmetic tree."""
    if not isinstance(entry.target, DimFloorDiv) or len(entry.args) != 2:
        return None
    numerator, divisor = entry.args
    if not (
        isinstance(numerator, Call)
        and isinstance(numerator.target, DimSub)
        and len(numerator.args) == 2
        and isinstance(numerator.args[1], Constant)
        and numerator.args[1].value == 1
    ):
        return None
    added = numerator.args[0]
    if not (
        isinstance(added, Call)
        and isinstance(added.target, DimAdd)
        and len(added.args) == 2
        and added.args[1] == divisor
    ):
        return None
    return added.args[0], divisor


def _classify_shard_attrs(
    sl: ShardLayout, mesh_name: str
) -> tuple[dict[int, list[str]], list[str]] | None:
    """Classify shard attributes into layout-axis splits and partials.

    Preserve mesh-axis order, allow nested axes to split one layout axis, and
    omit broadcasts. Return ``None`` for rank mismatch, invalid axes, or unknown
    attributes so callers use verbose fallback. Surface and compact renderers
    share the result, with the latter remapping splits onto tensor axes.
    """
    layout = sl.layout
    if not isinstance(layout, Layout) or len(sl.attrs) != len(sl.mesh.layout.shape):
        return None
    layout_rank = len(layout.shape)
    names = sl.mesh.names if hasattr(sl.mesh, "names") and sl.mesh.names else ()
    splits: dict[int, list[str]] = {}
    partials: list[str] = []
    for mesh_axis_idx, attr in enumerate(sl.attrs):
        axis_name = names[mesh_axis_idx] if mesh_axis_idx < len(names) else f"ax{mesh_axis_idx}"
        axis_ref = f"{mesh_name}.{axis_name}"
        if isinstance(attr, Split):
            if attr.axis >= layout_rank:
                return None
            splits.setdefault(attr.axis, []).append(axis_ref)
        elif isinstance(attr, Partial):
            partials.append(f'{axis_ref} @ P("{attr.reduction or "sum"}")')
        elif not isinstance(attr, Broadcast):
            return None
    return splits, partials


def _shard_layout_surface_str(
    sl: ShardLayout,
    mesh_name: str = "gpu",
    *,
    mesh_unique: bool = False,
) -> str | None:
    """Render canonical parser sugar for a shard layout.

    Inline splits on layout dimensions, emit partial value states as a set, and
    omit broadcasts. Include explicit strides only when present. Return ``None``
    when sugar cannot express the layout so callers use verbose fallback.

    A symbolic shape has no static C-order strides to compare against, so the
    ones it states are emitted rather than assumed contiguous.
    """
    layout = sl.layout
    if not isinstance(layout, Layout):
        return None
    classified = _classify_shard_attrs(sl, mesh_name)
    if classified is None:
        return None
    splits, partials = classified

    if not splits and not partials and not mesh_unique:
        return None

    c_strides = try_c_order_strides(layout.shape)
    explicit = layout.strides is not None and layout.strides != c_strides
    if explicit and any(
        i in splits and _shape_entry_str(dim, nested=True) != shape_entry_str(dim)
        for i, dim in enumerate(layout.shape)
    ):
        return None

    dims = [
        f"{_shape_entry_str(d, nested=True)} {' '.join(f'@ {r}' for r in splits[i])}"
        if i in splits
        else shape_entry_str(d)
        for i, d in enumerate(layout.shape)
    ]
    dim_str = ", ".join(dims)
    if len(dims) == 1:
        dim_str += ","
    axis_tuple = f"({dim_str})"

    stride_str = _shape_tuple(layout.strides) if explicit else None
    value_set = "{" + ", ".join(partials) + "}" if partials else None

    if stride_str is None and value_set is None:
        return axis_tuple
    parts = [axis_tuple]
    if stride_str is not None:
        parts.append(stride_str)
    if value_set is not None:
        parts.append(value_set)
    return "(" + ", ".join(parts) + ")"


def shard_compact_inline(
    sl: ShardLayout, mesh_name: str, tensor_shape: tuple
) -> tuple[dict[int, str], list[str]] | None:
    """Decompose a shard layout for compact tensor-axis display.

    Map splits to tensor axes, collect ordered partial states, and omit
    broadcasts. Return ``None`` for ambiguous split positions, invalid axes,
    unknown attributes, or rank mismatch so callers fall back to canonical
    rendering. Attribute classification is shared with surface rendering.
    """
    layout = sl.layout
    if not isinstance(layout, Layout):
        return None
    classified = _classify_shard_attrs(sl, mesh_name)
    if classified is None:
        return None
    splits, partials = classified
    la2ta = layout_axis_to_tensor_axis(layout.shape, tensor_shape)
    split_ref: dict[int, str] = {}
    for layout_axis, refs in splits.items():
        if len(refs) != 1:
            return None
        t_axis = la2ta[layout_axis]
        if t_axis in split_ref:
            return None
        split_ref[t_axis] = refs[0]
    return split_ref, partials


def _moved_window_ref(name: str, offset: int) -> str:
    """A tile-window indexer, carrying the compile-time offset that moves it."""
    if offset == 0:
        return name
    return f"{name} + {offset}" if offset > 0 else f"{name} - {-offset}"


def _attr_tuple_str(value: tuple) -> str:
    """Render an attribute's tuple value as a Python tuple literal.

    A shape-valued attribute -- `new_shape`, a tile's extents -- can hold a
    `DimVar` or dim arithmetic, and the tuple's own `str` would render those as
    dataclass reprs. Printing them the way the annotations do keeps one program
    described one way, and keeps the printed source importable: the declaration
    the header emits binds the name, not the repr.
    """
    rendered = tuple(
        shape_entry_str(entry) if _is_dim_entry(entry) else repr(entry)
        for entry in value
    )
    if len(rendered) == 1:
        return f"({rendered[0]},)"
    return "(" + ", ".join(rendered) + ")"


def _is_dim_entry(entry: object) -> bool:
    """Whether *entry* is a dimension rather than a plain attribute value."""
    return isinstance(entry, (DimVar, Var, Constant)) or (
        isinstance(entry, Call)
        and isinstance(entry.target, (DimConst, *_DIM_INFIX_OPS, *_DIM_FUNC_OPS))
    )


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
            fields.append(f"mesh={_mesh_str(item.mesh)}")
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


def _collect_meshes(
    fn: HirFunction,
    *,
    include_node_types: bool = False,
) -> tuple[dict[int, Mesh], dict[int, Mesh]]:
    """Collect meshes needed before emitting a function header.

    The function header declares every mesh and builds the name map used by
    parameter annotations and region bodies, so collection must precede body
    emission. It walks params, return type, and body references once; the
    optional node-type scan is used by the viewer for intermediate annotations.

    With ``include_node_types=True`` (the viewer's wider scan, via
    ``viewer.builder._collect_view_meshes``) every node's own result type is
    also walked, since the viewer renders shard sugar on intermediate types too.
    """
    type_meshes: dict[int, Mesh] = {}
    scope_meshes: dict[int, Mesh] = {}

    def _add_layout(layout) -> None:
        if isinstance(layout, ShardLayout):
            type_meshes.setdefault(id(layout.mesh), layout.mesh)

    def _add_type(ty) -> None:
        if isinstance(ty, TensorType):
            _add_layout(ty.layout)
        elif isinstance(ty, TupleType):
            for f in ty.fields:
                _add_type(f)

    for p in fn.params:
        _add_type(p.type)
    _add_type(fn.return_type)

    expressions = tuple(iter_exprs(fn.body))
    for expr in expressions:
        if include_node_types:
            _add_type(getattr(expr, "type", None))
        if isinstance(expr, Call) and isinstance(expr.target, Reshard):
            _add_layout(expr.target.layout)
        if isinstance(expr, MeshRegion):
            scope_meshes.setdefault(id(expr.mesh), expr.mesh)

    return type_meshes, scope_meshes


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
    fn: HirFunction, def_name: str, mesh_map: dict[int, str], indent: str,
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

    def _tuple_literal(elements) -> str:
        inner = ", ".join(
            repr(el.value) if isinstance(el, Constant) else _expr_ref(el)
            for el in elements
        )
        if len(elements) == 1:
            inner += ","
        return f"({inner})"

    def _expr_ref(expr: Expr) -> str:
        if id(expr) in _param_alias:
            return _expr_ref(_param_alias[id(expr)])
        projection = _region_projection(expr)
        if isinstance(projection, LoopRegion):
            return _names[id(projection.carried_args[expr.target.index])]
        if isinstance(expr, LoopRegion):



            carried = tuple(_names[id(carry)] for carry in expr.carried_args)
            if len(carried) == 1:
                return carried[0]
            return "(" + ", ".join(carried) + ")"
        if isinstance(projection, MeshRegion):
            return _expr_ref(projection.body.elements[expr.target.index])
        if isinstance(expr, MeshRegion):
            return _arg_ref(expr.body)
        return _names[id(expr)]

    def _arg_ref(a) -> str:



        return _tuple_literal(a.elements) if isinstance(a, Tuple) else _expr_ref(a)

    def _start_ref(start, size, stride) -> str:
        """One Slice start as source.

        A moved tile window prints as the move itself -- the offset is a
        compile-time constant, so it belongs in the indexer rather than in a
        statement of its own.
        """
        moved = _moved_window(start, size, stride)
        if moved is None:
            return repr(start.value) if isinstance(start, Constant) else _expr_ref(start)
        window, offset = moved
        return _moved_window_ref(_expr_ref(window), offset)



    return_ty = fn.return_type
    arrow = ""
    if isinstance(return_ty, TensorType):
        arrow = " -> " + _tensor_annotation(
            return_ty, mesh_name_map=mesh_map, indent=indent
        )
    elif not isinstance(return_ty, TupleType):
        arrow = " -> None"

    lines.append(f"def {def_name}(")
    param_strs = []
    for p in fn.params:
        name = _names[id(p)]
        if isinstance(p.type, TensorType):
            ann = _tensor_annotation(
                p.type, mesh_name_map=mesh_map, indent=indent, is_const=p.is_const,
            )
            param_strs.append(f"{indent}{name}: {ann}")
        else:
            param_strs.append(f"{indent}{name}")


    for index, text in enumerate(param_strs):
        suffix = "," if index < len(param_strs) - 1 else ""
        lines.extend((text + suffix).split("\n"))
    lines.append(f"){arrow}:")

    for param in fn.params:
        line = _constraint_line(param, indent, _names[id(param)])
        if line is not None:
            lines.append(line)


    if fn.body is None:
        lines.append(f"{indent}pass")
        return lines

    printed: set[int] = {id(param) for param in fn.params}

    def _format_call(expr: Call, indent_here: str) -> str:
        """Render a Call's RHS expression text.

        Render a Call's RHS expression text: the ``reshard(...)`` /
        ``<HirFunction>(...)`` special forms, else ``op_name(args, attr=val,
        ...)``. Shared by the inline (tile-loop body) emitter and the
        top-level emit loop so an attribute-rendering rule (``ShardLayout``,
        ``DType``, ...) only needs one edit. A reshard that gathers back to the
        whole names no mesh, so its target is a plain ``Layout`` and there is no
        mesh reference to abbreviate.
        """
        target = expr.target
        args_str = ", ".join(_arg_ref(arg) for arg in expr.args)
        if isinstance(target, Reshard):
            layout_kw = ""
            if isinstance(target.layout, ShardLayout):
                layout_text = _shard_layout_str(
                    target.layout,
                    indent=indent_here + "    ",
                    mesh_ref=(
                        mesh_map.get(id(target.layout.mesh))
                        if target.layout.mesh.names
                        else None
                    ),
                )
                layout_kw = ", layout=" + layout_text
            elif target.layout is not None:
                layout_kw = ", layout=" + _layout_str(target.layout, indent_here + "    ")
            storage = (
                f", storage={target.storage.name.lower()}"
                if target.storage is not None
                else ""
            )
            return f"reshard({args_str}{layout_kw}{storage})"
        if isinstance(target, HirFunction):
            binding = _module_callee_binding(target, child_entries)
            return f"{binding or target.name}({args_str})"
        if isinstance(target, Slice):
            indexers = []
            starts = expr.args[1]
            if not isinstance(starts, Tuple):
                raise ValueError("canonical_source: Slice starts must be a Tuple")
            runtime_starts = False
            for axis, (start, size, stride) in enumerate(
                zip(starts.elements, target.sizes, target.strides)
            ):
                if _moved_window(start, size, stride) is not None:
                    indexers.append(_start_ref(start, size, stride))
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
                start_refs = ", ".join(
                    _start_ref(start, size, stride)
                    for start, size, stride in zip(
                        starts.elements, target.sizes, target.strides
                    )
                )
                if len(starts.elements) == 1:
                    start_refs += ","
                return (
                    f"slice({_arg_ref(expr.args[0])}, ({start_refs}), "
                    f"sizes={_attr_tuple_str(target.sizes)}, "
                    f"strides={_attr_tuple_str(target.strides)})"
                )
            return f"{_arg_ref(expr.args[0])}[{', '.join(indexers)}]"

        alias_name = _kinded_alias_name(target)
        suppress_attrs = {"kind"} if alias_name is not None else set()
        attr_strs = []
        for param in type(target).params():
            if param.kind != "attribute":
                continue
            value = getattr(target, param.name, None)
            if value is None or param.name in suppress_attrs or param.name == "layout":
                continue
            if isinstance(value, str):
                attr_strs.append(f'{param.name}="{value}"')
            elif isinstance(value, DType):
                attr_strs.append(f'{param.name}="{value.name}"')
            elif isinstance(value, enum.Enum) and isinstance(value.value, str):
                attr_strs.append(f'{param.name}="{value.value}"')
            elif isinstance(value, float):
                if math.isinf(value):
                    literal = "-1e999" if value < 0 else "1e999"
                elif math.isnan(value):
                    literal = "(1e999 - 1e999)"
                else:
                    literal = repr(value)
                attr_strs.append(f"{param.name}={literal}")
            elif isinstance(value, ShardLayout):
                sl_str = _shard_layout_str(value, indent=indent_here + "        ")
                attr_strs.append(f"{param.name}={sl_str}")
            elif isinstance(value, TensorType):
                attr_strs.append(f"{param.name}={_compact_type(value, {})}")
            elif isinstance(value, tuple):
                attr_strs.append(f"{param.name}={_attr_tuple_str(value)}")
            else:
                attr_strs.append(f"{param.name}={value}")



        arguments = [_arg_ref(arg) for arg in expr.args] + attr_strs
        return f"{_op_name(target)}({', '.join(arguments)})"

    def _emit_inline_call(expr: Call, level: str) -> None:
        name = _names[id(expr)]
        if statements is not None:
            statements[id(expr)] = _PrintedStatement(
                value=name,
                line=line_offset + _physical_line_count(lines) + 1,
            )
        lines.append(
            f"{level}{name} = {_format_call(expr, level)}"
            f"{_comments(expr, options, mesh_map)}"
        )
        printed.add(id(expr))

    class _ExprEmitter(ExprFunctor[None]):
        def __init__(self, level: str) -> None:
            super().__init__()
            self.level = level

        def emit(self, expr: Expr) -> None:
            self.visit(expr)

        def visit(self, expr, ctx=None):
            key = id(expr)
            if key in printed:
                return None
            if key in _inlined_start_ids:
                printed.add(key)
                return None
            return super().visit(expr, ctx)

        def visit_Var(self, expr: Var, ctx=None) -> None:
            printed.add(id(expr))

        def visit_Constant(self, expr: Constant, ctx=None) -> None:
            lines.append(
                f"{self.level}{_names[id(expr)]} = {repr(expr.value)}"
                f"{_comments(expr, options, mesh_map)}"
            )
            printed.add(id(expr))

        def visit_Tuple(self, expr: Tuple, ctx=None) -> None:
            for element in expr.elements:
                if not isinstance(element, Constant):
                    self.visit(element, ctx)
            printed.add(id(expr))

        def visit_LoopRegion(self, expr: LoopRegion, ctx=None) -> None:
            _emit_loop_region(expr, self.level)

        def visit_MeshRegion(self, expr: MeshRegion, ctx=None) -> None:
            _emit_mesh_region(expr, self.level)

        def visit_Call(self, expr: Call, ctx=None) -> None:
            projection = _region_projection(expr)
            if isinstance(projection, LoopRegion):
                _emit_loop_region(projection, self.level)
                printed.add(id(expr))
                return
            if isinstance(projection, MeshRegion):
                _emit_mesh_region(projection, self.level)
                printed.add(id(expr))
                return
            for arg in expr.args:
                self.visit(arg, ctx)
            _emit_inline_call(expr, self.level)

        def default_visit(self, expr, ctx=None) -> None:
            return None

    _expr_emitter = _ExprEmitter("")

    def _emit_expr(expr: Expr, level: str) -> None:
        previous = _expr_emitter.level
        _expr_emitter.level = level
        try:
            _expr_emitter.emit(expr)
        finally:
            _expr_emitter.level = previous

    def _emit_loop_region(region: LoopRegion, level: str) -> None:
        key = id(region)
        if key in printed:
            return
        for init in region.init_args:
            _emit_expr(init, level)
        for carry in region.carried_args:
            printed.add(id(carry))
        extent = shape_entry_str(region.extent)
        step = shape_entry_str(region.step)
        start = shape_entry_str(region.start)
        if id(region.induction_var) in _tile_window_steps:
            loop = f"tile({extent}, {step})"
        elif region.start == 0 and region.step == 1:
            loop = f"range({extent})"
        else:
            loop = f"range({start}, {extent}, {step})"
        lines.append(f"{level}for {region.induction_var.name} in {loop}:{_comments(region, options, mesh_map)}")
        printed.add(key)
        inner = level + "    "
        _emit_expr(region.body, inner)
        for value in region.yield_values:
            _emit_expr(value, inner)
        for carry, value in zip(region.carried_args, region.yield_values):
            lines.append(f"{inner}{_names[id(carry)]} = {_expr_ref(value)}")

    def _emit_mesh_region(region: MeshRegion, level: str, *, terminal: bool = False) -> None:
        key = id(region)
        if key in printed:
            return
        for arg in region.args:
            _emit_expr(arg, level)
        mesh_name = mesh_map[id(region.mesh)]
        lines.append(
            f"{level}with {mesh_name} as _{mesh_name}:"
            f"{_comments(region, options, mesh_map)}"
        )
        printed.add(key)
        inner = level + "    "
        if terminal and isinstance(region.body, MeshRegion):
            _emit_mesh_region(region.body, inner, terminal=True)
            return
        _emit_expr(region.body, inner)
        if terminal:
            lines.append(f"{inner}return {_arg_ref(region.body)}")

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
            lines.append(f"{indent}{name} = {repr(expr.value)}{_comments(expr, options, mesh_map)}")
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
            lines.append(
                f"{indent}{name} = {_format_call(expr, indent)}"
                f"{_comments(expr, options, mesh_map)}"
            )
            line = _constraint_line(expr, indent, name)
            if line is not None:
                lines.append(line)
            printed.add(id(expr))



    if not isinstance(fn.body, MeshRegion):
        if isinstance(fn.body, Tuple):
            lines.append(f"{indent}return {_tuple_literal(fn.body.elements)}")
        elif isinstance(fn.body, LoopRegion):
            values = tuple(_names[id(carry)] for carry in fn.body.carried_args)
            result = values[0] if len(values) == 1 else "(" + ", ".join(values) + ")"
            lines.append(f"{indent}return {result}")
        else:
            body_name = _expr_ref(fn.body)
            lines.append(f"{indent}return {body_name}")
    return lines


_HIR_RENDERER = HirPrinter()
_dtype_str = _HIR_RENDERER.dtype_str
_mesh_name_map = _HIR_RENDERER.mesh_name_map
_pattern_ctor = _HIR_RENDERER.render_pattern


def _topologies_str(mesh: Mesh) -> str:
    values = ", ".join(
        f'Topology("{topology.name}", {shape_entry_str(topology.size)})'
        for topology in mesh.topologies
    )
    return f"({values}{',' if len(mesh.topologies) == 1 else ''})"


def _shape_tuple(shape: tuple) -> str:
    return _HIR_RENDERER.shape_tuple(shape)


def _layout_str(layout: LayoutBase | None, indent: str = "") -> str:
    return _HIR_RENDERER.render_layout(layout, HirPrintContext(), indent)


def _mesh_str(mesh: Mesh, indent: str = "") -> str:
    return _HIR_RENDERER.render_mesh(mesh, HirPrintContext(), indent)


def _shard_layout_str(sl: ShardLayout, indent: str = "", *, mesh_ref=None) -> str:
    ctx = HirPrintContext({id(sl.mesh): mesh_ref} if mesh_ref is not None else None)
    return _HIR_RENDERER.render_shard_layout(sl, ctx, indent)


def _tensor_annotation(ty: TensorType, *, mesh_name_map=None, indent="", is_const=False) -> str:
    return _HIR_RENDERER.render_tensor_type(
        ty, HirPrintContext(mesh_name_map), indent, is_const
    )


def _collect_all_meshes(
    fn: HirFunction,
) -> tuple[dict[int, Mesh], dict[int, Mesh]]:
    """Meshes referenced by *fn* and every specialization variant.

    Meshes referenced by *fn* and every specialization variant — the
    printer's mesh-name map must stay stable across the base prototype and
    each ``.specialize`` block.
    """
    type_meshes: dict[int, Mesh] = {}
    scope_meshes: dict[int, Mesh] = {}
    for f in (fn, *fn.variants):
        types, scopes = _collect_meshes(f)
        type_meshes.update(types)
        scope_meshes.update(scopes)
    return type_meshes, scope_meshes


def _dedup_meshes(meshes: dict[int, Mesh]) -> dict[int, Mesh]:
    """Collapse structurally identical descriptors before naming hoisted meshes."""
    result: dict[int, Mesh] = {}
    for identity, mesh in meshes.items():
        signature = _mesh_str(mesh)
        if any(signature == _mesh_str(existing) for existing in result.values()):
            continue
        result[identity] = mesh
    return result


def _emit_header(
    fn: HirFunction,
    meshes: dict[int, Mesh],
    mesh_map: dict[int, str],
    indent: str,
    *,
    for_module: bool = False,
    target: object | None = None,
    dim_vars: "dict[str, object] | None" = None,
    scope_mesh_ids: set[int] | None = None,
) -> list[str]:
    """Import header + mesh-prelude shared by ``hir_function_to_python`` and ``_module_to_python``.

    Import header + mesh-prelude shared by ``hir_function_to_python`` and
    ``_module_to_python`` — the only source for the imports/mesh-defs a
    dispatch prototype needs (the conditional ``DimVarRangePat`` import for
    ``fn.variants``, the ``ConstTensor``/``Tensor`` selection), so standalone
    and module-wrapped output cannot drift out of sync.
    """
    lines: list[str] = ["from __future__ import annotations", ""]
    if for_module:
        lines.append("from tilefoundry.module import module")
    lines.append("from tilefoundry import func")
    if target is not None:
        rendered: PythonExpr = target.to_python()
        lines.extend(rendered.imports)
    lines.append("from tilefoundry.dsl.tf import *  # noqa: F401, F403")
    lines.append(f"from tilefoundry.dsl import {_tensor_import_names(fn)}")
    lines.append("from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401")
    lines.append("from tilefoundry.ir.types.shard import (")
    lines.append(f"{indent}B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,")
    lines.append(")")
    if fn.variants:
        lines.append("from tilefoundry.ir.core.pattern import DimVarRangePat")
    if dim_vars:
        lines.append("from tilefoundry.ir.types.dim import DimVar, ceildiv")
    lines.append("")





    if dim_vars:
        for name, var in dim_vars.items():
            lines.append(f'{name} = DimVar("{var.name}", {var.lo}, {var.hi})')
        lines.append("")


    if any(mesh.names or mid in (scope_mesh_ids or ()) for mid, mesh in meshes.items()):
        for mid, mesh in meshes.items():
            if not mesh.names and mid not in (scope_mesh_ids or ()):
                continue
            name = mesh_map[mid]
            topologies = _topologies_str(mesh)
            names_repr = repr(tuple(mesh.names)) if mesh.names else "()"
            lines.append(
                f"{name} = Mesh("
                f"{topologies}, "
                f"{_layout_str(mesh.layout)}, "
                f"names={names_repr}"
                f")"
            )
        lines.append("")
    return lines


def _variant_binding_name(variant: HirFunction) -> str:
    """Return a valid source binding for a variant without display metadata."""
    label = display_name(variant)
    if label is not None:
        return label
    signature = canonical_specialization_signature(variant.specializations)
    return "variant_" + re.sub(r"[^0-9A-Za-z_]", "_", signature)


def _emit_decorated_defs(
    fn: HirFunction, mesh_map: dict[int, str], indent: str, options: PythonPrintOptions,
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
    lines: list[str] = ["@func"]
    lines.extend(
        _emit_def(
            fn,
            fn.name,
            mesh_map,
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
            f"@{fn.name}.specialize({_pattern_ctor(variant.specializations[0])})"
        )
        lines.extend(
            _emit_def(
                variant, _variant_binding_name(variant), mesh_map, indent, options,
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
    one ``@<name>.specialize(pattern)`` block per variant. When the function
    uses meshes with named axes, compact sugar form is emitted; otherwise the
    verbose ``ShardLayout(...)`` form is used.
    """
    indent = "    "
    type_meshes, scope_meshes = _collect_all_meshes(fn)
    meshes = {**type_meshes, **scope_meshes}
    mesh_map = _mesh_name_map(meshes)
    lines = _emit_header(
        fn,
        meshes,
        mesh_map,
        indent,
        dim_vars=dim_vars_reached(fn),
        scope_mesh_ids=set(scope_meshes),
    )
    statements: dict[int, _PrintedStatement] = {}
    lines.extend(
        _emit_decorated_defs(
            fn,
            mesh_map,
            indent,
            options or PythonPrintOptions(),
            line_offset=_physical_line_count(lines),
            statements=statements,
        )
    )
    return _PythonRendering("\n".join(lines) + "\n", statements)


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
    return tuple(fn for fn in mod.functions if isinstance(fn, HirFunction))


def _module_tree_functions(mod: Module) -> tuple[HirFunction, ...]:
    """Every HIR function owned by *mod* or any Module beneath it."""
    functions = list(_module_hir_functions(mod))
    for child in mod.modules:
        functions.extend(_module_tree_functions(child))
    return tuple(functions)


def _module_decorator_line(mod: Module, entry_name: str | None) -> str:
    """Render the context this Module declares as an ``@module(...)`` line.

    Always the called form. A bare decorator has not run while the class body
    is evaluated, so a body naming a child call could not resolve it.
    """
    kwargs: list[str] = [] if entry_name is None else [f'entry="{entry_name}"']
    if mod.target is not None:
        rendered: PythonExpr = mod.target.to_python()
        kwargs.append(f"target={rendered.text}")
    if mod.topologies is not None:
        topo_strs = [
            f'Topology("{t.name}", {shape_entry_str(t.size)})'
            for t in mod.topologies
        ]
        rendered_topologies = f'({", ".join(topo_strs)},)' if topo_strs else "()"
        kwargs.append(f"topologies={rendered_topologies}")
    return f"@module({', '.join(kwargs)})"


def _emit_module_class(
    mod: Module, module_name: str, mesh_map: dict[int, str], indent: str,
    options: PythonPrintOptions,
) -> list[str]:
    """One ``@module`` class block: its nested Modules, then its functions.

    Children first, because a body calling one names the attribute it is bound
    to and a class body binds in the order it is written.
    """
    functions = mod.functions
    entry = mod.entry_function() if functions and mod.entry is not None else None
    lines = [_module_decorator_line(mod, mod.entry), f"class {module_name}:"]
    ordered = tuple(fn for fn in functions if fn is not entry)
    if entry is not None:
        ordered += (entry,)
    child_entries = {
        id(child.entry_function()): child.name
        for child in mod.modules
        if child.entry is not None and isinstance(child.entry_function(), HirFunction)
    }
    blocks: list[list[str]] = [
        _emit_module_class(child, child.name, mesh_map, indent, options)
        for child in mod.modules
    ]
    for fn in ordered:
        if isinstance(fn, HirFunction):
            blocks.append(_emit_decorated_defs(fn, mesh_map, indent, options, child_entries))
        elif isinstance(fn, PrimFunction):
            blocks.append(_tir_function_block(fn))
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


    header_of = entry if entry is not None else functions[0]
    indent4 = "    "
    type_meshes: dict[int, Mesh] = {}
    scope_meshes: dict[int, Mesh] = {}
    for fn in functions:
        types, scopes = _collect_all_meshes(fn)
        type_meshes.update(types)
        scope_meshes.update(scopes)
    meshes = {**type_meshes, **scope_meshes}
    mesh_map = _mesh_name_map(meshes)



    dim_vars: dict[str, object] = {}
    for fn in functions:
        dim_vars.update(dim_vars_reached(fn))
    for node in _module_tree(root):
        dim_vars.update(dim_vars_by_name(node.topologies or ()))
    lines = _emit_header(
        header_of, meshes, mesh_map, indent4, for_module=True, target=root.target,
        dim_vars=dim_vars, scope_mesh_ids=set(scope_meshes),
    )
    if any(isinstance(fn, PrimFunction) for node in _module_tree(root) for fn in node.functions):
        lines = [
            line.replace("from tilefoundry import func", "from tilefoundry import func, prim_func")
            for line in lines
        ]
        tensor_line = next(i for i, line in enumerate(lines) if line.startswith("from tilefoundry.dsl import "))
        lines[tensor_line] = lines[tensor_line].replace("import ", "import T, ")
        target_imports = sorted({fn.target.to_python().imports[0] for node in _module_tree(root) for fn in node.functions if isinstance(fn, PrimFunction)})
        lines[2:2] = target_imports
    tensor_names = "ConstTensor, Tensor" if any(
        param.is_const for fn in functions for param in fn.params
    ) else "Tensor"
    lines = [
        f"from tilefoundry.dsl import {tensor_names}" if line.startswith("from tilefoundry.dsl import Tensor") else line
        for line in lines
    ]
    lines.extend(
        _emit_module_class(
            root, module_name, mesh_map, indent4, options or PythonPrintOptions(),
        )
    )
    return "\n".join(lines) + "\n"


def _module_tree(root: Module) -> Iterator[Module]:
    yield root
    for child in root.modules:
        yield from _module_tree(child)
