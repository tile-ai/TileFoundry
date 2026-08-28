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
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
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
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout, LayoutBase
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.substitute import dim_vars_by_name
from tilefoundry.ir.visitor import ExprFunctor, expr_children
from tilefoundry.utils.python_source import PythonExpr

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


def _dtype_str(dtype: DType) -> str:
    return dtype.name


def _shape_tuple(shape: tuple) -> str:
    """Render a shape as a Python tuple literal.

    Each entry is rendered via ``shape_entry_str`` so symbolic
    ``DimVar`` and dim-arithmetic ``Expr`` entries print as their
    canonical math-shaped string (``CTX_LEN``, ``CTX_LEN + 1``)
    instead of the dataclass repr. 1D rank renders as ``(N,)``.
    """
    rendered = tuple(shape_entry_str(e) for e in shape)
    if len(rendered) == 1:
        return f"({rendered[0]},)"
    return "(" + ", ".join(rendered) + ")"


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


def _shard_attr_str(attr) -> str:
    """Single ShardAttr to Python constructor string."""
    if isinstance(attr, Broadcast):
        return "B()"
    if isinstance(attr, Split):
        return f"S({attr.axis})"
    if isinstance(attr, Partial):
        return f'P("{attr.reduction}")'
    return f"/* {type(attr).__name__} */"


def _layout_str(layout: LayoutBase | None, indent: str = "") -> str:
    """Render a complete layout descriptor without flattening compositions."""
    if layout is None:
        return "None"
    if isinstance(layout, Layout):
        strides = _shape_tuple(layout.strides) if layout.strides is not None else "None"
        return f"Layout({_shape_tuple(layout.shape)}, {strides})"
    if isinstance(layout, ShardLayout):
        return _shard_layout_str(layout, indent=indent)
    if isinstance(layout, ComposedLayout):
        child_indent = indent + "    "
        return (
            "ComposedLayout(\n"
            f"{child_indent}inner={_layout_str(layout.inner, child_indent)},\n"
            f"{child_indent}offset={shape_entry_str(layout.offset)},\n"
            f"{child_indent}outer={_layout_str(layout.outer, child_indent)},\n"
            f"{indent})"
        )
    raise TypeError(f"unsupported layout type: {type(layout).__name__}")


def _topologies_str(mesh: Mesh) -> str:
    topologies = ", ".join(
        f'Topology("{topology.name}", {shape_entry_str(topology.size)})'
        for topology in mesh.topologies
    )
    return f"({topologies}{',' if len(mesh.topologies) == 1 else ''})"


def _mesh_str(mesh: Mesh, indent: str = "") -> str:
    """Mesh(...) constructor string, includes ``names=`` when non-empty."""
    base = f"Mesh({_topologies_str(mesh)}, {_layout_str(mesh.layout, indent)}"
    if mesh.names:
        base += f", names={repr(tuple(mesh.names))}"
    return base + ")"


def _shard_layout_str(
    sl: ShardLayout, indent: str = "", *, mesh_ref: str | None = None
) -> str:
    """ShardLayout(...) constructor string, multi-line for readability."""
    child_indent = indent + "    "
    layout = _layout_str(sl.layout, child_indent)
    mesh = mesh_ref or _mesh_str(sl.mesh, child_indent)
    attrs = ", ".join(_shard_attr_str(a) for a in sl.attrs)
    if len(sl.attrs) == 1:
        attrs += ","
    return (
        f"ShardLayout(\n"
        f"{child_indent}layout={layout},\n"
        f"{child_indent}attrs=({attrs}),\n"
        f"{child_indent}mesh={mesh},\n"
        f"{indent})"
    )


def _tensor_import_names(fn: HirFunction) -> str:
    """``"Tensor"`` or ``"ConstTensor, Tensor"``.

    ``"Tensor"`` or ``"ConstTensor, Tensor"`` — whichever the printed
    signature (base plus every variant) actually references.
    """
    if any(p.is_const for f in (fn, *fn.variants) for p in f.params):
        return "ConstTensor, Tensor"
    return "Tensor"


def _tensor_annotation(
    ty: TensorType,
    *,
    mesh_name_map: dict[int, str] | None = None,
    indent: str = "",
    is_const: bool = False,
) -> str:
    """Tensor[(shape), dtype, ShardLayout?, storage?] annotation string.

    When *mesh_name_map* is provided and the layout's mesh has named axes,
    compact sugar form is used instead of verbose ``ShardLayout(...)``.
    ``is_const`` selects the ``ConstTensor[...]`` head instead of ``Tensor``.
    """
    head = "ConstTensor" if is_const else "Tensor"
    base = f'{head}[{_shape_tuple(ty.shape)}, "{_dtype_str(ty.dtype)}"'
    if isinstance(ty.layout, ShardLayout):
        sl = ty.layout
        mesh = sl.mesh
        mesh_name = mesh_name_map.get(id(mesh)) if mesh_name_map else None
        mesh_unique = mesh_name_map is not None and len(mesh_name_map) == 1
        if mesh_name and mesh.names:
            sugar = _shard_layout_surface_str(sl, mesh_name=mesh_name, mesh_unique=mesh_unique)
            if sugar is not None:
                base += f", {sugar}"
            else:
                sl_str = _shard_layout_str(sl, indent=indent + "    ")
                base += f",\n{indent}    {sl_str}"
        else:
            sl_str = _shard_layout_str(sl, indent=indent + "    ")
            base += f",\n{indent}    {sl_str}"
    if ty.storage is not StorageKind.GMEM:
        base += f', "{ty.storage.name.lower()}"'
    base += "]"
    return base


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
    walkers this replaces, descends into ``GridRegionExpr``). Each node is
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


def _collect_meshes(fn: HirFunction, *, include_node_types: bool = False) -> dict[int, Mesh]:
    """Collect unique Mesh objects referenced anywhere in *fn* — params, return type.

    Collect unique Mesh objects referenced anywhere in *fn* — params,
    return type, and every ``Reshard`` layout in the body.

    With ``include_node_types=True`` (the viewer's wider scan, via
    ``viewer.builder._collect_view_meshes``) every node's own result type is
    also walked, since the viewer renders shard sugar on intermediate types
    too, not just params/return.
    """
    meshes: dict[int, Mesh] = {}

    def _add_layout(layout) -> None:
        if isinstance(layout, ShardLayout):
            meshes.setdefault(id(layout.mesh), layout.mesh)

    def _add_type(ty) -> None:
        if isinstance(ty, TensorType):
            _add_layout(ty.layout)
        elif isinstance(ty, TupleType):
            for f in ty.fields:
                _add_type(f)

    for p in fn.params:
        _add_type(p.type)
    _add_type(fn.return_type)

    for expr in iter_exprs(fn.body):
        if include_node_types:
            _add_type(getattr(expr, "type", None))
        if isinstance(expr, Call) and isinstance(expr.target, Reshard):
            _add_layout(expr.target.layout)

    return meshes


def _mesh_name_map(meshes: dict[int, Mesh]) -> dict[int, str]:
    """Assign stable variable names to each Mesh.

    Uses the first declared topology name when available; falls back to
    ``mesh_N``.
    """
    name_map: dict[int, str] = {}
    used: set[str] = set()
    for mid, mesh in meshes.items():
        base = mesh.topologies[0].name if mesh.topologies else "mesh"
        name = base
        n = 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        name_map[mid] = name
    return name_map


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
    _nested_grid_ids: set[int] = set()

    for expr in _order:
        if not isinstance(expr, GridRegionExpr):
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
            if isinstance(nested, GridRegionExpr) and nested is not expr:
                _nested_grid_ids.add(id(nested))
        for value in expr.yield_values:
            for nested in iter_exprs(value, set()):
                if isinstance(nested, GridRegionExpr) and nested is not expr:
                    _nested_grid_ids.add(id(nested))

    _root_grid_init_ids: set[int] = set()
    for expr in _order:
        if not isinstance(expr, GridRegionExpr) or id(expr) in _nested_grid_ids:
            continue
        for init in expr.init_args:
            for _ in iter_exprs(init, _root_grid_init_ids):
                pass
    _grid_internal_ids.difference_update(_root_grid_init_ids)

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
        if isinstance(expr, GridRegionExpr):
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
        if (
            isinstance(expr, Call)
            and isinstance(expr.target, TupleGetItem)
            and len(expr.args) == 1
            and isinstance(expr.args[0], GridRegionExpr)
        ):
            grid = expr.args[0]
            return _names[id(grid.carried_args[expr.target.index])]
        if isinstance(expr, GridRegionExpr):



            carried = tuple(_names[id(carry)] for carry in expr.carried_args)
            if len(carried) == 1:
                return carried[0]
            return "(" + ", ".join(carried) + ")"
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
        ``DType``, ...) only needs one edit.
        """
        target = expr.target
        args_str = ", ".join(_arg_ref(arg) for arg in expr.args)
        if isinstance(target, Reshard):
            layout_kw = ""
            if target.layout is not None:
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

        def visit_GridRegionExpr(self, expr: GridRegionExpr, ctx=None) -> None:
            _emit_grid(expr, self.level)

        def visit_Call(self, expr: Call, ctx=None) -> None:
            if (
                isinstance(expr.target, TupleGetItem)
                and len(expr.args) == 1
                and isinstance(expr.args[0], GridRegionExpr)
            ):
                _emit_grid(expr.args[0], self.level)
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

    def _emit_grid(grid: GridRegionExpr, level: str) -> None:
        key = id(grid)
        if key in printed:
            return
        for init in grid.init_args:
            _emit_expr(init, level)
        for carry in grid.carried_args:
            printed.add(id(carry))
        extent = shape_entry_str(grid.extent)
        step = shape_entry_str(grid.step)
        start = shape_entry_str(grid.start)
        if id(grid.induction_var) in _tile_window_steps:
            loop = f"tile({extent}, {step})"
        elif grid.start == 0 and grid.step == 1:
            loop = f"range({extent})"
        else:
            loop = f"range({start}, {extent}, {step})"
        lines.append(f"{level}for {grid.induction_var.name} in {loop}:{_comments(grid, options, mesh_map)}")
        printed.add(key)
        inner = level + "    "
        _emit_expr(grid.body, inner)
        for value in grid.yield_values:
            _emit_expr(value, inner)
        for carry, value in zip(grid.carried_args, grid.yield_values):
            lines.append(f"{inner}{_names[id(carry)]} = {_expr_ref(value)}")

    for expr in _order:
        if isinstance(expr, Var) or id(expr) in _grid_internal_ids:
            continue
        if id(expr) in _inlined_start_ids:
            printed.add(id(expr))
            continue
        if isinstance(expr, GridRegionExpr):
            _emit_grid(expr, indent)
            continue
        if (
            isinstance(expr, Call)
            and isinstance(expr.target, TupleGetItem)
            and len(expr.args) == 1
            and isinstance(expr.args[0], GridRegionExpr)
        ):
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



    if isinstance(fn.body, Tuple):
        lines.append(f"{indent}return {_tuple_literal(fn.body.elements)}")
    elif isinstance(fn.body, GridRegionExpr):
        values = tuple(_names[id(carry)] for carry in fn.body.carried_args)
        result = values[0] if len(values) == 1 else "(" + ", ".join(values) + ")"
        lines.append(f"{indent}return {result}")
    else:
        body_name = _expr_ref(fn.body)
        lines.append(f"{indent}return {body_name}")
    return lines


def _pattern_ctor(pat: Pattern) -> str:
    """Render a Pattern as its constructor, for a ``.specialize(...)`` decorator."""
    if isinstance(pat, DimVarRangePat):
        return f'DimVarRangePat("{pat.dim_var}", {pat.lo}, {pat.hi})'
    return repr(pat)


def _collect_all_meshes(fn: HirFunction) -> dict[int, Mesh]:
    """Meshes referenced by *fn* and every specialization variant.

    Meshes referenced by *fn* and every specialization variant — the
    printer's mesh-name map must stay stable across the base prototype and
    each ``.specialize`` block.
    """
    meshes: dict[int, Mesh] = {}
    for f in (fn, *fn.variants):
        meshes.update(_collect_meshes(f))
    return meshes


def _emit_header(
    fn: HirFunction,
    meshes: dict[int, Mesh],
    mesh_map: dict[int, str],
    indent: str,
    *,
    for_module: bool = False,
    target: object | None = None,
    dim_vars: "dict[str, object] | None" = None,
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


    if any(m.names for m in meshes.values()):
        for mid, mesh in meshes.items():
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
    meshes = _collect_all_meshes(fn)
    mesh_map = _mesh_name_map(meshes)
    lines = _emit_header(
        fn, meshes, mesh_map, indent, dim_vars=dim_vars_reached(fn)
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
    return _render_hir_function(fn, options=options).source


def as_script(
    fn: HirFunction | Module, *, module: str | None = None,
    options: PythonPrintOptions | None = None,
) -> str:
    """Convert an HIR function or module to Python DSL source.

    Without *module*, emit a standalone decorated function. With *module*, emit
    a named module-class wrapper with entry and mesh definitions. *options*
    controls canonical-source rendering.
    """
    if isinstance(fn, Module):
        return _module_to_python(fn, module, options=options)
    if module is not None:
        return _module_to_python(fn, module, options=options)
    return hir_function_to_python(fn, options=options)


def module_to_python(fn: HirFunction, module_name: str = "M") -> str:
    """Backward-compat alias for ``as_script(fn, module=module_name)``."""
    return as_script(fn, module=module_name)


def _module_hir_functions(mod: Module) -> tuple[HirFunction, ...]:
    """The Module's HIR functions, rejecting a mixed HIR/TIR container."""
    functions = tuple(fn for fn in mod.functions if isinstance(fn, HirFunction))
    if len(functions) != len(mod.functions):
        raise TypeError("HIR Module printer does not serialize mixed HIR/TIR Modules")
    return functions


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
    functions = _module_hir_functions(mod)
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
    blocks.extend(
        _emit_decorated_defs(fn, mesh_map, indent, options, child_entries)
        for fn in ordered
    )
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
    if entry is not None and not isinstance(entry, HirFunction):
        raise TypeError("HIR Module printer requires a HIR entry Function")


    header_of = entry if entry is not None else functions[0]
    indent4 = "    "
    meshes: dict[int, Mesh] = {}
    for fn in functions:
        meshes.update(_collect_all_meshes(fn))
    mesh_map = _mesh_name_map(meshes)



    dim_vars: dict[str, object] = {}
    for fn in functions:
        dim_vars.update(dim_vars_reached(fn))
    for node in _module_tree(root):
        dim_vars.update(dim_vars_by_name(node.topologies or ()))
    lines = _emit_header(
        header_of, meshes, mesh_map, indent4, for_module=True, target=root.target,
        dim_vars=dim_vars,
    )
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
