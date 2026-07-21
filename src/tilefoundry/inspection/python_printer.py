"""Canonical Python DSL printer for HIR Functions.

Converts a ``hir.Function`` to executable Python source using the
``@func`` DSL.  When meshes have named axes, compact sugar annotations
are emitted; otherwise the verbose ``ShardLayout(...)`` form is used.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterator

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.sharding.reshard import Reshard
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
from tilefoundry.ir.types.shard import c_order_strides
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
from tilefoundry.ir.visitor import _expr_children
from tilefoundry.schedule.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    StorageConstraint,
    constraint_metadata,
)
from tilefoundry.schedule.constraints.layout import is_layout_wildcard
from tilefoundry.target import CpuTarget, CudaTarget, Target
from tilefoundry.target.cuda import SM90

# ``Op class → infix symbol`` for dim-arithmetic shape entry rendering.
_DIM_INFIX_OPS: dict[type, str] = {
    DimAdd: "+",
    DimSub: "-",
    DimMul: "*",
    DimFloorDiv: "//",
    DimMod: "%",
}

# Dim ops without a natural infix form get function-style rendering.
_DIM_FUNC_OPS: dict[type, str] = {
    DimMin: "min",
    DimMax: "max",
}


def shape_entry_str(entry: object) -> str:
    """Render one ``TensorType.shape`` entry as a human-readable string.

    A ``ShapeDim`` entry is one of:
    - a static ``int``;
    - a ``DimVar`` (Op instance) — rendered as its ``name``;
    - a dim-arithmetic ``Expr`` tree built from ``DimAdd`` / ``DimSub``
      / ``DimMul`` / ``DimFloorDiv`` / ``DimMod`` / ``DimMin`` /
      ``DimMax`` ops — rendered as ``"<lhs> + <rhs>"`` (etc.) or
      ``"min(<a>, <b>)"`` for function-style ops.

    Used by ``_shape_tuple`` and the viewer label path so that a shape
    like ``(1, 2, DimAdd(CTX_LEN, 1), 256)`` prints as
    ``(1, 2, CTX_LEN + 1, 256)`` rather than the dataclass repr.
    """
    if isinstance(entry, bool):
        return repr(entry)
    if isinstance(entry, int):
        return str(entry)
    if isinstance(entry, DimVar):
        return entry.name
    if isinstance(entry, Constant):
        return str(entry.value)
    if isinstance(entry, Call):
        target = entry.target
        if isinstance(target, DimConst):
            return str(target.value)
        for op_cls, sym in _DIM_INFIX_OPS.items():
            if isinstance(target, op_cls):
                a, b = entry.args
                return f"{shape_entry_str(a)} {sym} {shape_entry_str(b)}"
        for op_cls, fname in _DIM_FUNC_OPS.items():
            if isinstance(target, op_cls):
                rendered = ", ".join(shape_entry_str(a) for a in entry.args)
                return f"{fname}({rendered})"
        return repr(entry)
    if isinstance(entry, Expr):
        return repr(entry)
    return repr(entry)


def _classify_shard_attrs(
    sl: ShardLayout, mesh_name: str
) -> tuple[dict[int, list[str]], list[str]] | None:
    """Classify ``sl.attrs`` into ``(splits, partials)``.

    ``splits`` maps each ``Split``'s **layout** axis to the ordered
    ``mesh.axis`` refs bound there (more than one entry when nested mesh axes
    split the same layout axis); ``partials`` is the ordered ``mesh.axis @
    P("reduction")`` suffix for every ``Partial``. ``Broadcast`` is omitted.

    Returns ``None`` — caller falls back to the verbose ``ShardLayout(...)``
    form — when the attr count doesn't match the mesh rank, a ``Split``
    targets an out-of-range layout axis, or an attr is none of ``Split`` /
    ``Partial`` / ``Broadcast``.

    Shared by ``_shard_layout_surface_str`` (keeps the layout-axis keying)
    and ``shard_compact_inline`` (remaps onto tensor axis via
    ``layout_axis_to_tensor_axis``).
    """
    layout = sl.layout
    if not isinstance(layout, Layout) or len(sl.attrs) != len(sl.mesh.axes):
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
    """Canonical shard-layout sugar (parser layout-sugar outer-tuple form):

    - ``(dims)``                          implicit strides, no value-state
    - ``((dims), (strides))``             explicit strides
    - ``((dims), {mesh.axis @ P(...)})``  value-state (Partial) set
    - ``((dims), (strides), {...})``      explicit strides + value-state

    ``Split`` is inlined on its layout dim; ``Partial`` is a mesh-axis value
    state rendered in the ``{...}`` set; ``Broadcast`` is omitted. Returns
    ``None`` when the layout cannot be expressed in sugar (verbose fallback).
    """
    layout = sl.layout
    if not isinstance(layout, Layout):
        return None
    classified = _classify_shard_attrs(sl, mesh_name)
    if classified is None:
        return None
    splits, partials = classified

    # All-Broadcast in a multi-mesh context is ambiguous → verbose fallback.
    if not splits and not partials and not mesh_unique:
        return None

    dims = [
        f"{d} {' '.join(f'@ {r}' for r in splits[i])}" if i in splits else str(d)
        for i, d in enumerate(layout.shape)
    ]
    dim_str = ", ".join(dims)
    if len(dims) == 1:
        dim_str += ","
    axis_tuple = f"({dim_str})"

    c_strides = c_order_strides(layout.shape)
    explicit = layout.strides is not None and layout.strides != c_strides
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
    """Decompose a ``ShardLayout`` for the compact display form.

    Maps each ``Split`` onto the tensor axis it lives in (so the shape can carry
    an inline ``size @ mesh.axis``) and collects ``Partial`` value states as
    ``mesh.axis @ P("reduction")``; ``Broadcast`` is omitted.

    Returns ``(split_ref_by_tensor_axis, partials)`` — a dict from tensor axis to
    the ``mesh.axis`` reference plus the ordered partial-suffix entries — or
    ``None`` when the layout cannot be inlined onto the tensor shape (a tensor
    axis split across more than one layout position, an out-of-range or unknown
    attr, or a rank mismatch), in which case the caller falls back to canonical.

    Shares the ``Split`` / ``Partial`` / ``Broadcast`` classification with the
    canonical ``_shard_layout_surface_str`` via ``_classify_shard_attrs``.
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
            return None  # same layout axis split by more than one mesh axis
        t_axis = la2ta[layout_axis]
        if t_axis in split_ref:
            return None  # tensor axis split across multiple layout positions
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
            f"{child_indent}offset={layout.offset},\n"
            f"{child_indent}outer={_layout_str(layout.outer, child_indent)},\n"
            f"{indent})"
        )
    raise TypeError(f"unsupported layout type: {type(layout).__name__}")


def _mesh_str(mesh: Mesh, indent: str = "") -> str:
    """Mesh(...) constructor string, includes ``names=`` when non-empty."""
    topo = mesh.topology
    base = f'Mesh(Topology("{topo.name}", {topo.size}), {_layout_str(mesh.layout, indent)}'
    if mesh.names:
        base += f", names={repr(tuple(mesh.names))}"
    return base + ")"


def _shard_layout_str(sl: ShardLayout, indent: str = "") -> str:
    """ShardLayout(...) constructor string, multi-line for readability."""
    child_indent = indent + "    "
    layout = _layout_str(sl.layout, child_indent)
    mesh = _mesh_str(sl.mesh, child_indent)
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
    """``"Tensor"`` or ``"ConstTensor, Tensor"`` — whichever the printed
    signature (base plus every variant) actually references."""
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
    if ty.storage is not None and ty.storage != StorageKind.GMEM:
        base += f', "{ty.storage.name.lower()}"'
    base += "]"
    return base


def _op_name(target) -> str:
    """Python DSL function name for an Op.

    Resolution order:

    1. Kinded ``Binary`` / ``Unary`` instances render as
       their surface alias (``add`` / ``cmp_eq`` / ``neg`` / ...) —
       not as ``binary(..., kind=BinaryKind.ADD)`` which would fail to
       re-parse without ``BinaryKind`` in scope.
    2. ``target._op_schema.name`` — set by ``@register_op``; this is
       the canonical DSL name and works for ops with non-trivial class
       names (e.g. ``Mma_SM80_16x8x16`` → ``mma_sm80_16x8x16``).
    3. CamelCase → snake_case fallback.
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
    """Return the surface alias name (``add`` / ``neg`` / ...) for a
    kinded ``Binary`` / ``Unary`` instance, else ``None``.

    Per-name HIR math classes are gone; the IR instance is
    ``Binary(kind=...)`` / ``Unary(kind=...)``. Round-trip printing
    must emit the alias surface name so the regenerated DSL source
    re-parses against the same alias schema.
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
        },
    )


_BINARY_KIND_TO_ALIAS, _UNARY_KIND_TO_ALIAS = _build_kinded_alias_maps()


def _op_display_name(target) -> str:
    """Display-only op name for DOT / viewer graph labels: the target's class
    name with a trailing ``Op`` / ``Expr`` / ``Stmt`` suffix stripped
    (``MatMul``, ``TupleGetItem``, ...). Distinct from ``_op_name``, which
    renders the round-trippable DSL callable name — this one is shared by
    ``dot.py`` and ``viewer/builder.py`` for human-facing labels only."""
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
    """Post-order traversal of *root* and its descendants via
    ``tilefoundry.ir.visitor._expr_children`` (which, unlike the hand-rolled
    walkers this replaces, descends into ``GridRegionExpr``). Each node is
    yielded exactly once by object identity; *seen* lets callers share dedup
    state across repeated calls (e.g. one per function param)."""
    if root is None:
        return
    if seen is None:
        seen = set()
    key = id(root)
    if key in seen:
        return
    seen.add(key)
    for child in _expr_children(root):
        yield from iter_exprs(child, seen)
    yield root


def _collect_meshes(fn: HirFunction, *, include_node_types: bool = False) -> dict[int, Mesh]:
    """Collect unique Mesh objects referenced anywhere in *fn* — params,
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

    Uses ``mesh.topology.name`` when available; falls back to ``mesh_N``.
    """
    name_map: dict[int, str] = {}
    used: set[str] = set()
    for mid, mesh in meshes.items():
        base = mesh.topology.name if mesh.topology and mesh.topology.name else "mesh"
        name = base
        n = 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        name_map[mid] = name
    return name_map


def _emit_def(
    fn: HirFunction, def_name: str, mesh_map: dict, indent: str
) -> list[str]:
    """Render one function ``def`` block: signature + body (or ``pass`` for a
    prototype). The caller prepends the decorator line(s). Each call builds its
    own SSA name scope, so a base and its variants do not share names."""
    lines: list[str] = []

    # Collect all SSA values and assign names.
    _counter = [0]
    _names: dict[int, str] = {}

    # Topological sort: post-order from body, via the shared iter_exprs
    # generator (one _seen set shared across body + params).
    _seen: set[int] = set()
    _order: list[Expr] = list(iter_exprs(fn.body, _seen))
    for p in fn.params:
        _order.extend(iter_exprs(p, _seen))

    # Collect op names first (must be before _assign_name references them)
    _op_names_set: set[str] = set()
    for expr in _order:
        if isinstance(expr, Call):
            _op_names_set.add(_op_name(expr.target))

    _forced_names: dict[int, str] = {}
    # Reuse iter_exprs' dedup-set parameter as the "reachable ids" accumulator
    # for each grid region's internal / init subtrees.
    _grid_internal_ids: set[int] = set()
    _grid_init_ids: set[int] = set()

    for expr in _order:
        if not isinstance(expr, GridRegionExpr):
            continue
        for carry, init, value in zip(
            expr.carried_args, expr.init_args, expr.yield_values
        ):
            _forced_names[id(carry)] = _sanitize_name(carry.name)
            _forced_names[id(init)] = _sanitize_name(carry.name)
            _forced_names[id(value)] = _sanitize_name(carry.name)
        for _ in iter_exprs(expr.body, _grid_internal_ids):
            pass
        for value in expr.yield_values:
            for _ in iter_exprs(value, _grid_internal_ids):
                pass
        for init in expr.init_args:
            for _ in iter_exprs(init, _grid_init_ids):
                pass
    _grid_internal_ids.difference_update(_grid_init_ids)

    def _assign_name(expr: Expr) -> str:
        key = id(expr)
        if key in _names:
            return _names[key]
        if key in _forced_names:
            name = _forced_names[key]
        elif isinstance(expr, Var):
            name = _sanitize_name(expr.name)
        elif isinstance(expr, Call) and expr.loc:
            name = _sanitize_name(expr.loc)
        else:
            name = f"v{_counter[0]}"
            _counter[0] += 1
        if key in _forced_names and name in _names.values():
            _names[key] = name
            return name
        # Avoid shadowing op names when assigning a call result.
        if name in _op_names_set:
            name = f"{name}_out"
        base = name
        n = 2
        while name in _names.values():
            name = f"{base}_{n}"
            n += 1
        _names[key] = name
        return name

    # Assign names
    for expr in _order:
        _assign_name(expr)
    for expr in _order:
        if isinstance(expr, GridRegionExpr):
            for carry in expr.carried_args:
                _assign_name(carry)

    def _tuple_literal(elements) -> str:
        inner = ", ".join(_expr_ref(el) for el in elements)
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
        return _names[id(expr)]

    def _arg_ref(a) -> str:
        # A tuple-valued input (e.g. insert_slice's per-axis offsets) renders
        # inline as a literal so the parser's narrow route lifts it back to a
        # core Tuple on re-parse.
        return _tuple_literal(a.elements) if isinstance(a, Tuple) else _expr_ref(a)

    # Function signature. A ``TupleType`` return has no surface annotation; it
    # is re-inferred from the literal tuple ``return`` body on re-parse.
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
    lines.append(",\n".join(param_strs))
    lines.append(f"){arrow}:")

    for param in fn.params:
        line = _constraint_line(param, indent, _names[id(param)])
        if line is not None:
            lines.append(line)

    # A dispatch prototype has no body — declare signature only.
    if fn.body is None:
        lines.append(f"{indent}pass")
        return lines

    printed: set[int] = {id(param) for param in fn.params}

    def _format_call(expr: Call, indent_here: str) -> str:
        """Render a Call's RHS expression text: the ``reshard(...)`` /
        ``<HirFunction>(...)`` special forms, else ``op_name(args, attr=val,
        ...)``. Shared by the inline (tile-loop body) emitter and the
        top-level emit loop so an attribute-rendering rule (``ShardLayout``,
        ``DType``, ...) only needs one edit."""
        target = expr.target
        args_str = ", ".join(_arg_ref(arg) for arg in expr.args)
        if isinstance(target, Reshard):
            layout_kw = ""
            if target.layout is not None:
                layout_kw = ", layout=" + _shard_layout_str(
                    target.layout, indent=indent_here + "    "
                )
            storage = (
                f", storage={target.storage.name.lower()}"
                if target.storage is not None
                else ""
            )
            return f"reshard({args_str}{layout_kw}{storage})"
        if isinstance(target, HirFunction):
            return f"{target.name}({args_str})"

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
                attr_strs.append(f"{param.name}={value!r}")
            elif isinstance(value, ShardLayout):
                sl_str = _shard_layout_str(value, indent=indent_here + "        ")
                attr_strs.append(f"{param.name}={sl_str}")
            elif isinstance(value, tuple):
                attr_strs.append(f"{param.name}={value}")
            else:
                attr_strs.append(f"{param.name}={value}")
        call_str = f"{_op_name(target)}({args_str}"
        if attr_strs:
            call_str += ", " + ", ".join(attr_strs)
        call_str += ")"
        return call_str

    def _emit_inline_call(expr: Call, level: str) -> None:
        name = _names[id(expr)]
        lines.append(f"{level}{name} = {_format_call(expr, level)}")
        printed.add(id(expr))

    def _emit_expr(expr: Expr, level: str) -> None:
        key = id(expr)
        if key in printed:
            return
        if isinstance(expr, Var):
            printed.add(key)
            return
        if isinstance(expr, Constant):
            lines.append(f"{level}{_names[key]} = {repr(expr.value)}")
            printed.add(key)
            return
        if isinstance(expr, Tuple):
            for element in expr.elements:
                _emit_expr(element, level)
            printed.add(key)
            return
        if isinstance(expr, GridRegionExpr):
            _emit_grid(expr, level)
            return
        if isinstance(expr, Call):
            if (
                isinstance(expr.target, TupleGetItem)
                and len(expr.args) == 1
                and isinstance(expr.args[0], GridRegionExpr)
            ):
                _emit_grid(expr.args[0], level)
                printed.add(key)
                return
            for arg in expr.args:
                _emit_expr(arg, level)
            _emit_inline_call(expr, level)

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
        if grid.start == 0 and grid.step == 1:
            loop = f"tile({extent})"
        elif grid.start == 0:
            loop = f"tile({extent}, {step})"
        else:
            loop = f"range({start}, {extent}, {step})"
        lines.append(f"{level}for {grid.induction_var.name} in {loop}:")
        printed.add(key)
        inner = level + "    "
        _emit_expr(grid.body, inner)
        for value in grid.yield_values:
            _emit_expr(value, inner)

    for expr in _order:
        if isinstance(expr, Var) or id(expr) in _grid_internal_ids:
            continue  # params already in signature
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
            lines.append(f"{indent}{name} = {repr(expr.value)}")
            line = _constraint_line(expr, indent, name)
            if line is not None:
                lines.append(line)
            printed.add(id(expr))
            continue
        if isinstance(expr, Tuple):
            # A tuple is rendered inline at its use site: as a literal argument
            # (op input) or by the ``return`` statement (function body). The
            # parser lifts an inline offset tuple back to a core Tuple, whereas a
            # hoisted ``name = (...)`` binding would not re-parse.
            continue
        if isinstance(expr, Call):
            name = _names[id(expr)]
            loc = f'  # loc="{name}"' if expr.loc else ""
            lines.append(f"{indent}{name} = {_format_call(expr, indent)}{loc}")
            line = _constraint_line(expr, indent, name)
            if line is not None:
                lines.append(line)
            printed.add(id(expr))

    # Return statement. A literal tuple body renders its elements inline
    # (``return (e0, e1)``) rather than a name for the un-emitted Tuple node.
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


def _target_str(target: Target) -> str:
    """Render one explicit target decorator value."""
    if isinstance(target, CpuTarget):
        return "CpuTarget()"
    if isinstance(target, CudaTarget):
        architecture = target.architecture
        if architecture == SM90():
            return "CudaTarget()"
        dtypes = ", ".join(
            f"DType.{dtype.name}" for dtype in architecture.supported_compute_dtypes
        )
        if len(architecture.supported_compute_dtypes) == 1:
            dtypes += ","
        return (
            "CudaTarget(architecture=SM90("
            f"name={architecture.name!r}, "
            f"supported_compute_dtypes=({dtypes}), "
            f"instruction_capabilities={architecture.instruction_capabilities!r}, "
            f"max_threads_per_cta={architecture.max_threads_per_cta}, "
            f"max_threads_per_warp={architecture.max_threads_per_warp}, "
            f"max_warps_per_cta={architecture.max_warps_per_cta}"
            "))"
        )
    raise TypeError(f"unsupported target for Python printing: {type(target).__name__}")


def _collect_all_meshes(fn: HirFunction) -> dict[int, Mesh]:
    """Meshes referenced by *fn* and every specialization variant — the
    printer's mesh-name map must stay stable across the base prototype and
    each ``.specialize`` block."""
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
) -> list[str]:
    """Import header + mesh-prelude shared by ``hir_function_to_python`` and
    ``_module_to_python`` — the only source for the imports/mesh-defs a
    dispatch prototype needs (the conditional ``DimVarRangePat`` import for
    ``fn.variants``, the ``ConstTensor``/``Tensor`` selection), so standalone
    and module-wrapped output cannot drift out of sync."""
    lines: list[str] = ["from __future__ import annotations", ""]
    if for_module:
        lines.append("from tilefoundry.module import module")
    lines.append("from tilefoundry import func")
    if fn.target is not None:
        lines.append("from tilefoundry.target import CpuTarget, CudaTarget")
        if isinstance(fn.target, CudaTarget) and fn.target.architecture != SM90():
            lines.append("from tilefoundry.ir.types import DType")
    lines.append("from tilefoundry.dsl.tf import *  # noqa: F401, F403")
    lines.append(f"from tilefoundry.dsl import {_tensor_import_names(fn)}")
    lines.append("from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401")
    lines.append("from tilefoundry.ir.types.shard import (")
    lines.append(f"{indent}B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,")
    lines.append(")")
    if fn.variants:
        lines.append("from tilefoundry.ir.core.pattern import DimVarRangePat")
    lines.append("")

    # Mesh definitions, emitted only when sugar is viable (mesh has named axes).
    if any(m.names for m in meshes.values()):
        for mid, mesh in meshes.items():
            name = mesh_map[mid]
            topo = mesh.topology
            names_repr = repr(tuple(mesh.names)) if mesh.names else "()"
            lines.append(
                f"{name} = Mesh("
                f'Topology("{topo.name}", {topo.size}), '
                f"{_layout_str(mesh.layout)}, "
                f"names={names_repr}"
                f")"
            )
        lines.append("")
    return lines


def _emit_decorated_defs(fn: HirFunction, mesh_map: dict[int, str], indent: str) -> list[str]:
    """Base ``@func`` decorator + ``def`` block, followed by one
    ``@<name>.specialize(pattern)`` block per variant (§2.6). Shared by
    standalone and module-wrapped output so a dispatch prototype prints
    identically in both."""
    lines: list[str] = []
    decorator_kwargs = []
    if fn.target is not None:
        decorator_kwargs.append(f"target={_target_str(fn.target)}")
    if fn.topologies:
        topo_strs = [f'Topology("{t.name}", {t.size})' for t in fn.topologies]
        decorator_kwargs.append(f'topologies=({", ".join(topo_strs)},)')
    if decorator_kwargs:
        lines.append(f"@func({', '.join(decorator_kwargs)})")
    else:
        lines.append("@func")
    lines.extend(_emit_def(fn, fn.name, mesh_map, indent))

    # Variant defs: each a `@<base>.specialize(pattern)` over a throwaway `def _`.
    for variant in fn.variants:
        lines.append("")
        lines.append(
            f"@{fn.name}.specialize({_pattern_ctor(variant.specializations[0])})"
        )
        lines.extend(_emit_def(variant, "_", mesh_map, indent))
    return lines


def hir_function_to_python(fn: HirFunction) -> str:
    """Convert a HIR Function to canonical Python DSL source.

    A normal function prints as a single ``@func``. A dispatch prototype
    (``variants != ()``) prints as a ``pass``-bodied ``@func`` base followed by
    one ``@<name>.specialize(pattern)`` block per variant. When the function
    uses meshes with named axes, compact sugar form is emitted; otherwise the
    verbose ``ShardLayout(...)`` form is used.
    """
    indent = "    "
    meshes = _collect_all_meshes(fn)
    mesh_map = _mesh_name_map(meshes)
    lines = _emit_header(fn, meshes, mesh_map, indent)
    lines.extend(_emit_decorated_defs(fn, mesh_map, indent))
    return "\n".join(lines) + "\n"


def as_script(fn: HirFunction | Module, *, module: str | None = None) -> str:
    """Convert a HIR Function or Module to Python DSL source.

    Without *module*: standalone ``@func`` output.

    With *module* (e.g. ``module="M"``): ``@module(entry="<fn>") class M:``
    wrapper with module-level mesh definitions (the class body stays a pure
    function container) and sugar annotations.

    Args:
        fn: The HIR function or module.
        module: Optional module class name.  When set, the output is
            wrapped in ``@module(entry="<fn>") class <name>:``.

    Returns:
        Python source string.
    """
    if isinstance(fn, Module):
        return _module_to_python(fn, module)
    if module is not None:
        return _module_to_python(fn, module)
    return hir_function_to_python(fn)

# backward-compat alias
def module_to_python(fn: HirFunction, module_name: str = "M") -> str:
    """Backward-compat alias for ``as_script(fn, module=module_name)``."""
    return as_script(fn, module=module_name)


def _module_to_python(
    fn_or_module: HirFunction | Module, module_name: str | None = None
) -> str:
    """Render a function or every HIR function in a Module wrapper."""
    if isinstance(fn_or_module, Module):
        entry = fn_or_module.entry_function()
        if not isinstance(entry, HirFunction):
            raise TypeError("HIR Module printer requires a HIR entry Function")
        functions = tuple(fn for fn in fn_or_module.functions if isinstance(fn, HirFunction))
        if len(functions) != len(fn_or_module.functions):
            raise TypeError("HIR Module printer does not serialize mixed HIR/TIR Modules")
        module_name = fn_or_module.name if module_name is None else module_name
    else:
        entry = fn_or_module
        functions = (entry,)
        module_name = "M" if module_name is None else module_name
    indent4 = "    "
    meshes: dict[int, Mesh] = {}
    for fn in functions:
        meshes.update(_collect_all_meshes(fn))
    mesh_map = _mesh_name_map(meshes)

    lines = _emit_header(entry, meshes, mesh_map, indent4, for_module=True)
    tensor_names = "ConstTensor, Tensor" if any(
        param.is_const for fn in functions for param in fn.params
    ) else "Tensor"
    lines = [
        f"from tilefoundry.dsl import {tensor_names}" if line.startswith("from tilefoundry.dsl import Tensor") else line
        for line in lines
    ]

    lines.append(f'@module(entry="{entry.name}")')
    lines.append(f"class {module_name}:")

    ordered_functions = tuple(fn for fn in functions if fn is not entry) + (entry,)
    for index, fn in enumerate(ordered_functions):
        if index:
            lines.append("")
        body = _emit_decorated_defs(fn, mesh_map, indent4)
        lines.extend(f"{indent4}{ln}" if ln else ln for ln in body)

    return "\n".join(lines) + "\n"
