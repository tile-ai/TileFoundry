"""Canonical Python DSL printer for HIR Functions.

Converts a ``hir.Function`` to executable Python source using the
``@func`` DSL.  When meshes have named axes, compact sugar annotations
are emitted; otherwise the verbose ``ShardLayout(...)`` form is used.
"""

from __future__ import annotations

import enum
import re

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.target.storage import StorageKind
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
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)

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

    ``Split`` is inlined on its cute dim; ``Partial`` is a mesh-axis value
    state rendered in the ``{...}`` set; ``Broadcast`` is omitted. Returns
    ``None`` when the layout cannot be expressed in sugar (verbose fallback).
    """
    layout = sl.layout
    if not isinstance(layout, Layout):
        return None
    names = sl.mesh.names if hasattr(sl.mesh, "names") and sl.mesh.names else ()
    mesh_rank = len(sl.mesh.axes)
    layout_rank = len(layout.shape)
    if len(sl.attrs) != mesh_rank:
        return None

    dim_descs: list[tuple[int, list[str]]] = [(d, []) for d in layout.shape]
    partials: list[str] = []
    has_binding = False
    for mesh_axis_idx, attr in enumerate(sl.attrs):
        axis_name = names[mesh_axis_idx] if mesh_axis_idx < len(names) else f"ax{mesh_axis_idx}"
        axis_ref = f"{mesh_name}.{axis_name}"
        if isinstance(attr, Split):
            k = attr.axis
            if k >= layout_rank:
                return None
            dim_descs[k][1].append(f"@ {axis_ref}")
            has_binding = True
        elif isinstance(attr, Partial):
            partials.append(f'{axis_ref} @ P("{attr.reduction or "sum"}")')
            has_binding = True
        elif not isinstance(attr, Broadcast):
            return None

    # All-Broadcast in a multi-mesh context is ambiguous → verbose fallback.
    if not has_binding and not mesh_unique:
        return None

    dims = [
        f"{d} {' '.join(b)}" if b else str(d) for d, b in dim_descs
    ]
    dim_str = ", ".join(dims)
    if len(dims) == 1:
        dim_str += ","
    axis_tuple = f"({dim_str})"

    c_strides = [1]
    for dd in reversed(layout.shape[1:]):
        c_strides.insert(0, c_strides[0] * dd)
    explicit = layout.strides is not None and layout.strides != tuple(c_strides)
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
    axis split across more than one cute position, an out-of-range or unknown
    attr, or a rank mismatch), in which case the caller falls back to canonical.

    Shares the ``Split`` / ``Partial`` / ``Broadcast`` classification and the
    mesh-name map with the canonical ``_shard_layout_surface_str``.
    """
    layout = sl.layout
    if not isinstance(layout, Layout):
        return None
    if len(sl.attrs) != len(sl.mesh.axes):
        return None
    names = sl.mesh.names if hasattr(sl.mesh, "names") and sl.mesh.names else ()
    la2ta = layout_axis_to_tensor_axis(layout.shape, tensor_shape)
    split_ref: dict[int, str] = {}
    partials: list[str] = []
    for mesh_axis_idx, attr in enumerate(sl.attrs):
        axis_name = names[mesh_axis_idx] if mesh_axis_idx < len(names) else f"ax{mesh_axis_idx}"
        axis_ref = f"{mesh_name}.{axis_name}"
        if isinstance(attr, Split):
            if attr.axis >= len(la2ta):
                return None
            t_axis = la2ta[attr.axis]
            if t_axis in split_ref:
                return None  # tensor axis split across multiple cute positions
            split_ref[t_axis] = axis_ref
        elif isinstance(attr, Partial):
            partials.append(f'{axis_ref} @ P("{attr.reduction or "sum"}")')
        elif not isinstance(attr, Broadcast):
            return None
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


def _mesh_str(mesh: Mesh) -> str:
    """Mesh(...) constructor string, includes ``names=`` when non-empty."""
    topo = mesh.topology
    layout = mesh.layout
    base = (
        f'Mesh(Topology("{topo.name}", {topo.size}), '
        f"Layout({_shape_tuple(layout.shape)}, {_shape_tuple(layout.strides)})"
    )
    if mesh.names:
        base += f", names={repr(tuple(mesh.names))}"
    return base + ")"


def _shard_layout_str(sl: ShardLayout, indent: str = "") -> str:
    """ShardLayout(...) constructor string, multi-line for readability."""
    layout = sl.layout
    mesh = _mesh_str(sl.mesh)
    attrs = ", ".join(_shard_attr_str(a) for a in sl.attrs)
    shape_tup = _shape_tuple(layout.shape)
    # ``layout.strides`` may be ``None`` for un-materialized sugar; the
    # printer surfaces that explicitly rather than crashing.
    stride_tup = _shape_tuple(layout.strides) if layout.strides is not None else "None"
    return (
        f"ShardLayout(\n"
        f"{indent}    layout=Layout({shape_tup}, {stride_tup}),\n"
        f"{indent}    attrs=({attrs}),\n"
        f"{indent}    mesh={mesh},\n"
        f"{indent})"
    )


def _tensor_annotation(
    ty: TensorType,
    *,
    mesh_name_map: dict[int, str] | None = None,
    indent: str = "",
) -> str:
    """Tensor[(shape), dtype, ShardLayout?, storage?] annotation string.

    When *mesh_name_map* is provided and the layout's mesh has named axes,
    compact sugar form is used instead of verbose ``ShardLayout(...)``.
    """
    base = f'Tensor[{_shape_tuple(ty.shape)}, "{_dtype_str(ty.dtype)}"'
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


# Mapping from HIR Op class names to Python DSL function names.
# Matches the dispatch tables in tilefoundry.parser.dispatch.
_OP_NAME_MAP: dict[str, str] = {
    "MatMul": "matmul",
    "Transpose": "transpose",
    "RMSNorm": "rms_norm",
    "ReLU": "relu",
    "Add": "add",
    "Mul": "mul",
    "Cast": "cast",
    "Reduce": "reduce",
    "Reshard": "reshard",
    "AllReduce": "all_reduce",
    "Reshape": "reshape",
}


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
    3. ``_OP_NAME_MAP`` lookup for legacy callers without
       ``_op_schema`` (kept for backward compat with hand-built fixtures).
    4. CamelCase → snake_case fallback.
    """
    alias_name = _kinded_alias_name(target)
    if alias_name is not None:
        return alias_name
    schema = getattr(target, "_op_schema", None)
    if schema is not None:
        return schema.name
    cls_name = type(target).__name__
    if cls_name in _OP_NAME_MAP:
        return _OP_NAME_MAP[cls_name]
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


def _sanitize_name(name: str) -> str:
    """Make a Python-safe identifier from a loc string."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "v"


def _collect_meshes(fn: HirFunction) -> dict[int, Mesh]:
    """Collect unique Mesh objects from all ShardLayouts in *fn*."""
    meshes: dict[int, Mesh] = {}

    def _add_layout(layout):
        if isinstance(layout, ShardLayout):
            meshes.setdefault(id(layout.mesh), layout.mesh)

    for p in fn.params:
        if isinstance(p.type, TensorType):
            _add_layout(p.type.layout)
    if isinstance(fn.return_type, TensorType):
        _add_layout(fn.return_type.layout)

    def _walk(expr: Expr):
        if isinstance(expr, Call):
            if isinstance(expr.target, Reshard):
                _add_layout(expr.target.layout)
            for arg in expr.args:
                _walk(arg)
        elif isinstance(expr, Tuple):
            for el in expr.elements:
                _walk(el)

    _walk(fn.body)
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

    # Collect all SSA values and assign names
    _counter = [0]
    _names: dict[int, str] = {}
    _seen: set[int] = set()

    # Topological sort: post-order from body
    _order: list[Expr] = []

    def _topo(expr: Expr):
        if expr is None:
            return
        key = id(expr)
        if key in _seen:
            return
        _seen.add(key)
        if isinstance(expr, Call):
            for arg in expr.args:
                _topo(arg)
        elif isinstance(expr, Tuple):
            for el in expr.elements:
                _topo(el)
        _order.append(expr)

    _topo(fn.body)
    for p in fn.params:
        _topo(p)

    # Collect op names first (must be before _assign_name references them)
    _op_names_set: set[str] = set()
    for expr in _order:
        if isinstance(expr, Call):
            _op_names_set.add(_op_name(expr.target))

    def _assign_name(expr: Expr) -> str:
        key = id(expr)
        if key in _names:
            return _names[key]
        if isinstance(expr, Var):
            name = _sanitize_name(expr.name)
        elif isinstance(expr, Call) and expr.loc:
            name = _sanitize_name(expr.loc)
        else:
            name = f"v{_counter[0]}"
            _counter[0] += 1
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

    def _tuple_literal(elements) -> str:
        inner = ", ".join(_names[id(el)] for el in elements)
        if len(elements) == 1:
            inner += ","
        return f"({inner})"

    def _arg_ref(a) -> str:
        # A tuple-valued input (e.g. insert_slice's per-axis offsets) renders
        # inline as a literal so the parser's narrow route lifts it back to a
        # core Tuple on re-parse.
        return _tuple_literal(a.elements) if isinstance(a, Tuple) else _names[id(a)]

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
            ann = _tensor_annotation(p.type, mesh_name_map=mesh_map, indent=indent)
            param_strs.append(f"{indent}{name}: {ann}")
        else:
            param_strs.append(f"{indent}{name}")
    lines.append(",\n".join(param_strs))
    lines.append(f"){arrow}:")

    # A dispatch prototype has no body — declare signature only.
    if fn.body is None:
        lines.append(f"{indent}pass")
        return lines

    for expr in _order:
        if isinstance(expr, Var):
            continue  # params already in signature
        if isinstance(expr, Constant):
            name = _names[id(expr)]
            lines.append(f"{indent}{name} = {repr(expr.value)}")
            continue
        if isinstance(expr, Tuple):
            # A tuple is rendered inline at its use site: as a literal argument
            # (op input) or by the ``return`` statement (function body). The
            # parser lifts an inline offset tuple back to a core Tuple, whereas a
            # hoisted ``name = (...)`` binding would not re-parse.
            continue
        if isinstance(expr, Call):
            name = _names[id(expr)]
            target = expr.target

            if isinstance(target, Reshard):
                # reshard(x, layout=..., storage=...) — kwargs only;
                # ``new_shape`` has been removed.
                src_name = _names[id(expr.args[0])]
                sl = target.layout
                if sl is None:
                    layout_kw = ""
                else:
                    sl_str = _shard_layout_str(sl, indent=indent + "    ")
                    layout_kw = f", layout={sl_str}"
                storage = (
                    f", storage={target.storage.name.lower()}"
                    if target.storage is not None else ""
                )
                loc = f'  # loc="{name}"' if expr.loc else ""
                lines.append(
                    f"{indent}{name} = reshard({src_name}{layout_kw}{storage}){loc}"
                )
            else:
                # Generic op: op_name(arg1, arg2, ..., attr=val)
                op_name_str = _op_name(target)
                arg_names = [_arg_ref(a) for a in expr.args]
                args_str = ", ".join(arg_names)

                # Extract keyword attributes from the op.
                # When the target prints as a kinded alias name
                # (``add`` / ``neg`` / ...), the alias *fixes* the
                # ``kind`` attribute — we must skip it from the kwarg
                # dump, otherwise the regenerated source would end up
                # as ``add(a, b, kind=BinaryKind.ADD)`` which fails to
                # re-parse against the alias schema (and pulls
                # ``BinaryKind`` into the closure unnecessarily).
                _alias_name = _kinded_alias_name(target)
                _suppress_attrs = (
                    {"kind"} if _alias_name is not None else set()
                )
                attrs = {}
                for pi in type(target).params():
                    if pi.kind == "attribute":
                        val = getattr(target, pi.name, None)
                        if (
                            val is not None
                            and pi.name not in ("layout",)
                            and pi.name not in _suppress_attrs
                        ):
                            attrs[pi.name] = val

                attr_strs = []
                for k, v in attrs.items():
                    if isinstance(v, str):
                        attr_strs.append(f'{k}="{v}"')
                    elif isinstance(v, enum.Enum) and isinstance(v.value, str):
                        # A string-valued enum attribute (ReduceKind / DType /
                        # ...) prints as its DSL string value so the source
                        # re-parses without a bare enum reference — the parser
                        # promotes the string back via the ParamDef annotation.
                        attr_strs.append(f'{k}="{v.value}"')
                    elif isinstance(v, float):
                        attr_strs.append(f"{k}={v!r}")
                    elif isinstance(v, ShardLayout):
                        sl_str = _shard_layout_str(v, indent=indent + "        ")
                        attr_strs.append(f"{k}={sl_str}")
                    elif isinstance(v, tuple):
                        attr_strs.append(f"{k}={v}")
                    elif v is None:
                        pass  # skip None attrs
                    else:
                        attr_strs.append(f"{k}={v}")

                call_str = f"{op_name_str}({args_str}"
                if attr_strs:
                    call_str += ", " + ", ".join(attr_strs)
                call_str += ")"

                loc = f'  # loc="{name}"' if expr.loc else ""
                lines.append(f"{indent}{name} = {call_str}{loc}")

    # Return statement. A literal tuple body renders its elements inline
    # (``return (e0, e1)``) rather than a name for the un-emitted Tuple node.
    if isinstance(fn.body, Tuple):
        lines.append(f"{indent}return {_tuple_literal(fn.body.elements)}")
    else:
        body_name = _names[id(fn.body)]
        lines.append(f"{indent}return {body_name}")
    return lines


def _pattern_ctor(pat: Pattern) -> str:
    """Render a Pattern as its constructor, for a ``.specialize(...)`` decorator."""
    if isinstance(pat, DimVarRangePat):
        return f'DimVarRangePat("{pat.dim_var}", {pat.lo}, {pat.hi})'
    return repr(pat)


def hir_function_to_python(fn: HirFunction) -> str:
    """Convert a HIR Function to canonical Python DSL source.

    A normal function prints as a single ``@func``. A dispatch prototype
    (``variants != ()``) prints as a ``pass``-bodied ``@func`` base followed by
    one ``@<name>.specialize(pattern)`` block per variant. When the function
    uses meshes with named axes, compact sugar form is emitted; otherwise the
    verbose ``ShardLayout(...)`` form is used.
    """
    lines: list[str] = []
    indent = "    "

    # Collect meshes across the base and any variants, build the name map.
    meshes: dict = {}
    for f in (fn, *fn.variants):
        meshes.update(_collect_meshes(f))
    mesh_map = _mesh_name_map(meshes)

    # Header — imports
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from tilefoundry import func")
    lines.append("from tilefoundry.dsl.tf import *  # noqa: F401, F403")
    lines.append("from tilefoundry.dsl import Tensor")
    lines.append("from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401")
    lines.append("from tilefoundry.ir.types.shard import (")
    lines.append(f"{indent}B, S, P, Layout, Mesh, ShardLayout, Topology,")
    lines.append(")")
    if fn.variants:
        lines.append("from tilefoundry.ir.core.pattern import DimVarRangePat")
    lines.append("")

    # Emit mesh definitions when sugar is viable (mesh has named axes)
    _has_sugar_mesh = any(m.names for m in meshes.values())
    if _has_sugar_mesh:
        for mid, mesh in meshes.items():
            name = mesh_map[mid]
            topo = mesh.topology
            ml = mesh.layout
            names_repr = repr(tuple(mesh.names)) if mesh.names else "()"
            lines.append(
                f"{name} = Mesh("
                f'Topology("{topo.name}", {topo.size}), '
                f"Layout({_shape_tuple(ml.shape)}, {_shape_tuple(ml.strides)}), "
                f"names={names_repr}"
                f")"
            )
        lines.append("")

    # Base @func decorator (with topologies if any), then the def block.
    if fn.topologies:
        topo_strs = [f'Topology("{t.name}", {t.size})' for t in fn.topologies]
        lines.append(f'@func(topologies=({", ".join(topo_strs)},))')
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

    return "\n".join(lines) + "\n"


def as_script(fn: HirFunction, *, module: str | None = None) -> str:
    """Convert a HIR Function to Python DSL source.

    Without *module*: standalone ``@func`` output.

    With *module* (e.g. ``module="M"``): ``@module(entry="<fn>") class M:``
    wrapper with module-level mesh definitions (the class body stays a pure
    function container) and sugar annotations.

    Args:
        fn: The HIR function.
        module: Optional module class name.  When set, the output is
            wrapped in ``@module(entry="<fn>") class <name>:``.

    Returns:
        Python source string.
    """
    if module is not None:
        return _module_to_python(fn, module)
    return hir_function_to_python(fn)

# backward-compat alias
def module_to_python(fn: HirFunction, module_name: str = "M") -> str:
    """Backward-compat alias for ``as_script(fn, module=module_name)``."""
    return as_script(fn, module=module_name)


def _module_to_python(fn: HirFunction, module_name: str = "M") -> str:
    """Internal: ``@module``-wrapped Python DSL source."""
    lines: list[str] = []
    indent4 = "    "
    indent8 = "        "  # noqa: F841

    # Collect meshes
    meshes = _collect_meshes(fn)
    mesh_map = _mesh_name_map(meshes)
    _has_sugar_mesh = any(m.names for m in meshes.values())

    # Build the function body using the same logic as hir_function_to_python
    # but indented for class body
    func_source = hir_function_to_python(fn)
    func_lines = func_source.rstrip("\n").split("\n")

    # Imports
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from tilefoundry.module import module")
    lines.append("from tilefoundry import func")
    lines.append("from tilefoundry.dsl.tf import *  # noqa: F401, F403")
    lines.append("from tilefoundry.dsl import Tensor")
    lines.append("from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401")
    lines.append("from tilefoundry.ir.types.shard import (")
    lines.append(f"{indent4}B, S, P, Layout, Mesh, ShardLayout, Topology,")
    lines.append(")")
    lines.append("")

    # Shared mesh definitions live at module level (before the class), so the
    # @module class body stays a pure function container and the @func bodies
    # still resolve the meshes via globals.
    if _has_sugar_mesh:
        for mid, mesh in meshes.items():
            name = mesh_map[mid]
            topo = mesh.topology
            ml = mesh.layout
            names_repr = repr(tuple(mesh.names)) if mesh.names else "()"
            lines.append(
                f"{name} = Mesh("
                f'Topology("{topo.name}", {topo.size}), '
                f"Layout({_shape_tuple(ml.shape)}, {_shape_tuple(ml.strides)}), "
                f"names={names_repr}"
                f")"
            )
        lines.append("")

    # @module class header — entry names this function (explicit, required).
    lines.append(f'@module(entry="{fn.name}")')
    lines.append(f"class {module_name}:")

    # Function body (indented into class)
    # Skip standalone imports/header lines from func_source, use module-level ones
    in_func = False
    for fl in func_lines:
        # Skip standalone imports and mesh defs (they're in the module header)
        if fl.startswith("from __future__"):
            continue
        if fl.startswith("from tilefoundry import func"):
            continue
        if fl.startswith("from tilefoundry.ir.types import"):
            continue
        if fl.startswith("from tilefoundry.ir.types.shard import"):
            continue
        if fl.startswith("    B, S, P, Layout"):
            continue
        if fl == ")":
            continue
        if not in_func and fl == "":
            continue
        # Skip standalone mesh definitions (they're in the class body)
        if not in_func and " = Mesh(" in fl:
            continue
        if fl.startswith("@func"):
            in_func = True
            # Preserve @func(topologies=...) when present
            lines.append(f"{indent4}{fl}")
        elif in_func:
            lines.append(f"{indent4}{fl}")

    return "\n".join(lines) + "\n"
