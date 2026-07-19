"""Unified layout sugar parser.

Core model: tuple sugar is a type-directed layout literal.
Shared bottom layer ``_parse_layout_literal()`` extracts shape + strides
from a tuple AST node.  Target-specific entry points lower the literal
to ``Layout`` or ``ShardLayout``.

Consumers (type annotation, ``with Mesh``, body ``reshard`` calls) call
the same parser helpers; the only contextual difference is the
``mesh_resolver`` callable for ``ShardLayout`` sugar.
"""

from __future__ import annotations

import ast
from typing import Any, Callable

from tilefoundry.ir.core.expr import Expr
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh, MeshAxis
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardAttr,
    ShardLayout,
    Split,
)
from tilefoundry.ir.types.storage import StorageKind, resolve_storage


class LayoutSugarError(ValueError):
    """A layout-sugar node was recognized structurally but is malformed
    (e.g. a dynamic ``DimVar`` / ``bool`` static extent).

    It subclasses ``ValueError`` so existing ``except ValueError`` handlers
    still catch it, but callers that speculatively try sugar parsing (and fall
    back to generic static evaluation on a plain ``ValueError``) MUST let this
    propagate so the real diagnostic is not masked by a downstream error.
    """


# ── AST helpers ─────────────────────────────────────────────────────────────


def _is_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant)


def _is_matmul(node: ast.AST) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)


def _is_placeholder(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "_"


def _is_strided_layout_tuple(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and isinstance(node.elts[0], ast.Tuple)
        and isinstance(node.elts[1], ast.Tuple)
    )


def _is_tuple_sugar(node: ast.AST) -> bool:
    """Check whether an AST node is a tuple literal that could be layout sugar.

    Returns True for Tuple nodes (which may contain ``@`` operators).
    Bare Constant (single int) is checked separately by the consumer
    based on whether meshes are available.
    """
    if isinstance(node, ast.Tuple):
        return True
    return False


def _has_sugar(node: ast.AST) -> bool:
    """Check whether an AST node contains a ``@`` sugar operator."""
    found = False

    def visitor(n: ast.AST):
        nonlocal found
        if found:
            return
        if _is_matmul(n):
            found = True
            return
        for _field, child in ast.iter_fields(n):
            if isinstance(child, ast.AST):
                visitor(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, ast.AST):
                        visitor(item)

    visitor(node)
    return found


def _eval_ast(node: ast.AST, closure: dict[str, Any] | None = None) -> Any:
    """Evaluate a Python literal AST node (int, str, tuple of literals).

    Also accepts ``DimVar("S", lo, hi)`` inline ``Call`` nodes and
    closure-resolved ``Name`` references to ``DimVar`` instances.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(e, closure) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand, closure)
    if closure is not None:
        if isinstance(node, ast.Name):
            val = closure.get(node.id)
            if val is not None:
                return val
        elif isinstance(node, ast.Attribute):
            obj = _eval_ast(node.value, closure)
            return getattr(obj, node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DimVar":
            from tilefoundry.ir.types.dim import DimVar  # noqa: PLC0415
            pos = [_eval_ast(a, closure) for a in node.args]
            kw = {k.arg: _eval_ast(k.value, closure) for k in node.keywords}
            return DimVar(*pos, **kw)
    raise ValueError(f"cannot evaluate static value from {ast.dump(node)}")


def _is_shape_dim(v: Any) -> bool:
    """True for a valid layout axis extent: ``ShapeDim = int | DimVar | Expr``.

    ``bool`` is a subclass of ``int`` but is rejected (never a real extent).
    """
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, DimVar, Expr))


def _name_of(node: ast.AST) -> str:
    """Extract bare Name id from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    raise ValueError(f"expected Name, got {ast.dump(node)}")


def _auto_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """C-order contiguous strides: ``(d0, d1, d2)`` → ``(d1*d2, d2, 1)``."""
    if not shape:
        return ()
    strides = [1]
    for d in reversed(shape[1:]):
        strides.insert(0, strides[0] * d)
    return tuple(strides)


def _resolve_dtype_ast(node: ast.AST, closure: dict[str, Any]) -> DType | None:
    """Resolve a dtype from an AST node (bare name, string, or DType.attr)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        member = getattr(DType, node.value, None)
        return member if isinstance(member, DType) else None
    if isinstance(node, ast.Name):
        val = closure.get(node.id)
        if isinstance(val, DType):
            return val
        return getattr(DType, node.id, None)
    if isinstance(node, ast.Attribute):
        try:
            val = _eval_ast(node, closure)
            if isinstance(val, DType):
                return val
        except ValueError:
            pass
    return None


# ── mesh axis resolution ───────────────────────────────────────────────────


def _resolve_mesh(name: str, mesh_by_name: dict[str, Mesh]) -> Mesh:
    """Look up a Mesh by variable name."""
    mesh = mesh_by_name.get(name)
    if mesh is None:
        available = list(mesh_by_name.keys())
        raise ValueError(f"undefined mesh {name!r}; available: {available}")
    if not isinstance(mesh, Mesh):
        raise ValueError(f"{name!r} is not a Mesh, got {type(mesh).__name__}")
    return mesh


def _resolve_mesh_axis(mesh: Mesh, axis_name: str) -> MeshAxis:
    """Resolve a mesh axis by name (preferred) or x/y/z position fallback.

    - If *mesh* has ``names``, resolve by matching name.
    - If *axis_name* is ``"x"``, ``"y"``, or ``"z"``, resolve by position.
    - Otherwise, raises ``ValueError``.
    """
    # First try named axis
    axis = mesh.axis_named(axis_name)
    if axis is not None:
        return axis
    # Fallback: x/y/z positional
    if axis_name == "x":
        return mesh.x
    if axis_name == "y":
        return mesh.y
    if axis_name == "z":
        return mesh.z
    available = list(mesh.names) if mesh.names else ["x", "y", "z"][: len(mesh.axes)]
    raise ValueError(
        f"mesh has no axis named {axis_name!r}; available: {available}"
    )


# ── layout literal parser (shared bottom layer) ────────────────────────────


def _parse_layout_literal(
    node: ast.AST, *, closure: dict[str, Any] | None = None
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Parse a tuple AST node into (shape, strides_or_none).

    This is the shared bottom layer.  It does NOT choose a final type.

    Forms::

        (d0, d1, ...)              → (shape, None)       auto strides
        ((d0, d1,...), (s0,s1,...)) → (shape, strides)    explicit strides
        N                           → ((N,), None)        single-element 1D

    When dim elements contain ``@`` sugar operators, only the left-hand
    int constant is extracted for the shape; axis binding is handled by
    the target-specific lowering (e.g. ``parse_shard_layout_sugar``).

    ``closure`` lets a dim that is a closure/global ``Name`` (e.g. ``WARPS``)
    resolve to its static int value.
    """
    if isinstance(node, ast.Tuple):
        if len(node.elts) == 2 and isinstance(node.elts[0], ast.Tuple) and isinstance(node.elts[1], ast.Tuple):
            # Full form: ((dims), (strides))
            dim_nodes = list(node.elts[0].elts)
            strides = _eval_ast(node.elts[1], closure)
            shape = tuple(_extract_dim_int(dn, closure=closure) for dn in dim_nodes)
        else:
            # Short form: (dims)
            dim_nodes = list(node.elts)
            strides = None
            shape = tuple(_extract_dim_int(dn, closure=closure) for dn in dim_nodes)
    elif _is_constant(node):
        shape = (node.value,)
        strides = None
    else:
        raise ValueError(f"expected tuple layout literal, got {ast.dump(node)}")

    if not all(isinstance(d, int) for d in shape):
        raise ValueError(f"layout shape must be all ints, got {shape}")

    if strides is not None:
        if not isinstance(strides, tuple):
            raise ValueError(f"strides must be a tuple, got {strides!r}")
        if len(strides) != len(shape):
            raise ValueError(
                f"strides rank {len(strides)} != layout shape rank {len(shape)}"
            )

    return shape, strides


def _extract_dim_int(node: ast.AST, *, closure: dict[str, Any] | None = None) -> int:
    """Extract the static-int dimension from a layout dim node.

    Handles: plain ``Constant(32)``, closure/global ``Name`` references bound to
    an ``int`` (e.g. ``WARPS = 4``), and ``BinOp(<dim>, MatMult, ...)`` sugar
    forms where only the left operand is the dimension. The resolved value MUST
    be a static ``int``; ``bool`` and dynamic (``DimVar``) extents are rejected
    with a clear diagnostic rather than a raw AST / attribute error.
    """
    dim_node = node.left if _is_matmul(node) else node
    if _is_constant(dim_node):
        val = dim_node.value
    else:
        try:
            val = _eval_ast(dim_node, closure)
        except ValueError:
            raise ValueError(f"expected int dim, got {ast.dump(node)}") from None
    if isinstance(val, bool) or not isinstance(val, int):
        raise LayoutSugarError(f"layout dim must be a static int, got {val!r}")
    return val


# ── target-specific sugar parsers ──────────────────────────────────────────


def parse_layout_sugar(node: ast.AST) -> Layout:
    """Parse a tuple sugar as a plain ``Layout``.

    >>> parse_layout_sugar(ast.parse("(1, 1536)", mode="eval").body)
    Layout(shape=(1, 1536), strides=(1536, 1))
    """
    shape, strides = _parse_layout_literal(node)
    if strides is None:
        strides = _auto_strides(shape)
    return Layout(shape=shape, strides=strides)


def parse_mesh_layout_sugar(
    node: ast.AST, *, closure: dict[str, Any] | None = None
) -> Layout:
    """Parse a mesh's layout tuple sugar as a ``Layout``.

    ``closure`` lets mesh dims be closure/global ``int`` names (e.g. a
    ``WARPS = 4`` constant used as ``(WARPS, LANES)``).

    >>> parse_mesh_layout_sugar(ast.parse("(128,)", mode="eval").body)
    Layout(shape=(128,), strides=(1,))
    """
    shape, strides = _parse_layout_literal(node, closure=closure)
    if strides is None:
        strides = _auto_strides(shape)
    return Layout(shape=shape, strides=strides)


# Type alias for the mesh resolver callback used by ShardLayout sugar.
MeshResolver = Callable[[str], Mesh]


def parse_shard_layout_sugar(
    node: ast.AST,
    mesh_resolver: MeshResolver,
    *,
    default_mesh: Mesh | None = None,
    closure: dict[str, Any] | None = None,
) -> ShardLayout:
    """Parse a tuple sugar as a ``ShardLayout``.

    *mesh_resolver* is called with a mesh variable name (e.g. ``"gpu"`` or
    ``"thread_mesh"``) and must return the corresponding ``Mesh`` object.

    *default_mesh* is used when no ``@ mesh.axis`` bindings are present
    (all-Broadcast layout).  If both bindings and *default_mesh* are
    absent, parsing fails.

    Surface syntax (axis-tuple carries placement; an optional final ``{...}``
    set carries mesh-axis ``Partial`` value states):
    - bare int → physical layout dim, not split (Broadcast on the mesh axes)
    - ``dim @ mesh.axis`` → physical layout dim split on that mesh axis
    - ``{mesh.axis @ P("reduction"), ...}`` → mesh-axis Partial value states
    A mesh axis named in no Split and no Partial is Broadcast (the default).

    Returns a ``ShardLayout`` with mesh-rank ``attrs`` (§8.2).
    """
    axis_node, strides, value_set_node = _split_layout_outer(node)

    dim_nodes = _get_dim_nodes(axis_node)
    # Verbose ``((dims), (strides))`` form (user-supplied strides) MUST NOT
    # be canonicalized; see parser.md §1.5.
    canonicalize = strides is None
    parsed: list[_LayoutItem] = []
    for dn in dim_nodes:
        parsed.extend(
            _parse_layout_item(dn, mesh_resolver, canonicalize=canonicalize, closure=closure)
        )
    # ``{mesh.axis @ P("reduction")}`` value-state set (mesh-axis Partials).
    value_states = (
        _parse_value_state(value_set_node, mesh_resolver)
        if value_set_node is not None
        else []
    )

    shape: list[int] = []
    # Collect the unique mesh from any bindings (layout splits + value states).
    resolved_mesh: Mesh | None = None
    for _d, mesh, _mi, _k, _r in parsed:
        if mesh is None:
            continue
        if resolved_mesh is None:
            resolved_mesh = mesh
        elif id(mesh) != id(resolved_mesh):
            raise ValueError("all layout dims must reference the same mesh")
    for mesh, _mi, _r in value_states:
        if resolved_mesh is None:
            resolved_mesh = mesh
        elif id(mesh) != id(resolved_mesh):
            raise ValueError("value-state mesh must match the layout mesh")

    if resolved_mesh is None:
        if default_mesh is not None:
            resolved_mesh = default_mesh
        else:
            raise ValueError(
                "all-Broadcast ShardLayout sugar requires a mesh from "
                "context; use verbose ShardLayout(...) to disambiguate"
            )

    mesh_rank = len(resolved_mesh.axes)
    attrs_list: list[ShardAttr] = [Broadcast() for _ in range(mesh_rank)]

    for dim, _mesh, m_axis, kind, _reduction in parsed:
        if dim is not None:
            shape.append(dim)
        layout_axis = len(shape) - 1
        if kind == "split":
            if m_axis is None or m_axis >= mesh_rank:
                raise ValueError(f"layout dim {layout_axis}: invalid mesh axis {m_axis}")
            if not isinstance(attrs_list[m_axis], Broadcast):
                raise ValueError(
                    f"mesh axis {m_axis} already bound; "
                    "one layout dim per mesh axis (§8.4)"
                )
            attrs_list[m_axis] = Split(layout_axis)

    for _mesh, m_axis, reduction in value_states:
        if m_axis >= mesh_rank:
            raise ValueError(f"value-state: invalid mesh axis {m_axis}")
        if not isinstance(attrs_list[m_axis], Broadcast):
            raise ValueError(f"mesh axis {m_axis} already bound")
        attrs_list[m_axis] = Partial(reduction or "sum")

    # Sugar (`strides is None`) leaves the layout strides un-materialized;
    # `Reshard` typeinfer / `Function` signature binding discharges them
    # per spec docs/spec/hir.md §3 / §1. Verbose `((shape),(strides))`
    # carries explicit strides verbatim.
    return ShardLayout(
        layout=Layout(shape=tuple(shape), strides=strides),
        attrs=tuple(attrs_list),
        mesh=resolved_mesh,
    )


def _split_layout_outer(
    node: ast.AST,
) -> tuple[ast.AST, "tuple | None", "ast.Set | None"]:
    """Split a layout-sugar node into (axis-tuple node, strides, value-state set).

    Outer-tuple grammar (parser layout sugar):
    - ``(d0, d1, ...)``                       → implicit strides, no value-state
    - ``((dims), (strides))``                 → explicit strides
    - ``((dims), {value-state})``             → implicit strides + value-state
    - ``((dims), (strides), {value-state})``  → explicit strides + value-state

    The value-state `set` literal (if present) MUST be the last outer item.
    """
    if _is_constant(node) or _is_matmul(node):
        return node, None, None
    if not isinstance(node, ast.Tuple):
        raise ValueError(f"expected tuple layout, got {ast.dump(node)}")
    # Outer form: the first element is itself the axis-tuple. (A bare axis-spec
    # is a Constant / BinOp, never a Tuple, so this is unambiguous.)
    if node.elts and isinstance(node.elts[0], ast.Tuple):
        axis_node = node.elts[0]
        strides = None
        value_set: ast.Set | None = None
        for elt in node.elts[1:]:
            if value_set is not None:
                # The value-state set MUST be the final outer item.
                raise ValueError(
                    "layout sugar: the value-state set must be the last outer item"
                )
            if isinstance(elt, ast.Set):
                value_set = elt
            elif isinstance(elt, ast.Tuple):
                if strides is not None:
                    raise ValueError("layout sugar: at most one stride tuple")
                strides = _eval_ast(elt)
            else:
                raise ValueError(
                    f"layout sugar outer item must be a stride tuple or value-state "
                    f"set, got {ast.dump(elt)}"
                )
        return axis_node, strides, value_set
    # Flat axis-tuple.
    return node, None, None


def _parse_value_state(
    node: "ast.Set", mesh_resolver: MeshResolver
) -> list[tuple[Mesh, int, str]]:
    """Parse a ``{mesh.axis @ P("reduction"), ...}`` value-state set into a list
    of ``(mesh, mesh_axis_index, reduction)``. Element order carries no meaning."""
    if not isinstance(node, ast.Set):
        raise ValueError(f"value-state must be a set literal, got {ast.dump(node)}")
    out: list[tuple[Mesh, int, str]] = []
    for elt in node.elts:
        if not (
            _is_matmul(elt)
            and isinstance(elt.right, ast.Call)
            and isinstance(elt.right.func, ast.Name)
            and elt.right.func.id == "P"
        ):
            raise ValueError(
                'value-state entry must be `mesh.axis @ P("reduction")`, got '
                f"{ast.dump(elt)}"
            )
        if len(elt.right.args) != 1:
            raise ValueError(
                'value-state P(...) requires exactly one reduction argument, '
                'e.g. `mesh.axis @ P("sum")`'
            )
        mesh_name, axis_name = _parse_axis_ref(elt.left)
        mesh = mesh_resolver(mesh_name)
        if mesh is None:
            raise ValueError(f"undefined mesh {mesh_name!r}")
        axis = _resolve_mesh_axis(mesh, axis_name)
        reduction = _eval_ast(elt.right.args[0])
        out.append((mesh, axis.index, reduction))
    return out


def _get_dim_nodes(node: ast.AST) -> list[ast.AST]:
    """Extract dimension sub-nodes from a layout sugar tuple.

    Accepts both Tuple and BinOp (for standalone sugar like
    ``1536 @ (m.w, m.t)`` without a wrapping tuple).
    """
    if isinstance(node, ast.Tuple):
        if len(node.elts) == 2 and isinstance(node.elts[0], ast.Tuple) and isinstance(node.elts[1], ast.Tuple):
            return list(node.elts[0].elts)
        return list(node.elts)
    if _is_constant(node) or _is_matmul(node):
        return [node]
    raise ValueError(f"expected tuple layout, got {ast.dump(node)}")


# Type alias for a single parsed layout item.
_LayoutItem = tuple[int | None, Mesh | None, int | None, str, str | None]


def _parse_layout_item(
    node: ast.AST,
    mesh_resolver: MeshResolver,
    *,
    canonicalize: bool = True,
    closure: dict[str, Any] | None = None,
) -> list[_LayoutItem]:
    """Parse a single layout-dim element into one or more layout items.

    Returns a list of (dim_size_or_none, mesh, mesh_axis_index, kind, reduction).
    The axis-tuple carries only placement; value states (`Partial`) live in the
    separate ``{...}`` set parsed by ``_parse_value_state``.

    Forms::
        dim                              → [(dim, None, None, "broadcast", None)]
        dim @ mesh.axis                  → [(dim, mesh, axis_idx, "split", None)]
        dim @ (mesh.axis, ...)           → [split items…, bare remainder item]
    """
    # Case 1: bare dim (literal int)
    if _is_constant(node):
        return [(node.value, None, None, "broadcast", None)]

    # Case 2+3: dim @ ...
    if _is_matmul(node):
        rhs = node.right
        dim = None if _is_placeholder(node.left) else _eval_ast(node.left, closure)
        if dim is None:
            raise ValueError(
                "layout placeholder `_` is not valid in the axis tuple; "
                'value states go in the `{mesh.axis @ P("reduction")}` set'
            )
        if not isinstance(dim, int) or isinstance(dim, bool):
            # A split axis (``dim @ mesh.axis``) participates in
            # canonicalisation (factorisation against the mesh extent), so it
            # must resolve to a static int. A dynamic (DimVar) split axis is
            # not expressible in v1 sugar — only bare axes may be dynamic.
            raise LayoutSugarError(
                f"split layout dim `dim @ mesh.axis` must be a static int, got {dim!r}"
            )

        # Case 2: dim @ (mesh.axis, ...) — sequential decomposition sugar
        if isinstance(rhs, ast.Tuple):
            return _expand_multi_axis_sugar(dim, rhs.elts, mesh_resolver)

        # Case 3: dim @ mesh.axis
        if isinstance(rhs, ast.Attribute):
            mesh_name, axis_name = _parse_axis_ref(rhs)
            mesh = mesh_resolver(mesh_name)
            if mesh is None:
                raise ValueError(f"undefined mesh {mesh_name!r}")
            axis = _resolve_mesh_axis(mesh, axis_name)
            if not canonicalize:
                return [(dim, mesh, axis.index, "split", None)]
            return _canonicalize_single_axis(dim, mesh, axis)

        # Case 3-bis (``int @ mesh`` shorthand for single-axis mesh):
        # ``8192 @ cta`` resolves to ``8192 @ cta.<only-axis>`` when the
        # named mesh has exactly one axis. Multi-axis meshes still
        # require an explicit ``mesh.axis`` reference.
        if isinstance(rhs, ast.Name):
            mesh = mesh_resolver(rhs.id)
            if mesh is None:
                raise ValueError(f"undefined mesh {rhs.id!r}")
            axes = mesh.axes
            if len(axes) != 1:
                raise ValueError(
                    f"``int @ {rhs.id}`` shorthand requires a single-axis mesh "
                    f"(found {len(axes)} axes); write ``{rhs.id}.<axis>`` explicitly"
                )
            if not canonicalize:
                return [(dim, mesh, 0, "split", None)]
            return _canonicalize_single_axis(dim, mesh, axes[0])

    # Case 4: bare dynamic / closure-resolved axis (a DimVar like ``S`` or a
    # closure Name bound to an int / DimVar). A bare axis is Broadcast — it
    # carries no mesh binding — so a dynamic extent is fine (unlike a split
    # axis); strides stay deferred to Reshard typeinfer.
    if closure is not None:
        try:
            dim = _eval_ast(node, closure)
        except ValueError:
            dim = None
        if _is_shape_dim(dim):
            return [(dim, None, None, "broadcast", None)]

    raise ValueError(f"unexpected layout dim AST: {ast.dump(node)}")


def _canonicalize_single_axis(
    dim: int,
    mesh: Mesh,
    axis: MeshAxis,
) -> list[_LayoutItem]:
    """Canonicalize ``N @ m.a`` into the factorised pair when ``N > mesh_extent(a)``.

    Per parser.md §1.5 / shard.md §7.1.1: surface sugar ``N @ m.a`` where
    ``N > mesh_extent(a)`` MUST be expanded into
    ``(mesh_extent(a) @ m.a, N // mesh_extent(a))`` so every Split-bound
    layout dim has ``local_shape == 1``. ``N % mesh_extent(a) == 0`` is
    required; otherwise ``ValueError``.
    """
    extent = axis.size
    if dim % extent != 0:
        raise ValueError(
            f"dim {dim} not divisible by mesh extent {extent} on axis "
            f"{axis.index}; cannot canonicalize ``{dim} @ m.<axis>``"
        )
    if dim == extent:
        return [(dim, mesh, axis.index, "split", None)]
    residual = dim // extent
    return [
        (extent, mesh, axis.index, "split", None),
        (residual, None, None, "broadcast", None),
    ]


def _expand_multi_axis_sugar(
    dim: int,
    axis_nodes: list[ast.AST],
    mesh_resolver: MeshResolver,
) -> list[_LayoutItem]:
    """Expand ``dim @ (mesh.axis, ...)`` into split + remainder items.

    Each mesh axis gets extent = mesh_extent (Split).  The remainder
    ``dim / ∏(mesh_extents)`` becomes a bare (Broadcast) value axis
    appended at the end.

    Raises ``ValueError`` if *dim* is not divisible by the product of
    all mesh extents.
    """
    items: list[_LayoutItem] = []
    remaining = dim

    for i, ax_node in enumerate(axis_nodes):
        mesh, axis = _resolve_axis_node(ax_node, mesh_resolver)
        extent = axis.size
        if remaining % extent != 0:
            raise ValueError(
                f"dim {dim} not divisible by mesh extent {extent} "
                f"at axis position {i}; remaining={remaining}"
            )
        per_axis = extent
        remaining //= extent
        items.append((per_axis, mesh, axis.index, "split", None))

    if remaining > 0:
        items.append((remaining, None, None, "broadcast", None))

    return items


def _resolve_axis_node(
    node: ast.AST,
    mesh_resolver: MeshResolver,
) -> tuple[Mesh, MeshAxis]:
    """Resolve a mesh-axis reference node to (Mesh, MeshAxis).

    Accepts ``mesh.axis`` attribute references and single-axis
    ``mesh`` name shorthand.
    """
    if isinstance(node, ast.Attribute):
        mesh_name, axis_name = _parse_axis_ref(node)
        mesh = mesh_resolver(mesh_name)
        if mesh is None:
            raise ValueError(f"undefined mesh {mesh_name!r}")
        axis = _resolve_mesh_axis(mesh, axis_name)
        return (mesh, axis)
    if isinstance(node, ast.Name):
        mesh = mesh_resolver(node.id)
        if mesh is None:
            raise ValueError(f"undefined mesh {node.id!r}")
        axes = mesh.axes
        if len(axes) != 1:
            raise ValueError(
                f"``int @ (..., {node.id}, ...)`` shorthand requires a "
                f"single-axis mesh (found {len(axes)} axes); "
                f"write ``{node.id}.<axis>`` explicitly"
            )
        return (mesh, axes[0])
    raise ValueError(f"expected mesh.axis, got {ast.dump(node)}")


def _parse_axis_ref(node: ast.AST) -> tuple[str, str]:
    """Parse a mesh-qualified axis reference.

    ``gpu.cluster`` → ``("gpu", "cluster")``
    ``gpu.x``       → ``("gpu", "x")``
    """
    if isinstance(node, ast.Attribute):
        mesh_name = _name_of(node.value)
        axis_name = node.attr
        return (mesh_name, axis_name)
    raise ValueError(f"expected mesh.axis (e.g. gpu.cluster), got {ast.dump(node)}")


# ── legacy parse_sugar_layout alias (internal compat) ─────────────────────


def parse_sugar_layout(
    node: ast.AST,
    mesh_by_name: dict[str, Mesh],
) -> tuple[tuple[int, ...], tuple[int, ...], Mesh, tuple[ShardAttr, ...]]:
    """Legacy alias for ``parse_shard_layout_sugar`` that returns a raw
    (shape, strides, mesh, attrs) tuple.

    Prefer ``parse_shard_layout_sugar(node, mesh_by_name.get)`` over this
    function.  Kept for backward compatibility.
    """
    sl = parse_shard_layout_sugar(node, mesh_by_name.get)
    return (sl.layout.shape, sl.layout.strides, sl.mesh, tuple(sl.attrs))


# ── top-level Tensor annotation parser ─────────────────────────────────────


def try_parse_sugar_tensor_type(
    node: ast.AST,
    closure: dict[str, Any],
) -> TensorType | None:
    """Parse a ``Tensor[...]`` or ``ConstTensor[...]`` annotation with sugar
    layout.

    ``ConstTensor[...]`` resolves to the same ``TensorType`` as
    ``Tensor[...]``; the caller reads the annotation head name separately to
    set ``Var.is_const``.

    Returns the parsed ``TensorType`` on success, or ``None`` if the
    annotation is not in sugar form (caller falls through to ``eval()``
    for the verbose ``ShardLayout(...)`` path).
    """
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id not in (
        "Tensor", "ConstTensor",
    ):
        return None
    if not isinstance(node.slice, ast.Tuple):
        return None
    elts = node.slice.elts
    if len(elts) < 2:
        return None

    # Check for sugar markers
    has_sugar = any(_has_sugar(elt) for elt in elts[2:])

    # Build mesh-by-name table from closure
    mesh_by_name: dict[str, Mesh] = {}
    for key, val in closure.items():
        if isinstance(val, Mesh):
            mesh_by_name[key] = val

    # Also treat bare-tuple/bare-constant as sugar when meshes are available
    if not has_sugar and len(elts) >= 3:
        third = elts[2]
        if mesh_by_name and (isinstance(third, ast.Tuple) or _is_constant(third)):
            has_sugar = True

    if not has_sugar:
        return None

    try:
        shape = _eval_ast(elts[0])
    except ValueError:
        return None
    if not isinstance(shape, tuple):
        return None

    dtype_val = _resolve_dtype_ast(elts[1], closure)
    if dtype_val is None:
        return None

    layout = None
    storage = StorageKind.GMEM
    if len(elts) >= 3:
        # Use the single mesh as default for all-Broadcast layouts
        default_mesh = next(iter(mesh_by_name.values())) if len(mesh_by_name) == 1 else None
        sl = parse_shard_layout_sugar(elts[2], mesh_by_name.get, default_mesh=default_mesh)
        # Function signature binding: the
        # ``Tensor[..., (sugar)]`` annotation sits at the kernel
        # boundary where the underlying engine is a shared FFI
        # buffer. Materialize any ``Layout.strides=None`` from the
        # sugar path into shared-engine C-order over the canonical
        # global shape before the resulting TensorType enters the
        # body. Verbose ``((shape), (strides))`` annotations are
        # preserved verbatim because ``parse_shard_layout_sugar``
        # already produces a concrete strides tuple in that case.
        if sl.layout.strides is None:
            sl = ShardLayout(
                layout=Layout(
                    shape=sl.layout.shape,
                    strides=_auto_strides(sl.layout.shape),
                ),
                attrs=sl.attrs,
                mesh=sl.mesh,
            )
        layout = sl
    if len(elts) >= 4:
        try:
            storage = resolve_storage(_eval_ast(elts[3], closure))
        except ValueError:
            return None

    return TensorType(shape=shape, dtype=dtype_val, layout=layout, storage=storage)
