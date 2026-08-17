"""Unified layout sugar parser.

Core model: tuple sugar is a type-directed layout literal.
Shared bottom layer ``_parse_layout_literal()`` extracts shape + strides
from a tuple AST node.  Target-specific entry points lower the literal
to ``Layout`` or ``ShardLayout``.

Consumers call :func:`parse_sugar` with the expected result type; contextual
differences are closure lookup and mesh resolution.
"""

from __future__ import annotations

import ast
from typing import Any, Callable

from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.expr import Expr
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimMul, DimVar, simplify_dim
from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shard import c_order_strides
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardAttr,
    ShardLayout,
    Split,
)
from tilefoundry.ir.types.storage import StorageKind, resolve_storage
from tilefoundry.utils.spec_ref import spec_ref_render

from .static_eval import eval_static

_SHARD_ATTR = "[shard §6](docs/spec/shard.md#6-shardattr)"


class LayoutSugarError(VerifyError):
    """Report a structurally recognized but malformed layout-sugar node.

    A layout-sugar node was recognized structurally but is malformed
    (e.g. a dynamic ``DimVar`` / ``bool`` static extent).

    It subclasses ``VerifyError`` (itself a ``ValueError``) so both
    still catch it, but callers that speculatively try sugar parsing (and fall
    back to generic static evaluation on a plain ``ValueError``) MUST let this
    propagate so the real diagnostic is not masked by a downstream error.
    """


def _is_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant)


def _is_layout_slot_constant(node: ast.AST) -> bool:
    """Whether a constant in the third annotation slot is a layout rather than storage.

    ``Tensor[shape, dtype, storage]`` puts a storage name where a layout would
    go, and ``Tensor[shape, dtype, None, storage]`` leaves that slot empty
    ([parser §1.4](docs/spec/parser.md#14-tensor-and-consttensor-annotations)
    makes both optional and independent). Neither is layout sugar, so neither
    may pull the annotation onto the sugar path.
    """
    return _is_constant(node) and node.value is not None and not isinstance(node.value, str)


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


_EVAL_AST_NODES_NO_CLOSURE = (ast.Constant, ast.Tuple, ast.UnaryOp)
_EVAL_AST_NODES_WITH_CLOSURE = (
    *_EVAL_AST_NODES_NO_CLOSURE,
    ast.Name,
    ast.Attribute,
    ast.Call,
    ast.BinOp,
)


def _eval_ast(node: ast.AST, closure: dict[str, Any] | None = None) -> Any:
    """Evaluate layout literals, inline dimensions, and closure dimensions.

    Translate the shared static evaluator's ``VerifyError`` to ``ValueError``.
    Names and attributes resolve only when a closure is supplied; inline
    ``DimVar`` calls are recognized directly rather than as general callees.
    """
    if (
        closure is not None
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DimVar"
    ):
        pos = [_eval_ast(a, closure) for a in node.args]
        kw = {k.arg: _eval_ast(k.value, closure) for k in node.keywords}
        return DimVar(*pos, **kw)
    allowed = _EVAL_AST_NODES_WITH_CLOSURE if closure is not None else _EVAL_AST_NODES_NO_CLOSURE
    try:
        return eval_static(node, closure=closure or {}, allowed_nodes=allowed)
    except VerifyError as exc:
        raise VerifyError(str(exc)) from exc


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
    raise VerifyError(f"expected Name, got {ast.dump(node)}")


def _dim_mul(a: ShapeDim, b: ShapeDim) -> ShapeDim:
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    return simplify_dim(DimMul, (a, b))


def _auto_strides(shape: tuple[ShapeDim, ...]) -> tuple[ShapeDim, ...]:
    """C-order contiguous strides: ``(d0, d1, d2)`` → ``(d1*d2, d2, 1)``."""
    return c_order_strides(shape, mul=_dim_mul)


def _resolve_dtype_ast(node: ast.AST, closure: dict[str, Any]) -> DType | None:
    """Resolve a dtype from an AST node (bare name, string, or DType.attr)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return DType._members().get(node.value)
    if isinstance(node, ast.Name):
        val = closure.get(node.id)
        if isinstance(val, DType):
            return val
        if isinstance(val, str):
            return DType._members().get(val)
        return DType._members().get(node.id)
    if isinstance(node, ast.Attribute):
        try:
            val = _eval_ast(node, closure)
            if isinstance(val, DType):
                return val
        except ValueError:
            pass
    return None


def _resolve_mesh(name: str, mesh_by_name: dict[str, Mesh]) -> Mesh:
    """Look up a Mesh by variable name."""
    mesh = mesh_by_name.get(name)
    if mesh is None:
        available = list(mesh_by_name.keys())
        raise VerifyError(f"undefined mesh {name!r}; available: {available}")
    if not isinstance(mesh, Mesh):
        raise VerifyError(f"{name!r} is not a Mesh, got {type(mesh).__name__}")
    return mesh


def _resolve_mesh_axis(mesh: Mesh, axis_name: str) -> int:
    """Resolve a mesh axis by name (preferred) or x/y/z position fallback.

    - If *mesh* has ``names``, resolve by matching name.
    - If *axis_name* is ``"x"``, ``"y"``, or ``"z"``, resolve by position.
    - Otherwise, raises ``ValueError``.
    """
    for index, name in enumerate(mesh.names):
        if name == axis_name:
            return index
    if axis_name in ("x", "y", "z"):
        index = ("x", "y", "z").index(axis_name)
        if index < len(mesh.layout.shape):
            return index
    available = list(mesh.names) if mesh.names else ["x", "y", "z"][: len(mesh.layout.shape)]
    raise VerifyError(f"mesh has no axis named {axis_name!r}; available: {available}")


def _parse_layout_literal(
    node: ast.AST, *, closure: dict[str, Any] | None = None
) -> tuple[tuple[ShapeDim, ...], tuple[ShapeDim, ...] | None]:
    """Parse layout shape and optional strides without choosing a final type.

    Accept flat dimensions, explicit ``(dimensions, strides)``, or a scalar 1-D
    form. For placement sugar, extract the left-hand dimension while a target
    parser handles axis binding. A closure may resolve named static dimensions.
    """
    if isinstance(node, ast.Tuple):
        if (
            len(node.elts) == 2
            and isinstance(node.elts[0], ast.Tuple)
            and isinstance(node.elts[1], ast.Tuple)
        ):
            dim_nodes = list(node.elts[0].elts)
            strides = _eval_ast(node.elts[1], closure)
            shape = tuple(_extract_dim(dn, closure=closure) for dn in dim_nodes)
        else:
            dim_nodes = list(node.elts)
            strides = None
            shape = tuple(_extract_dim(dn, closure=closure) for dn in dim_nodes)
    elif _is_constant(node):
        shape = (_extract_dim(node, closure=closure),)
        strides = None
    else:
        try:
            resolved = _eval_ast(node, closure)
        except ValueError:
            resolved = None
        if not isinstance(resolved, tuple) or not all(_is_shape_dim(dim) for dim in resolved):
            raise VerifyError(f"expected tuple layout literal, got {ast.dump(node)}")
        shape = resolved
        strides = None

    if strides is not None:
        if not isinstance(strides, tuple):
            raise VerifyError(f"strides must be a tuple, got {strides!r}")
        if len(strides) != len(shape):
            raise VerifyError(f"strides rank {len(strides)} != layout shape rank {len(shape)}")

    return shape, strides


def _extract_dim(node: ast.AST, *, closure: dict[str, Any] | None = None) -> ShapeDim:
    """Extract a static or symbolic dimension from a layout dim node.

    Handles: plain ``Constant(32)``, closure/global ``Name`` references bound to
    a ``ShapeDim``, and ``BinOp(<dim>, MatMult, ...)`` sugar forms where only the
    left operand is the dimension. ``bool`` and non-dimension values are rejected
    with a clear diagnostic rather than a raw AST / attribute error.
    """
    dim_node = node.left if _is_matmul(node) else node
    if _is_constant(dim_node):
        val = dim_node.value
    else:
        try:
            val = _eval_ast(dim_node, closure)
        except ValueError:
            raise VerifyError(f"expected shape dimension, got {ast.dump(node)}") from None
    if not _is_shape_dim(val):
        if isinstance(val, bool):
            raise LayoutSugarError(
                f"layout dim must be a shape dimension, and bool {val!r} is not one; bool is "
                "an int subclass, so it is refused here rather than silently becoming 0 or 1"
            )
        raise LayoutSugarError(
            f"layout dim must be a shape dimension (int / DimVar / dim-op Expr), got "
            f"{type(val).__name__} {val!r}"
        )
    return val


def _parse_layout_sugar(node: ast.AST, *, closure: dict[str, Any] | None = None) -> Layout:
    shape, strides = _parse_layout_literal(node, closure=closure)
    if strides is None:
        strides = _auto_strides(shape)
    return Layout(shape=shape, strides=strides)


MeshResolver = Callable[[str], Mesh]


def _parse_shard_layout_sugar(
    node: ast.AST,
    mesh_resolver: MeshResolver,
    *,
    default_mesh: Mesh | None = None,
    closure: dict[str, Any] | None = None,
) -> ShardLayout:
    """Parse placement and value-state sugar into a shard layout.

    Resolve named meshes, using *default_mesh* only for all-broadcast layouts.
    Bare dimensions broadcast, ``dim @ mesh.axis`` splits, and a final set maps
    mesh axes to partial reductions. Unmentioned mesh axes broadcast. Missing
    both explicit and default mesh information is an error.
    See [parser §1.5](docs/spec/parser.md#15-layout-sugar) and
    [shard §6](docs/spec/shard.md#6-shardattr).
    """
    axis_node, strides, value_set_node = _split_layout_outer(node)

    dim_nodes = _get_dim_nodes(axis_node)

    canonicalize = strides is None
    parsed: list[_LayoutItem] = []
    for dn in dim_nodes:
        parsed.extend(
            _parse_layout_item(dn, mesh_resolver, canonicalize=canonicalize, closure=closure)
        )

    value_states = (
        _parse_value_state(value_set_node, mesh_resolver) if value_set_node is not None else []
    )

    shape: list[int] = []

    resolved_mesh: Mesh | None = None
    for _d, mesh, _mi, _k, _r in parsed:
        if mesh is None:
            continue
        if resolved_mesh is None:
            resolved_mesh = mesh
        elif id(mesh) != id(resolved_mesh):
            raise VerifyError("all layout dims must reference the same mesh")
    for mesh, _mi, _r in value_states:
        if resolved_mesh is None:
            resolved_mesh = mesh
        elif id(mesh) != id(resolved_mesh):
            raise VerifyError("value-state mesh must match the layout mesh")

    if resolved_mesh is None:
        if default_mesh is not None:
            resolved_mesh = default_mesh
        else:
            raise VerifyError(
                "all-Broadcast ShardLayout sugar requires a mesh from "
                "context; use verbose ShardLayout(...) to disambiguate"
            )

    mesh_rank = len(resolved_mesh.layout.shape)
    attrs_list: list[ShardAttr] = [Broadcast() for _ in range(mesh_rank)]

    for dim, _mesh, m_axis, kind, _reduction in parsed:
        if dim is not None:
            shape.append(dim)
        layout_axis = len(shape) - 1
        if kind == "split":
            if m_axis is None or m_axis >= mesh_rank:
                raise VerifyError(f"layout dim {layout_axis}: invalid mesh axis {m_axis}")
            if not isinstance(attrs_list[m_axis], Broadcast):
                raise VerifyError(
                    f"mesh axis {m_axis} already bound; "
                    f"one layout dim per mesh axis ({spec_ref_render(_SHARD_ATTR)})"
                )
            attrs_list[m_axis] = Split(layout_axis)

    for _mesh, m_axis, reduction in value_states:
        if m_axis >= mesh_rank:
            raise VerifyError(f"value-state: invalid mesh axis {m_axis}")
        if not isinstance(attrs_list[m_axis], Broadcast):
            raise VerifyError(f"mesh axis {m_axis} already bound")
        attrs_list[m_axis] = Partial(reduction or "sum")

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
        raise VerifyError(f"expected tuple layout, got {ast.dump(node)}")

    if node.elts and isinstance(node.elts[0], ast.Tuple):
        axis_node = node.elts[0]
        strides = None
        value_set: ast.Set | None = None
        for elt in node.elts[1:]:
            if value_set is not None:
                raise VerifyError("layout sugar: the value-state set must be the last outer item")
            if isinstance(elt, ast.Set):
                value_set = elt
            elif isinstance(elt, ast.Tuple):
                if strides is not None:
                    raise VerifyError("layout sugar: at most one stride tuple")
                strides = _eval_ast(elt)
            else:
                raise VerifyError(
                    f"layout sugar outer item must be a stride tuple or value-state "
                    f"set, got {ast.dump(elt)}"
                )
        return axis_node, strides, value_set

    return node, None, None


def _parse_value_state(node: "ast.Set", mesh_resolver: MeshResolver) -> list[tuple[Mesh, int, str]]:
    """Parse value-state entries into mesh-axis reductions.

    Parse a ``{mesh.axis @ P("reduction"), ...}`` value-state set into a list
    of ``(mesh, mesh_axis_index, reduction)``. Element order carries no meaning.
    """
    if not isinstance(node, ast.Set):
        raise VerifyError(f"value-state must be a set literal, got {ast.dump(node)}")
    out: list[tuple[Mesh, int, str]] = []
    for elt in node.elts:
        if not (
            _is_matmul(elt)
            and isinstance(elt.right, ast.Call)
            and isinstance(elt.right.func, ast.Name)
            and elt.right.func.id == "P"
        ):
            raise VerifyError(
                f'value-state entry must be `mesh.axis @ P("reduction")`, got {ast.dump(elt)}'
            )
        if len(elt.right.args) != 1:
            raise VerifyError(
                "value-state P(...) requires exactly one reduction argument, "
                'e.g. `mesh.axis @ P("sum")`'
            )
        mesh_name, axis_name = _parse_axis_ref(elt.left)
        mesh = mesh_resolver(mesh_name)
        if mesh is None:
            raise VerifyError(f"undefined mesh {mesh_name!r}")
        axis = _resolve_mesh_axis(mesh, axis_name)
        reduction = _eval_ast(elt.right.args[0])
        out.append((mesh, axis, reduction))
    return out


def _get_dim_nodes(node: ast.AST) -> list[ast.AST]:
    """Extract dimension sub-nodes from a layout sugar tuple.

    Accepts both Tuple and BinOp (for standalone sugar like
    ``1536 @ (m.w, m.t)`` without a wrapping tuple).
    """
    if isinstance(node, ast.Tuple):
        if (
            len(node.elts) == 2
            and isinstance(node.elts[0], ast.Tuple)
            and isinstance(node.elts[1], ast.Tuple)
        ):
            return list(node.elts[0].elts)
        return list(node.elts)
    if _is_constant(node) or _is_matmul(node):
        return [node]
    raise VerifyError(f"expected tuple layout, got {ast.dump(node)}")


_LayoutItem = tuple[ShapeDim | None, Mesh | None, int | None, str, str | None]


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
    if _is_constant(node):
        return [(_extract_dim(node, closure=closure), None, None, "broadcast", None)]

    if _is_matmul(node):
        rhs = node.right
        dim = None if _is_placeholder(node.left) else _extract_dim(node.left, closure=closure)
        if dim is None:
            raise VerifyError(
                "layout placeholder `_` is not valid in the axis tuple; "
                'value states go in the `{mesh.axis @ P("reduction")}` set'
            )
        if isinstance(rhs, ast.Tuple):
            return _expand_multi_axis_sugar(dim, rhs.elts, mesh_resolver)

        if isinstance(rhs, ast.Attribute):
            mesh_name, axis_name = _parse_axis_ref(rhs)
            mesh = mesh_resolver(mesh_name)
            if mesh is None:
                raise VerifyError(f"undefined mesh {mesh_name!r}")
            axis = _resolve_mesh_axis(mesh, axis_name)
            if not canonicalize:
                return [(dim, mesh, axis, "split", None)]
            return _canonicalize_single_axis(dim, mesh, axis)

        if isinstance(rhs, ast.Name):
            mesh = mesh_resolver(rhs.id)
            if mesh is None:
                raise VerifyError(f"undefined mesh {rhs.id!r}")
            mesh_rank = len(mesh.layout.shape)
            if mesh_rank != 1:
                raise VerifyError(
                    f"``int @ {rhs.id}`` shorthand requires a single-axis mesh "
                    f"(found {mesh_rank} axes); write ``{rhs.id}.<axis>`` explicitly"
                )
            if not canonicalize:
                return [(dim, mesh, 0, "split", None)]
            return _canonicalize_single_axis(dim, mesh, 0)

    if closure is not None:
        try:
            dim = _eval_ast(node, closure)
        except ValueError:
            dim = None
        if _is_shape_dim(dim):
            return [(dim, None, None, "broadcast", None)]

    raise VerifyError(f"unexpected layout dim AST: {ast.dump(node)}")


def _canonicalize_single_axis(
    dim: ShapeDim,
    mesh: Mesh,
    axis: int,
) -> list[_LayoutItem]:
    """Factor an oversized single-axis split by its mesh extent.

    Expand ``N @ m.a`` so the split-bound dimension has local size one. ``N``
    must divide evenly by the mesh extent or parsing raises ``ValueError``.
    See [parser §1.5](docs/spec/parser.md#15-layout-sugar) and
    [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    extent = mesh.layout.shape[axis]
    if not isinstance(dim, int) or not isinstance(extent, int):
        if dim == extent:
            return [(dim, mesh, axis, "split", None)]
        raise LayoutSugarError(
            f"split layout dim {dim!r} and mesh extent {extent!r} do not have "
            "a decidable divisibility relation; bind symbolic dimensions before "
            "authoring this split"
        )
    if dim % extent != 0:
        raise VerifyError(
            f"dim {dim} not divisible by mesh extent {extent} on axis "
            f"{axis}; cannot canonicalize ``{dim} @ m.<axis>``"
        )
    if dim == extent:
        return [(dim, mesh, axis, "split", None)]
    residual = dim // extent
    return [
        (extent, mesh, axis, "split", None),
        (residual, None, None, "broadcast", None),
    ]


def _expand_multi_axis_sugar(
    dim: ShapeDim,
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
        extent = mesh.layout.shape[axis]
        if not isinstance(remaining, int) or not isinstance(extent, int):
            if remaining == extent and i == len(axis_nodes) - 1:
                items.append((extent, mesh, axis, "split", None))
                remaining = 1
                continue
            raise LayoutSugarError(
                f"split layout dim {dim!r} and mesh extent {extent!r} at axis "
                f"position {i} do not have a decidable divisibility relation; "
                "bind symbolic dimensions before authoring this split"
            )
        if remaining % extent != 0:
            raise VerifyError(
                f"dim {dim} not divisible by mesh extent {extent} "
                f"at axis position {i}; remaining={remaining}"
            )
        per_axis = extent
        remaining //= extent
        items.append((per_axis, mesh, axis, "split", None))

    if remaining > 0:
        items.append((remaining, None, None, "broadcast", None))

    return items


def _resolve_axis_node(
    node: ast.AST,
    mesh_resolver: MeshResolver,
) -> tuple[Mesh, int]:
    """Resolve a mesh-axis reference node to a mesh and layout-axis index.

    Accepts ``mesh.axis`` attribute references and single-axis
    ``mesh`` name shorthand.
    """
    if isinstance(node, ast.Attribute):
        mesh_name, axis_name = _parse_axis_ref(node)
        mesh = mesh_resolver(mesh_name)
        if mesh is None:
            raise VerifyError(f"undefined mesh {mesh_name!r}")
        axis = _resolve_mesh_axis(mesh, axis_name)
        return (mesh, axis)
    if isinstance(node, ast.Name):
        mesh = mesh_resolver(node.id)
        if mesh is None:
            raise VerifyError(f"undefined mesh {node.id!r}")
        mesh_rank = len(mesh.layout.shape)
        if mesh_rank != 1:
            raise VerifyError(
                f"``int @ (..., {node.id}, ...)`` shorthand requires a "
                f"single-axis mesh (found {mesh_rank} axes); "
                f"write ``{node.id}.<axis>`` explicitly"
            )
        return (mesh, 0)
    raise VerifyError(f"expected mesh.axis, got {ast.dump(node)}")


def _parse_axis_ref(node: ast.AST) -> tuple[str, str]:
    """Parse a mesh-qualified axis reference.

    ``gpu.cluster`` → ``("gpu", "cluster")``
    ``gpu.x``       → ``("gpu", "x")``
    """
    if isinstance(node, ast.Attribute):
        mesh_name = _name_of(node.value)
        axis_name = node.attr
        return (mesh_name, axis_name)
    raise VerifyError(f"expected mesh.axis (e.g. gpu.cluster), got {ast.dump(node)}")


def _parse_tensor_type_sugar(
    node: ast.AST,
    closure: dict[str, Any],
    *,
    mesh_resolver: MeshResolver | None = None,
    default_mesh: Mesh | None = None,
) -> TensorType | None:
    """Parse a ``Tensor[...]`` or ``ConstTensor[...]`` type literal."""
    if not isinstance(node, ast.Subscript):
        return None
    head = node.value.id if isinstance(node.value, ast.Name) else None
    if isinstance(node.value, ast.Attribute):
        head = node.value.attr
    if head not in ("Tensor", "ConstTensor"):
        return None
    if not isinstance(node.slice, ast.Tuple):
        raise VerifyError("Tensor[...] requires shape and dtype slots")
    elts = node.slice.elts
    if len(elts) not in (2, 3, 4):
        raise VerifyError("Tensor[...] requires shape, dtype, and optional layout/storage")

    shape, _ = _parse_layout_literal(elts[0], closure=closure)
    dtype_val = _resolve_dtype_ast(elts[1], closure)
    if dtype_val is None:
        raise VerifyError(f"unknown tensor dtype {ast.unparse(elts[1])!r}")

    meshes = {key: value for key, value in closure.items() if isinstance(value, Mesh)}
    resolver = mesh_resolver or meshes.get
    if default_mesh is None and len(meshes) == 1:
        default_mesh = next(iter(meshes.values()))

    layout = None
    storage = StorageKind.GMEM
    embedded_layout = _has_sugar(elts[0])
    if embedded_layout:
        layout = _parse_shard_layout_sugar(
            elts[0], resolver, default_mesh=default_mesh, closure=closure
        )

    if len(elts) >= 3:
        third = elts[2]
        if isinstance(third, ast.Constant) and third.value is None:
            pass
        else:
            try:
                third_value = _eval_ast(third, closure)
            except ValueError:
                third_value = None
            try:
                storage_value = resolve_storage(third_value)
            except (TypeError, ValueError):
                storage_value = None
            if storage_value is not None:
                storage = storage_value
            else:
                if embedded_layout:
                    raise VerifyError(
                        "Tensor[...] cannot specify placement in both shape and layout slots"
                    )
                if isinstance(third_value, (Layout, ShardLayout)):
                    layout = third_value
                else:
                    layout = _parse_shard_layout_sugar(
                        third, resolver, default_mesh=default_mesh, closure=closure
                    )
    if len(elts) == 4:
        if embedded_layout:
            raise VerifyError(
                "Tensor[...] with placement in its shape takes storage as the third slot"
            )
        storage = resolve_storage(_eval_ast(elts[3], closure))

    if (
        isinstance(layout, ShardLayout)
        and isinstance(layout.layout, Layout)
        and layout.layout.strides is None
    ):
        layout = ShardLayout(
            layout=Layout(
                shape=layout.layout.shape,
                strides=_auto_strides(layout.layout.shape),
            ),
            attrs=layout.attrs,
            mesh=layout.mesh,
        )
    return TensorType(shape=shape, dtype=dtype_val, layout=layout, storage=storage)


def parse_sugar(
    node: ast.AST,
    expected: type,
    *,
    closure: dict[str, Any] | None = None,
    mesh_resolver: MeshResolver | None = None,
    default_mesh: Mesh | None = None,
) -> Layout | ShardLayout | TensorType | None:
    """Parse one type-directed layout or tensor-type sugar form."""
    closure = closure or {}
    if expected is Layout:
        return _parse_layout_sugar(node, closure=closure)
    if expected is ShardLayout:
        if mesh_resolver is None:
            meshes = {key: value for key, value in closure.items() if isinstance(value, Mesh)}
            mesh_resolver = meshes.get
            if default_mesh is None and len(meshes) == 1:
                default_mesh = next(iter(meshes.values()))
        return _parse_shard_layout_sugar(
            node,
            mesh_resolver,
            default_mesh=default_mesh,
            closure=closure,
        )
    if expected is TensorType:
        return _parse_tensor_type_sugar(
            node,
            closure,
            mesh_resolver=mesh_resolver,
            default_mesh=default_mesh,
        )
    raise TypeError(f"unsupported sugar result type {expected!r}")


__all__ = ["LayoutSugarError", "parse_sugar"]
