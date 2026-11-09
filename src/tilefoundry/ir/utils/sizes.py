"""Element counts and byte sizes of types."""

from __future__ import annotations

import math

from tilefoundry.ir import expr, types

from .bounds import upper_bound
from .leaves import tensor_types

__all__ = ["bytes_by_storage", "numel", "shape_runtime_total", "tensor_bytes", "topology_extent"]


def numel(type: types.Type) -> int:
    """Element count of ``type``, summed over a tuple's leaves.

    A symbolic or negative extent is rejected rather than skipped: a size that
    silently drops a dimension reads as a smaller tensor, not as an unknown
    one. A concrete zero extent is a zero-sized tensor.
    """
    return sum(leaf_numel(leaf) for leaf in tensor_types(type))


def leaf_numel(type: types.TensorType) -> int:
    """Element count of one tensor leaf; a symbolic or negative extent is refused."""
    values = []
    for dim in type.shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            names = expr.dim_variables(dim)
            hint = f"; bind it with --dim {next(iter(names))}=EXTENT" if names else ""
            raise ValueError(f"numel: tensor extent {dim!r} is not concrete{hint}")
        if dim < 0:
            raise ValueError(f"numel: tensor extent {dim} is negative")
        values.append(dim)
    return math.prod(values)


def tensor_bytes(type: types.Type) -> int:
    """Byte size of ``type``, summed over a tuple's leaves.

    This is the logical size the type states, so it is the same number for
    every backend. A sub-byte dtype rounds up to whole bytes per leaf, because
    a leaf is addressed on its own.
    """
    return sum(
        math.ceil(leaf_numel(leaf) * leaf.dtype.bit_width / 8) for leaf in tensor_types(type)
    )


def bytes_by_storage(type: types.Type, *, umat_level: str | None = None) -> dict[str, int]:
    """Logical bytes occupied by *type*, grouped by storage level."""
    result: dict[str, int] = {}
    for tensor in tensor_types(type):
        if tensor.storage is expr.StorageKind.UMAT:
            if umat_level is None:
                continue
            memory_level = umat_level
        else:
            memory_level = str(tensor.storage)
        result[memory_level] = result.get(memory_level, 0) + tensor_bytes(tensor)
    return result


def topology_extent(type: types.Type, name: str) -> int | None:
    """The one logical extent *type* states for topology *name*, if any."""
    extents: set[int] = set()
    for tensor in tensor_types(type):
        layout = types.shard_layout_of(tensor.layout)
        if layout is None:
            continue
        names = tuple(topology.name for topology in layout.mesh.topologies)
        if len(names) != 1 or names[0] != name:
            continue
        count = types.size(types.flatten(layout.mesh.layout))
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"topology_extent: {name!r} needs a positive static layout size")
        extents.add(count)
    if len(extents) > 1:
        raise ValueError(f"one value references conflicting {name!r} extents {sorted(extents)}")
    return next(iter(extents), None)


def shape_runtime_total(shape, dim_var_expr: dict[str, str]) -> object:
    """Return the runtime element count of *shape*.

    All-static shape → an ``int``. Any ``DimVar`` axis pulls its
    runtime extent from ``dim_var_expr[name]``; the result is a C++
    expression string ``"(a * b * ...)"`` that the codegen splices
    verbatim into the generated source. Static dims fold into a single
    leading constant factor when present, otherwise the constant is
    elided.
    """
    if not shape:
        return 1
    static_prod = 1
    dyn_terms: list[str] = []
    for s in shape:
        if isinstance(s, expr.DimVar):
            node = dim_var_expr.get(s.name)
            if node is None:
                static_prod *= upper_bound(s)
            else:
                dyn_terms.append(node)
        else:
            static_prod *= upper_bound(s)
    if not dyn_terms:
        return static_prod
    if static_prod == 1:
        if len(dyn_terms) == 1:
            return dyn_terms[0]
        return "(" + " * ".join(dyn_terms) + ")"
    return "(" + " * ".join([str(static_prod), *dyn_terms]) + ")"
