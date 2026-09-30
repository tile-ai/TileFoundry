from __future__ import annotations

import math
from itertools import product
from typing import Optional

from tilefoundry.ir.types.int_tuple import flatten as flatten_tuple
from tilefoundry.ir.types.int_tuple import repeat_like
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.storage import StorageKind

from .dtype import DType
from .layout import ComposedLayout, Layout, apply
from .layout_algebra import size
from .mesh import Mesh, Topology, separate
from .shard_layout import (
    ShardLayout,
    Split,
    canonical_shard_layout,
    shard_layout_of,
    split_target_axes,
)
from .stride import compact_row_major
from .tensor_type import TensorType, TupleType, Type


def _tile_counts(whole: tuple, inner: tuple) -> tuple[int, ...]:
    if len(whole) != len(inner):
        raise ValueError(f"whole shape {whole} and inner shape {inner} have different ranks")
    counts = []
    for whole_extent, inner_extent in zip(whole, inner, strict=True):
        if (
            not isinstance(whole_extent, int)
            or isinstance(whole_extent, bool)
            or not isinstance(inner_extent, int)
            or isinstance(inner_extent, bool)
            or inner_extent < 1
            or whole_extent % inner_extent
        ):
            raise ValueError(
                f"whole extent {whole_extent} is not divisible by inner extent {inner_extent}"
            )
        counts.append(whole_extent // inner_extent)
    return tuple(counts)


def _inner_layout(layout: Layout, whole: tuple, inner: tuple) -> tuple[Layout, dict[int, int]]:
    """Drop each logical axis's leading tile modes, preserving its mode nesting."""
    counts = _tile_counts(whole, inner)
    shapes = [tuple(flatten_tuple(mode)) for mode in layout.shape]
    strides = (
        None if layout.strides is None else [tuple(flatten_tuple(mode)) for mode in layout.strides]
    )
    grouped = any(isinstance(mode, tuple) for mode in layout.shape)
    if grouped and len(shapes) != len(counts):
        raise ValueError(f"grouped layout has {len(shapes)} modes for a rank-{len(counts)} tensor")

    prefixes: list[int] = []
    if grouped:
        for axis, (mode, count) in enumerate(zip(shapes, counts, strict=True)):
            taken = 1
            prefix = 0
            while taken != count and prefix < len(mode):
                extent = mode[prefix]
                if not isinstance(extent, int) or isinstance(extent, bool):
                    break
                taken *= extent
                prefix += 1
            if taken != count:
                raise ValueError(
                    f"layout axis {axis} has no leading modes whose product is tile count {count}"
                )
            prefixes.append(prefix)
    else:
        tiled = tuple(count for count in counts if count != 1)
        if tuple(mode[0] for mode in shapes[: len(tiled)]) != tiled:
            raise ValueError(
                f"flat layout does not begin with tile modes {tiled}: {tuple(layout.shape)}"
            )
        prefixes = [1] * len(tiled) + [0] * (len(shapes) - len(tiled))

    shape_modes = []
    stride_modes = []
    old_to_new: dict[int, int] = {}
    for old_index, shape_mode in enumerate(shapes):
        prefix = prefixes[old_index]
        kept_shape = list(shape_mode[prefix:])
        stride_mode = None if strides is None else strides[old_index]
        kept_stride = [] if stride_mode is None else list(stride_mode[prefix:])
        if not kept_shape:
            continue
        old_to_new[old_index] = len(shape_modes)
        nested = isinstance(layout.shape[old_index], tuple) and len(kept_shape) > 1
        shape_modes.append(tuple(kept_shape) if nested else kept_shape[0])
        if strides is not None:
            stride_modes.append(tuple(kept_stride) if nested else kept_stride[0])
    return (
        Layout(tuple(shape_modes), None if strides is None else tuple(stride_modes)),
        old_to_new,
    )


def participant_mesh(source: Mesh, required: Mesh) -> tuple[Mesh, int]:
    """Select the trailing modes that state the required physical frame."""
    required_names = tuple(getattr(topology, "name", topology) for topology in required.topologies)
    source_names = tuple(getattr(topology, "name", topology) for topology in source.topologies)
    if source_names != required_names:
        source = next(
            (
                level
                for level in separate(source)
                if tuple(getattr(topology, "name", topology) for topology in level.topologies)
                == required_names
            ),
            source,
        )
    source_layout = flatten(source.layout)
    required_layout = flatten(required.layout)
    if not isinstance(source_layout, Layout) or not isinstance(required_layout, Layout):
        raise ValueError("tile participant meshes require concrete layouts")
    source_shape = tuple(source_layout.shape)
    if source_layout.strides is None or required_layout.strides is None:
        raise ValueError("tile participant meshes require strided layouts")
    source_strides = tuple(source_layout.strides)
    stated = {apply(required_layout, index) for index in range(size(required_layout))}
    dropped = next(
        (
            begin
            for begin in range(len(source_shape) - 1, -1, -1)
            if (
                (suffix := Layout(source_shape[begin:], source_strides[begin:]))
                and {apply(suffix, index) for index in range(size(suffix))} == stated
            )
        ),
        None,
    )
    if dropped is None:
        raise ValueError(
            f"mesh layout {source_shape} does not end in participant frame "
            f"{tuple(required_layout.shape)}"
        )
    from .mesh import starts  # noqa: PLC0415 - Mesh imports this module through types

    return (
        Mesh(
            source.topologies,
            ComposedLayout(None, starts(source)[0], required.layout),
            required.names,
        ),
        dropped,
    )


def nonunit_mesh(source: Mesh) -> Mesh:
    """Return the source mesh with lexical unit modes omitted."""
    layout = flatten(source.layout)
    if not isinstance(layout, Layout) or layout.strides is None:
        raise ValueError("instruction issue needs a static strided mesh")
    modes = tuple(
        (extent, stride)
        for extent, stride in zip(layout.shape, layout.strides, strict=True)
        if extent != 1
    ) or ((1, 1),)
    from .mesh import starts  # noqa: PLC0415 - Mesh imports this module through types

    return Mesh(
        source.topologies,
        ComposedLayout(None, starts(source)[0], Layout(*zip(*modes, strict=True))),
        tuple(f"d{index}" for index in range(len(modes))),
    )


def issue_frames(source: Mesh, required: Mesh, repeat: tuple, tile: tuple):
    """Yield each declared issue frame and its grouped work-axis origins."""
    physical = flatten(source.layout)
    if not isinstance(physical, Layout) or physical.strides is None:
        raise ValueError("group axes need a static strided physical mesh")
    participant, dropped = participant_mesh(source, required)
    outer = tuple(
        (index, extent)
        for index, extent in enumerate(physical.shape[:dropped])
        if extent != 1
    )
    grouped = tuple(axis for axis, count in enumerate(repeat) if count > 1)[: len(outer)]
    expected = tuple(repeat[axis] for axis in grouped)
    if tuple(extent for _, extent in outer) != expected:
        raise ValueError(
            f"physical mesh {tuple(physical.shape)} is not outer modes followed by "
            "participant frame"
        )
    from .mesh import starts  # noqa: PLC0415 - Mesh imports this module through types

    local = flatten(participant.layout)
    for coordinates in product(*(range(extent) for _, extent in outer)):
        offset = starts(source)[0] + sum(
            coordinate * physical.strides[index]
            for (index, _extent), coordinate in zip(outer, coordinates, strict=True)
        )
        frame = Mesh(
            source.topologies,
            ComposedLayout(None, offset, local),
            required.names,
        )
        yield frame, {
            axis: coordinate * tile[axis]
            for axis, coordinate in zip(grouped, coordinates, strict=True)
        }


def tile_view_layout(
    type_: TensorType,
    inner_shape: tuple,
    *,
    participant: Mesh | None = None,
    enclosing: Mesh | None = None,
    shard_attrs: tuple | None = None,
) -> object:
    """Return the layout presented by one tile after leading modes are removed.

    Tensor and mesh layouts use the same convention: leading modes are tile or
    group coordinates and trailing modes are the instruction fragment/frame.
    """
    whole = tuple(type_.shape)
    _tile_counts(whole, inner_shape)
    layout = type_.layout
    if layout is None:
        inner_layout = Layout(inner_shape, tuple(compact_row_major(inner_shape)))
    else:
        composed = isinstance(layout, ComposedLayout)
        inner = layout.inner if composed else None
        outer = layout.outer if composed else layout
        if isinstance(outer, ShardLayout):
            held, positions = _inner_layout(outer.layout, whole, inner_shape)
            mesh, dropped = (
                (outer.mesh, 0)
                if participant is None
                else participant_mesh(outer.mesh, participant)
            )
            attrs = []
            for attr in outer.attrs[dropped:]:
                if isinstance(attr, Split):
                    if attr.axis not in positions:
                        raise ValueError(f"tile prefix removes sharded layout axis {attr.axis}")
                    attr = Split(positions[attr.axis])
                attrs.append(attr)
            inner_layout = ShardLayout(held, tuple(attrs), mesh)
        elif isinstance(outer, Layout):
            inner_layout, _positions = _inner_layout(outer, whole, inner_shape)
        else:
            raise ValueError(f"cannot take tile inner modes from {type(outer).__name__}")
        if composed:
            inner_layout = ComposedLayout(inner, 0, inner_layout)
    if shard_attrs is not None and not isinstance(inner_layout, ShardLayout):
        if participant is None or enclosing is None:
            raise ValueError("a declared shard fragment needs an enclosing participant mesh")
        frame, _dropped = participant_mesh(enclosing, participant)
        inner_layout = ShardLayout(inner_layout, shard_attrs, frame)
    return inner_layout


def tile_inner_type(
    type_: TensorType,
    inner_shape: tuple,
    *,
    participant: Mesh | None = None,
    enclosing: Mesh | None = None,
    shard_attrs: tuple | None = None,
) -> TensorType:
    """Take the inner fragment after leading CuTe tile modes are removed."""
    layout = tile_view_layout(
        type_,
        inner_shape,
        participant=participant,
        enclosing=enclosing,
        shard_attrs=shard_attrs,
    )
    return TensorType(tuple(inner_shape), type_.dtype, layout, type_.storage)


def types_compatible(declared: Type, actual: Type) -> bool:
    """Return whether *actual* may bind a position declared as *declared*.

    Every non-``None`` declared field constrains the corresponding actual
    field. ``UMAT`` is the storage field's undecided value. Layout descriptors
    apply the same rule recursively.
    """
    def field_compatible(declared_field, actual_field) -> bool:
        return declared_field is None or declared_field == actual_field

    def layout_compatible(declared_layout, actual_layout) -> bool:
        if declared_layout is None:
            return True
        if isinstance(declared_layout, Layout):
            return (
                isinstance(actual_layout, Layout)
                and field_compatible(declared_layout.shape, actual_layout.shape)
                and field_compatible(declared_layout.strides, actual_layout.strides)
            )
        if isinstance(declared_layout, ShardLayout):
            return (
                isinstance(actual_layout, ShardLayout)
                and field_compatible(declared_layout.mesh, actual_layout.mesh)
                and field_compatible(declared_layout.attrs, actual_layout.attrs)
                and layout_compatible(declared_layout.layout, actual_layout.layout)
            )
        return actual_layout == declared_layout

    if isinstance(declared, TensorType):
        return (
            isinstance(actual, TensorType)
            and declared.shape == actual.shape
            and declared.dtype == actual.dtype
            and (
                declared.storage is StorageKind.UMAT
                or declared.storage == actual.storage
            )
            and layout_compatible(declared.layout, actual.layout)
        )
    if isinstance(declared, TupleType):
        return (
            isinstance(actual, TupleType)
            and len(declared.fields) == len(actual.fields)
            and all(
                types_compatible(declared_field, actual_field)
                for declared_field, actual_field in zip(declared.fields, actual.fields)
            )
        )
    return actual == declared


def numel(type: Type) -> int:
    """Element count of ``type``, summed over a tuple's leaves.

    A symbolic or negative extent is rejected rather than skipped: a size that
    silently drops a dimension reads as a smaller tensor, not as an unknown
    one. A concrete zero extent is a zero-sized tensor.
    """
    return sum(_leaf_numel(leaf) for leaf in tensor_types(type))


def _leaf_numel(type: TensorType) -> int:
    values = []
    for dim in type.shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            from .substitute import dim_vars_by_name  # noqa: PLC0415

            names = dim_vars_by_name(dim)
            hint = f"; bind it with --dim {next(iter(names))}=EXTENT" if names else ""
            raise ValueError(f"numel: tensor extent {dim!r} is not concrete{hint}")
        if dim < 0:
            raise ValueError(f"numel: tensor extent {dim} is negative")
        values.append(dim)
    return math.prod(values)


def tensor_bytes(type: Type) -> int:
    """Byte size of ``type``, summed over a tuple's leaves.

    This is the logical size the type states, so it is the same number for
    every backend. A sub-byte dtype rounds up to whole bytes per leaf, because
    a leaf is addressed on its own.
    """
    return sum(
        math.ceil(_leaf_numel(leaf) * leaf.dtype.bit_width / 8)
        for leaf in tensor_types(type)
    )


def tensor_types(type: Type) -> tuple[TensorType, ...]:
    """The tensor leaves of *type*, flattened out of tuple nesting."""
    if isinstance(type, TensorType):
        return (type,)
    if isinstance(type, TupleType):
        return tuple(leaf for field in type.fields for leaf in tensor_types(field))
    return ()


def bytes_by_storage(
    type: Type, *, umat_level: str | None = None
) -> dict[str, int]:
    """Logical bytes occupied by *type*, grouped by storage level."""
    result: dict[str, int] = {}
    for tensor in tensor_types(type):
        if tensor.storage is StorageKind.UMAT:
            if umat_level is None:
                continue
            memory_level = umat_level
        else:
            memory_level = str(tensor.storage)
        result[memory_level] = result.get(memory_level, 0) + tensor_bytes(tensor)
    return result


def topology_extent(type: Type, name: str) -> int | None:
    """The one logical extent *type* states for topology *name*, if any."""
    extents: set[int] = set()
    for tensor in tensor_types(type):
        layout = shard_layout_of(tensor.layout)
        if layout is None:
            continue
        names = tuple(topology.name for topology in layout.mesh.topologies)
        if len(names) != 1 or names[0] != name:
            continue
        count = size(flatten(layout.mesh.layout))
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(
                f"topology_extent: {name!r} needs a positive static layout size"
            )
        extents.add(count)
    if len(extents) > 1:
        raise ValueError(
            f"one value references conflicting {name!r} extents {sorted(extents)}"
        )
    return next(iter(extents), None)


def make_tensor_type(
    shape: tuple,
    dtype: DType = DType.f32,
    storage: "str | StorageKind" = "gmem",
    layout: object = None,
) -> "TensorType":
    """Convenience constructor for a plain (unsharded) ``TensorType``."""
    return TensorType(shape=tuple(shape), dtype=dtype, layout=layout, storage=storage)


def make_shard_tensor_type(
    shape: tuple,
    dtype: DType = DType.f32,
    storage: "str | StorageKind" = "gmem",
    mesh: Optional[Mesh] = None,
    attrs: tuple = (),
) -> "TensorType":
    """Build a canonical sharded ``TensorType`` from its logical description.

    ``attrs`` contains one entry per mesh axis. With no mesh or attributes the
    result is unsharded; otherwise :func:`canonical_shard_layout` supplies the
    shared canonical representation.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    shape = tuple(shape)
    if mesh is None or not attrs:
        return TensorType(shape=shape, dtype=dtype, layout=None, storage=storage)
    layout = canonical_shard_layout(shape, mesh, attrs)
    return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


def _all_split_local_type(type: Type, *, refuse_indivisible: bool) -> Type | None:
    """Project every Split, optionally returning None for an indivisible axis."""
    if not isinstance(type, TensorType):
        return type
    layout = shard_layout_of(type.layout)
    if layout is None:
        return type
    local = list(type.shape)
    for mesh_axis, tensor_axis in enumerate(split_target_axes(layout, type.shape)):
        if tensor_axis is None:
            continue
        extent = flatten(layout.mesh.layout).shape[mesh_axis]
        if extent is None:
            local[tensor_axis] = 1
            continue
        size = local[tensor_axis]
        if not isinstance(size, int) or isinstance(size, bool):
            raise ValueError(
                f"tensor axis {tensor_axis} is Split-sharded but its extent "
                f"{size!r} is not a static int"
            )
        if size % extent != 0:
            if not refuse_indivisible:
                return None
            raise ValueError(
                f"tensor axis {tensor_axis} (extent {size}) is not evenly "
                f"divisible by its mesh extent {extent}"
            )
        local[tensor_axis] = size // extent
    return TensorType(shape=tuple(local), dtype=type.dtype, layout=None, storage=type.storage)


def try_local_type_of(type: Type) -> Type | None:
    """Project all splits, returning None when a static split is indivisible."""
    return _all_split_local_type(type, refuse_indivisible=False)


def local_type_of(
    type: Type, *, topology_level: str | None = None, topologies: tuple[Topology, ...] = ()
) -> Type:
    """Project every tensor leaf to what one unit holds.

    With ``topology_level``: a ``Split`` at that level or coarser divides, while
    finer splits, ``Broadcast``, and ``Partial`` do not; logical axes may factor
    into layout positions, and ``topologies`` supplies the ordered hierarchy.
    Without it: every ``Split`` divides, the layout is dropped, and the logical
    rank is preserved. This form is for relations over logical axes, where
    factoring an axis into layout positions would lose the modeled flow.
    """
    if topology_level is None:
        projected = _all_split_local_type(type, refuse_indivisible=True)
        assert projected is not None
        return projected

    levels = {topology.name: index for index, topology in enumerate(topologies)}
    if topology_level not in levels:
        available = ", ".join(levels) or "none"
        raise ValueError(
            f"local_type_of: topology level {topology_level!r} is not declared; "
            f"available levels are {available}"
        )
    if len(levels) != len(topologies):
        raise ValueError("local_type_of: topology level names must be unique")
    if isinstance(type, TupleType):
        return TupleType(
            fields=tuple(
                local_type_of(field, topology_level=topology_level, topologies=topologies)
                for field in type.fields
            )
        )
    if not isinstance(type, TensorType):
        return type
    layout = type.layout
    if layout is None:
        return type
    shard = shard_layout_of(layout)
    if shard is not None:
        return TensorType(
            shape=_local_layout_shape(
                shard, selected_topology_level=levels[topology_level], topologies=topologies
            ),
            dtype=type.dtype,
            layout=layout,
            storage=type.storage,
        )
    if isinstance(layout, (Layout, ComposedLayout)):
        return type
    raise ValueError(
        f"local_type_of: {type!r} has unresolved layout {layout!r}; local "
        "projection requires None or a resolved ShardLayout"
    )


def _require_concrete(shape: tuple | list) -> None:
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
        raise ValueError(
            "local_type_of: local tensor extent is not a concrete non-negative integer"
        )


def _nested_layout_shape(
    layout: object, *, selected_topology_level: int, topologies: tuple[Topology, ...]
) -> tuple:
    if isinstance(layout, ShardLayout):
        return _local_layout_shape(
            layout, selected_topology_level=selected_topology_level, topologies=topologies
        )
    if isinstance(layout, (Layout, ComposedLayout)):
        return tuple(layout.shape)
    raise ValueError(
        f"local_type_of: unresolved layout {layout!r}; local projection requires a resolved Layout"
    )


def _local_layout_shape(
    layout: ShardLayout, *, selected_topology_level: int, topologies: tuple[Topology, ...]
) -> tuple[int, ...]:
    shape = list(
        _nested_layout_shape(
            layout.layout, selected_topology_level=selected_topology_level, topologies=topologies
        )
    )
    declared = {topology.name: index for index, topology in enumerate(topologies)}
    for topology in layout.mesh.topologies:
        if topology.name not in declared:
            raise ValueError(
                f"local_type_of: shard uses undeclared topology level {topology.name!r}"
            )
    mesh_layout = layout.mesh.layout
    stated = mesh_layout.outer if isinstance(mesh_layout, ComposedLayout) else mesh_layout
    axis_topology_level = flatten(
        tuple(
            repeat_like(mode, declared[topology.name])
            for mode, topology in zip(stated.shape, layout.mesh.topologies, strict=True)
        )
    )
    mesh_shape = flatten(layout.mesh.layout).shape
    for mesh_axis, attr in enumerate(layout.attrs):
        if not isinstance(attr, Split):
            continue
        here = (
            axis_topology_level[mesh_axis]
            if mesh_axis < len(axis_topology_level)
            else selected_topology_level
        )
        if here > selected_topology_level:
            continue
        if mesh_axis >= len(mesh_shape):
            raise ValueError("local_type_of: shard attribute exceeds mesh layout rank")
        extent = mesh_shape[mesh_axis]
        axis = attr.axis
        if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < len(shape):
            raise ValueError("local_type_of: Split axis is not a concrete layout axis")
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ValueError("local_type_of: mesh extent is not a concrete positive integer")
        if extent == shape[axis]:
            shape[axis] = 1
        elif not isinstance(shape[axis], int) or isinstance(shape[axis], bool):
            raise ValueError(
                f"local_type_of: axis {axis} has dynamic extent {shape[axis]!r} "
                f"and mesh extent {extent} states a fixed position count; bind "
                "the axis before local projection"
            )
        elif shape[axis] % extent:
            raise ValueError(
                f"local_type_of: extent {shape[axis]} is not divisible by mesh "
                f"extent {extent}; write the two loops out as (ceildiv(N, T), T) "
                "and bind the tile count"
            )
        else:
            shape[axis] //= extent
    _require_concrete(shape)
    return tuple(shape)


def static_dim_value(dim):
    """Return the compile-time ``int`` value of a *static* shape dim, else ``None``.

    A static dim is a plain ``int`` or an integer-valued ``Constant`` (the latter
    only appears transiently before ``TensorType`` canonicalizes it to ``int``).
    ``DimVar`` / dynamic dim ``Call`` exprs are not static → ``None``. The
    detection is exact (real ``Constant`` with an ``int`` value), never "anything
    with a ``.value``".
    """
    from .dim import Constant  # noqa: PLC0415 - cycle guard

    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    if isinstance(dim, Constant) and isinstance(dim.value, int) and not isinstance(dim.value, bool):
        return int(dim.value)
    return None


def i64_const(value: int) -> "Constant":
    """The canonical i64 shape-scalar ``Constant`` (meta-scalar typed)."""
    from .dim import Constant  # noqa: PLC0415 - cycle guard
    from .tensor_type import TensorType  # noqa: PLC0415 - cycle guard

    return Constant(type=TensorType.umat_scalar(), value=int(value))


def upper_bound(dim) -> int:
    """Return a concrete int upper-bound element count for ``dim``."""
    from .dim import DimVar  # noqa: PLC0415 - cycle guard

    if isinstance(dim, DimVar):
        return int(dim.hi) - 1
    static = static_dim_value(dim)
    if static is not None:
        return static
    return int(dim)


def shape_numel_upper_bound(shape) -> int:
    """Product of per-dim upper bounds.

    Product of per-dim upper bounds: the static element count a buffer or
    layout must hold across every runtime shape in the dispatch envelope.
    """
    n = 1
    for s in shape:
        n *= upper_bound(s)
    return n


def shape_upper_bound(shape) -> tuple[int, ...]:
    """Map ``upper_bound`` over every entry of *shape*."""
    return tuple(upper_bound(s) for s in shape)


def shape_has_dim_var(shape) -> bool:
    """True iff *shape* contains at least one ``DimVar`` entry."""
    from .dim import DimVar  # noqa: PLC0415 - cycle guard

    return any(isinstance(s, DimVar) for s in shape)


def shape_runtime_total(shape, dim_var_expr: dict[str, str]) -> object:
    """Return the runtime element count of *shape*.

    All-static shape → an ``int``. Any ``DimVar`` axis pulls its
    runtime extent from ``dim_var_expr[name]``; the result is a C++
    expression string ``"(a * b * ...)"`` that the codegen splices
    verbatim into the generated source. Static dims fold into a single
    leading constant factor when present, otherwise the constant is
    elided.
    """
    from .dim import DimVar  # noqa: PLC0415 - cycle guard

    if not shape:
        return 1
    static_prod = 1
    dyn_terms: list[str] = []
    for s in shape:
        if isinstance(s, DimVar):
            expr = dim_var_expr.get(s.name)
            if expr is None:
                static_prod *= upper_bound(s)
            else:
                dyn_terms.append(expr)
        else:
            static_prod *= upper_bound(s)
    if not dyn_terms:
        return static_prod
    if static_prod == 1:
        if len(dyn_terms) == 1:
            return dyn_terms[0]
        return "(" + " * ".join(dyn_terms) + ")"
    return "(" + " * ".join([str(static_prod), *dyn_terms]) + ")"
