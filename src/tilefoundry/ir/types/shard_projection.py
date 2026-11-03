"""Factor domain modes and fix selected mesh coordinates.

See [semantic-analysis §3.1](docs/spec/semantic-analysis.md#31-logical-shape-to-layout-domain).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .int_tuple import compatible_profile, flatten, product, unflatten
from .layout import ComposedLayout, Layout
from .shard_layout import Split
from .stride import compact_row_major, crd2idx


def factor_axis(axis_size, mesh_shape, splitting_mesh_axes, logical_axis):
    """Factor a domain axis by its ordered mesh cuts, without choosing strides."""
    from .dim import exact_quotient  # noqa: PLC0415 -- staged type imports

    axis_static = isinstance(axis_size, int) and not isinstance(axis_size, bool)
    if len(splitting_mesh_axes) == 1 and not axis_static:
        return (axis_size,)
    factors = []
    if not axis_static:
        residual = axis_size
        for mesh_axis in splitting_mesh_axes:
            extent = mesh_shape[mesh_axis]
            quotient = exact_quotient(residual, extent)
            if quotient is None:
                raise ValueError(
                    f"default_shard_layout: logical axis {logical_axis} size "
                    f"{axis_size!r} is dynamic; cannot factorize across multiple "
                    "mesh axes (undecidable division by axis "
                    f"{mesh_axis} extent {extent!r})"
                )
            factors.append(extent)
            residual = quotient
    else:
        extent_product = 1
        for mesh_axis in splitting_mesh_axes:
            extent = mesh_shape[mesh_axis]
            if not isinstance(extent, int) or isinstance(extent, bool):
                raise ValueError(
                    f"default_shard_layout: mesh axis {mesh_axis} has a "
                    f"dynamic extent {extent!r}; cannot factorize logical "
                    f"axis {logical_axis}"
                )
            factors.append(extent)
            extent_product *= extent
        if axis_size % extent_product:
            raise ValueError(
                f"default_shard_layout: logical axis {logical_axis} size "
                f"{axis_size} is not divisible by mesh extent product "
                f"{extent_product}"
            )
        residual = axis_size // extent_product
    if residual != 1:
        factors.append(residual)
    return tuple(factors)


def factor_domain(shape, mesh_shape, attrs):
    """Translate logical-axis Split attributes to factored domain positions."""
    bindings = {}
    for mesh_axis, attr in enumerate(attrs):
        if isinstance(attr, Split):
            if mesh_axis >= len(mesh_shape) or attr.axis >= len(shape):
                raise ValueError(f"Split({attr.axis}) has no matching mesh axis {mesh_axis}")
            bindings.setdefault(attr.axis, []).append(mesh_axis)
    domain, positions = [], {}
    for logical_axis, extent in enumerate(shape):
        splitting = bindings.get(logical_axis, ())
        for index, mesh_axis in enumerate(splitting):
            positions[mesh_axis] = len(domain) + index
        domain.extend(factor_axis(extent, mesh_shape, splitting, logical_axis) if splitting else (extent,))
    return tuple(domain), tuple(
        Split(positions[axis]) if isinstance(attr, Split) else attr
        for axis, attr in enumerate(attrs)
    )


@dataclass(frozen=True)
class DomainProjection:
    """Factored domain coordinates and their local logical tensor shape."""

    shape: tuple
    profile: tuple
    coordinates: tuple
    source_modes: tuple[int, ...]
    steps: tuple
    split_positions: tuple[int | None, ...]
    local_shape: tuple

    def apply(self, mapping):
        """Compose the projection with a logical index or physical address map."""
        from tilefoundry.ir.isl_interop import normalize_dim  # noqa: PLC0415

        from .layout import flatten as flatten_layout  # noqa: PLC0415
        from .layout import unflatten as unflatten_layout  # noqa: PLC0415
        from .layout_algebra import slice_and_offset  # noqa: PLC0415

        primitive = mapping.outer if isinstance(mapping, ComposedLayout) else mapping
        flat = flatten_layout(primitive)
        strides = tuple(
            normalize_dim(crd2idx(step, extent, flat.strides[mode]))
            for step, extent, mode in zip(self.steps, self.shape, self.source_modes, strict=True)
        )
        grouped = unflatten_layout(Layout(self.shape, strides), self.profile)
        sliced, offset = slice_and_offset(
            unflatten(self.coordinates, self.profile),
            replace(mapping, outer=grouped) if isinstance(mapping, ComposedLayout) else grouped,
        )
        inner = sliced.outer if isinstance(sliced, ComposedLayout) else sliced
        inner = unflatten_layout(flatten_layout(inner), compatible_profile(inner.shape, self.local_shape))
        result = replace(sliced, outer=inner, offset=normalize_dim(sliced.offset)) if isinstance(sliced, ComposedLayout) else inner
        return result, normalize_dim(offset)


def project_domain(domain_shape, tensor_shape, mesh_shape, attrs, coordinates):
    """Fix selected mesh coordinates; None preserves an axis for later composition."""
    from tilefoundry.ir.isl_interop import normalize_dim  # noqa: PLC0415

    compatible_profile(domain_shape, tensor_shape)
    coordinates = tuple(coordinates) + (None,) * (len(attrs) - len(coordinates))
    shape, fixed, modes, steps = [], [], [], []
    positions = [None] * len(attrs)
    for mode, extent in enumerate(flatten(domain_shape)):
        axes = [axis for axis, attr in enumerate(attrs) if isinstance(attr, Split) and attr.axis == mode]
        normalize = any(coordinates[axis] is not None for axis in axes)
        factors = factor_axis(extent, mesh_shape, axes, mode) if normalize else (extent,)
        for index, mesh_axis in enumerate(axes):
            positions[mesh_axis] = len(shape) + (index if normalize else 0)
        for index, (factor, step) in enumerate(zip(factors, compact_row_major(factors), strict=True)):
            axis = axes[index] if index < len(axes) else None
            symbolic_whole = len(axes) == 1 and not isinstance(extent, int)
            fixed.append(
                coordinates[axis]
                if axis is not None and not symbolic_whole and factor == mesh_shape[axis]
                else None
            )
            shape.append(factor)
            modes.append(mode)
            steps.append(step)
    profile = compatible_profile(tuple(shape), tensor_shape)
    local_shape = tuple(
        normalize_dim(product(tuple(n for n, at in zip(flatten(mode), flatten(coord), strict=True) if at is None)))
        for mode, coord in zip(profile, unflatten(tuple(fixed), profile), strict=True)
    )
    return DomainProjection(tuple(shape), profile, tuple(fixed), tuple(modes), tuple(steps), tuple(positions), local_shape)


def logical_projection(domain_shape, tensor_shape, mesh_shape, attrs, coordinates):
    """Apply the coordinate projection to the canonical row-major logical map."""
    projection = project_domain(domain_shape, tensor_shape, mesh_shape, attrs, coordinates)
    mapping, offset = projection.apply(Layout(domain_shape))
    return axis_layout(mapping), offset


def axis_layout(mapping):
    """Regroup a projected coordinate map into its logical tensor axes."""
    from .layout_algebra import coalesce  # noqa: PLC0415

    result = coalesce(mapping, (0,) * len(mapping.shape), major="row")
    primitive = result.outer if isinstance(result, ComposedLayout) else result
    shape = tuple(mode[0] if isinstance(mode, tuple) and len(mode) == 1 else mode for mode in primitive.shape)
    strides = tuple(mode[0] if isinstance(mode, tuple) and len(mode) == 1 else mode for mode in primitive.strides)
    plain = Layout(shape, strides)
    return replace(result, outer=plain) if isinstance(result, ComposedLayout) else plain
