"""The logical-coordinate interpretation shared by layout readers.

See [semantic-analysis §3.1](docs/spec/semantic-analysis.md#31-logical-shape-to-layout-domain).
"""

from __future__ import annotations

from dataclasses import dataclass

from .int_tuple import flatten
from .int_tuple import product as shape_size


def _same_extent(left, right):
    from ..isl_interop import normalize_dim  # noqa: PLC0415
    from .dim import DimSub, simplify_dim  # noqa: PLC0415

    if left is right or left == right:
        return True
    try:
        return normalize_dim(simplify_dim(DimSub, (left, right))) == 0
    except (TypeError, ValueError, NotImplementedError):
        return False


@dataclass(frozen=True)
class LayoutCorrespondence:
    """Mode coordinates and their logical-axis groups, in authored positions."""

    shape: tuple
    extents: tuple
    groups: tuple[tuple[int, ...], ...] | None
    layout_shape: tuple

    def tensor_axes(self) -> list[int]:
        """Reject a mode crossing logical axes instead of assigning it arbitrarily.

        Leading units belong to the following group, trailing units to the final
        axis; unit modes contribute no coordinate.
        """
        if self.groups is None:
            raise ValueError(
                f"layout positions {self.extents} cross logical axes of {self.shape}"
            )
        axes = [None] * len(self.extents)
        for axis, group in enumerate(self.groups):
            for position in group:
                axes[position] = axis
        return [len(self.shape) - 1 if axis is None else axis for axis in axes]

    def axis_mode(self, values, axis):
        """Read one logical axis's authored shape or stride mode tree."""
        if any(isinstance(mode, tuple) for mode in self.layout_shape) and len(values) == len(
            self.shape
        ):
            return values[axis]
        flat = tuple(flatten(values))
        mode = tuple(flat[position] for position in self.groups[axis])
        return mode[0] if len(mode) == 1 else mode

    def digits(self, linear: int) -> tuple[int, ...]:
        """Mode digits of a static logical row-major index, for concrete readers."""
        from .utils import static_dim_value  # noqa: PLC0415

        digits = [0] * len(self.extents)
        held = [position for position, extent in enumerate(self.extents) if static_dim_value(extent) != 1]
        for position in reversed(held):
            extent = static_dim_value(self.extents[position])
            if extent is None or extent <= 0:
                raise ValueError(f"concrete mode coordinates need positive static extents: {self.extents}")
            digits[position] = linear if position == held[0] else linear % extent
            linear //= extent
        return tuple(digits)

    def coordinates(self, coords: list[str]) -> list[str]:
        """ISL expressions for each mode; the outer digit is not folded modulo."""
        from .utils import static_dim_value  # noqa: PLC0415

        image = ["0"] * len(self.extents)
        self.tensor_axes()
        runs = zip(coords, self.groups, strict=True)
        for coord, group in runs:
            held = [position for position in group if static_dim_value(self.extents[position]) != 1]
            inner = 1
            for position in reversed(held):
                term = coord if inner == 1 else f"floor({coord}/{inner})"
                if position == held[0]:
                    image[position] = term
                    break
                extent = static_dim_value(self.extents[position])
                if extent is None:
                    raise ValueError(
                        f"layout position {position} has extent {self.extents[position]!r}; "
                        "dividing a coordinate by it is not affine"
                    )
                image[position] = f"({term}) mod {extent}"
                inner *= extent
        return image


def layout_correspondence(
    layout_shape: tuple, tensor_shape: tuple
) -> LayoutCorrespondence:
    """Compute authored row-major mode coordinates and logical-axis groups.

    Strides are physical addresses and do not participate in correspondence.
    A unit logical axis can own an authored unit mode, but owns no digit.
    """
    from .dim import DimMul, simplify_dim  # noqa: PLC0415
    from .utils import static_dim_value  # noqa: PLC0415

    extents = tuple(flatten(layout_shape))
    shape = tuple(tensor_shape)
    if (
        any(isinstance(mode, tuple) for mode in layout_shape)
        and len(layout_shape) == len(shape)
        and all(_same_extent(shape_size(mode), extent) for mode, extent in zip(layout_shape, shape, strict=True))
    ):
        groups = []
        cursor = 0
        for mode in layout_shape:
            width = len(tuple(flatten(mode)))
            groups.append(tuple(range(cursor, cursor + width)))
            cursor += width
        return LayoutCorrespondence(shape, extents, tuple(groups), tuple(layout_shape))
    groups = []
    cursor = 0
    for extent in shape:
        group = []
        product = 1
        if static_dim_value(extent) == 1 and cursor < len(extents) and static_dim_value(extents[cursor]) == 1:
            group.append(cursor)
            cursor += 1
        while not _same_extent(product, extent):
            if cursor >= len(extents):
                return LayoutCorrespondence(shape, extents, None, tuple(layout_shape))
            position = cursor
            held = extents[position]
            left, right = static_dim_value(product), static_dim_value(held)
            product = left * right if left is not None and right is not None else (
                held if product == 1 else simplify_dim(DimMul, (product, held))
            )
            a, b = static_dim_value(product), static_dim_value(extent)
            if a is not None and b is not None and a > b:
                return LayoutCorrespondence(shape, extents, None, tuple(layout_shape))
            group.append(position)
            cursor += 1
        groups.append(tuple(group))
    if any(static_dim_value(extent) != 1 for extent in extents[cursor:]):
        return LayoutCorrespondence(shape, extents, None, tuple(layout_shape))
    if groups and cursor < len(extents):
        groups[-1] = (*groups[-1], *range(cursor, len(extents)))
    return LayoutCorrespondence(shape, extents, tuple(groups), tuple(layout_shape))
