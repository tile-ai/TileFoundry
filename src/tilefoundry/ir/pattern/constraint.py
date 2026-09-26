"""Relations between operands of one operation declaration."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Swizzle
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.tensor_type import TensorType

from .match import _named


@dataclass(frozen=True)
class Constraint:
    """One relation between two named operands."""

    field: str
    left: str
    right: str

    def pair(self, operands: dict):
        held = tuple(operands.get(name) for name in (self.left, self.right))
        if any(value is None for value in held):
            return None
        found = tuple(getattr(value, self.field, None) for value in held)
        return None if any(value is None for value in found) else found

    def holds(self, operands: dict) -> bool:
        raise NotImplementedError

    def written(self) -> str:
        raise NotImplementedError

    def refused(self, operands: dict) -> str:
        pair = self.pair(operands)
        if pair is None:
            return self.written()
        left, right = pair
        return (
            f"{self.left}.{self.field}={_named(left)} "
            f"{self.right}.{self.field}={_named(right)}, reads {self.written()}"
        )


@dataclass(frozen=True)
class DistinctConstraint(Constraint):
    def holds(self, operands: dict) -> bool:
        pair = self.pair(operands)
        return pair is None or pair[0] != pair[1]

    def written(self) -> str:
        return f"{self.left}.{self.field} != {self.right}.{self.field}"


@dataclass(frozen=True)
class SameConstraint(Constraint):
    def holds(self, operands: dict) -> bool:
        pair = self.pair(operands)
        return pair is None or pair[0] == pair[1]

    def written(self) -> str:
        return f"{self.left}.{self.field} = {self.right}.{self.field}"


@dataclass(frozen=True, init=False)
class SameModesConstraint(Constraint):
    """Require two tensor layouts to run along the same final mode of one axis."""

    def __init__(self, left: str, right: str):
        object.__setattr__(self, "field", "layout")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    def pair(self, operands: dict):
        held = tuple(operands.get(name) for name in (self.left, self.right))
        return None if any(not isinstance(value, TensorType) for value in held) else held

    @staticmethod
    def reading(tensor: TensorType, arrangement=None):
        layout = tensor.layout if arrangement is None else arrangement
        if isinstance(layout, ComposedLayout):
            if layout.inner is not None and not isinstance(layout.inner, Swizzle):
                layout = None
            else:
                layout = layout.outer
        if not isinstance(layout, Layout) or layout.strides is None:
            return None, f"{tensor.layout!r} is no strided arrangement"
        shape, strides = tuple(layout.shape), tuple(layout.strides)
        extents = tuple(tensor.shape)
        if len(shape) != len(extents) or any(
            prod(flatten(group)) != extent for group, extent in zip(shape, extents)
        ):
            return None, (
                f"{layout!r} is not one group of modes per axis of the {extents} tile it arranges"
            )
        unit = []
        for axis, (group, steps) in enumerate(zip(shape, strides)):
            walked = tuple(
                step for extent, step in zip(flatten(group), flatten(steps)) if extent > 1
            )
            unit += [
                (axis, position, position == len(walked) - 1)
                for position, step in enumerate(walked)
                if step == 1
            ]
        if len(unit) != 1:
            return None, f"{layout!r} has {len(unit)} modes at step 1, not one"
        return unit[0], None

    def readings(self, pair: tuple[TensorType, TensorType]):
        """Read below one shared shard frame only after proving it is the same."""
        left, right = (value.layout for value in pair)
        if (
            isinstance(left, ShardLayout)
            and isinstance(right, ShardLayout)
            and left.attrs == right.attrs
            and left.mesh == right.mesh
        ):
            arrangements = (left.layout, right.layout)
        else:
            arrangements = (None, None)
        return tuple(
            self.reading(value, arrangement)
            for value, arrangement in zip(pair, arrangements)
        )

    def holds(self, operands: dict) -> bool:
        pair = self.pair(operands)
        if pair is None:
            return True
        (left, _), (right, _) = self.readings(pair)
        return (
            left is not None and right is not None and left[0] == right[0] and left[2] and right[2]
        )

    def written(self) -> str:
        return (
            f"{self.left} and {self.right} step 1 along the last mode of one tile "
            "axis, each grouped by tile axis"
        )

    def refused(self, operands: dict) -> str:
        pair = self.pair(operands)
        if pair is None:
            return self.written()
        readings = tuple(
            (name, *reading)
            for name, reading in zip((self.left, self.right), self.readings(pair))
        )
        for name, _, why in readings:
            if why is not None:
                return f"{name}.layout: {why}; reads {self.written()}"
        (
            (left_name, (axis, _, left_fastest), _),
            (
                right_name,
                (other, _, right_fastest),
                _,
            ),
        ) = readings
        if axis != other:
            return (
                f"{left_name} steps 1 along tile axis {axis} and {right_name} along "
                f"tile axis {other}: that is a transpose; reads {self.written()}"
            )
        name = left_name if not left_fastest else right_name
        return (
            f"{name} does not step 1 along the last mode of tile axis {axis}; "
            f"reads {self.written()}"
        )


__all__ = [
    "Constraint",
    "DistinctConstraint",
    "SameConstraint",
    "SameModesConstraint",
]
