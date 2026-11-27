"""H14 coefficient layouts and lane packing shared by every container consumer."""

from dataclasses import dataclass
from itertools import product

from tilefoundry_apple.ane.isa import arith, intrinsic

MAX_COEFF_LANES = 16
COEFF_LANE_ALIGN = 64
LUT_WEIGHT_BYTES = 0x80


@dataclass(frozen=True)
class PackingAxis:
    """A loop extent and its source/destination byte strides for one packing axis."""

    extent: int
    source_step: int
    destination_step: int


@dataclass(frozen=True)
class CoefficientLayout:
    """One OIHW tensor's byte range and optional lookup table within the coefficient section."""

    shape: tuple[int, int, int, int]
    base: int = 0
    lut: bytes = b""

    def __post_init__(self) -> None:
        if len(self.shape) != 4 or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("coefficient shape must have four positive static extents")
        if type(self.base) is not int or self.base < 0 or self.base % COEFF_LANE_ALIGN:
            raise ValueError("coefficient base must be a nonnegative 64-byte offset")
        if self.shape[0] % self.lanes:
            raise ValueError(
                f"output channels {self.shape[0]} do not split into {self.lanes} lanes"
            )
        if self.lut and len(self.lut) != intrinsic.NE_LUT_BYTES:
            raise ValueError(f"lookup table must hold {intrinsic.NE_LUT_BYTES} bytes")
        if self.lut and self.lane_bytes > LUT_WEIGHT_BYTES:
            raise ValueError(
                f"lookup-table lane over {LUT_WEIGHT_BYTES} weight bytes is not decoded yet"
            )

    @property
    def lanes(self) -> int:
        """The number of output-channel lanes."""
        return min(MAX_COEFF_LANES, self.shape[0])

    @property
    def lane_bytes(self) -> int:
        """Unpadded weight bytes consumed by one lane."""
        outputs, inputs, kh, kw = self.shape
        return outputs // self.lanes * inputs * kh * kw * 2

    @property
    def stride(self) -> int:
        """The aligned byte stride between consecutive lanes."""
        return arith.align_up(self.weight_offset + self.lane_bytes, COEFF_LANE_ALIGN)

    @property
    def size(self) -> int:
        """The complete padded byte range of this tensor."""
        return self.lanes * self.stride

    @property
    def weight_offset(self) -> int:
        """The first weight byte within each lane, after its lookup table."""
        return len(self.lut)

    @property
    def weight_axes(self) -> tuple[PackingAxis, ...]:
        """Loop rules in lane, input, kernel row, kernel column, lane-output order."""
        outputs, inputs, kh, kw = self.shape
        per_lane = outputs // self.lanes
        return (
            PackingAxis(self.lanes, per_lane * inputs * kh * kw * 2, self.stride),
            PackingAxis(inputs, kh * kw * 2, kh * kw * per_lane * 2),
            PackingAxis(kh, kw * 2, kw * per_lane * 2),
            PackingAxis(kw, 2, per_lane * 2),
            PackingAxis(per_lane, inputs * kh * kw * 2, 2),
        )

    def writer_body(self) -> str:
        """Generate constant-bound C++ loops from the same axes as offline packing."""
        lines = []
        for index, axis in enumerate(self.weight_axes):
            lines.append(f"  for (size_t i{index} = 0; i{index} < {axis.extent}; ++i{index}) {{")
        source = " + ".join(
            f"i{index} * {axis.source_step}" for index, axis in enumerate(self.weight_axes)
        )
        destination = " + ".join(
            f"i{index} * {axis.destination_step}" for index, axis in enumerate(self.weight_axes)
        )
        lines.append(
            f"    std::memcpy(coefficients + {self.base + self.weight_offset} + {destination}, "
            f"source + {source}, 2);"
        )
        lines.extend("  }" for _ in self.weight_axes)
        return "\n".join(lines)


@dataclass(frozen=True)
class Coefficient:
    """Weight values bound to the same layout used by emission, symbols and relocations."""

    layout: CoefficientLayout
    values: bytes

    def validate(self) -> None:
        """Require a complete tensor before lane packing."""
        if len(self.values) != self.layout.lanes * self.layout.lane_bytes:
            raise ValueError("coefficient bytes do not match the declared shape")

    def pack(self) -> bytes:
        """Reorder little-endian OIHW fp16 values into aligned lane streams led by the table."""
        self.validate()
        layout = self.layout
        out = bytearray(layout.size)
        for lane in range(layout.lanes):
            out[lane * layout.stride : lane * layout.stride + layout.weight_offset] = layout.lut
        axes = layout.weight_axes
        for indices in product(*(range(axis.extent) for axis in axes)):
            source = sum(
                index * axis.source_step for index, axis in zip(indices, axes, strict=True)
            )
            destination = layout.weight_offset + sum(
                index * axis.destination_step for index, axis in zip(indices, axes, strict=True)
            )
            out[destination : destination + 2] = self.values[source : source + 2]
        return bytes(out)


def coefficient_section(coefficients: tuple[Coefficient, ...]) -> bytes:
    """Pack the blocks back to back; each block's planned base is its offset in the section."""
    return b"".join(coefficient.pack() for coefficient in coefficients)


__all__ = [
    "COEFF_LANE_ALIGN", "Coefficient", "CoefficientLayout", "coefficient_section",
]
