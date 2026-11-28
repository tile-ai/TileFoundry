"""Feature-map geometry shared by register encoding, HWX bindings and the launcher."""

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

from .arith import align_up

ELEMENT_BYTES = MappingProxyType({"f16": 2, "f32": 4})
ELEMENT_CODES = MappingProxyType({"f16": 5, "f32": 6})
SURFACE_ROW_ALIGN = 64
L2_ROW_ALIGN = 16
NE_CHANNEL_GROUP = 16


class Engine(IntEnum):
    """The engine a staged buffer is prepared for."""

    PLANAR = 0
    NEURAL = 1


def canonical_strides(dtype: str, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Canonical NCHW strides in elements, including the DMA row padding."""
    if dtype not in ELEMENT_BYTES:
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    if len(shape) != 4 or any(type(n) is not int or not 0 < n < 2**32 for n in shape):
        raise ValueError("expected four positive tensor extents")
    row = align_up(shape[3] * ELEMENT_BYTES[dtype], SURFACE_ROW_ALIGN) // ELEMENT_BYTES[dtype]
    plane = shape[2] * row
    return shape[1] * plane, plane, row, 1


@dataclass(frozen=True)
class TensorDesc:
    """A feature map's dtype, NCHW shape and strides in elements."""

    dtype: str
    shape: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]

    @classmethod
    def from_shape(cls, dtype: str, shape: tuple[int, int, int, int]) -> "TensorDesc":
        """Construct the canonical DMA layout of a feature map."""
        desc = cls(dtype, shape, canonical_strides(dtype, shape))
        desc.validate()
        return desc

    @property
    def channels(self) -> int:
        """Number of channels in the feature map."""
        return self.shape[1]

    @property
    def height(self) -> int:
        """Number of rows in the feature map."""
        return self.shape[2]

    @property
    def width(self) -> int:
        """Number of columns in the feature map."""
        return self.shape[3]

    @property
    def element_bytes(self) -> int:
        """Bytes per element of the declared dtype."""
        if self.dtype not in ELEMENT_BYTES:
            raise ValueError(f"unsupported tensor dtype {self.dtype!r}")
        return ELEMENT_BYTES[self.dtype]

    @property
    def element_code(self) -> int:
        """Element code used by the HWX binding record."""
        return ELEMENT_CODES[self.dtype]

    @property
    def byte_strides(self) -> tuple[int, int, int, int]:
        """Byte steps with singleton axes filled from the inner storage blocks."""
        steps = list(canonical_strides(self.dtype, self.shape))
        for axis in range(3, -1, -1):
            if self.shape[axis] > 1:
                steps[axis] = self.strides[axis]
            elif axis < 2:
                steps[axis] = self.shape[axis + 1] * steps[axis + 1]
        return tuple(step * self.element_bytes for step in steps)

    @property
    def row(self) -> int:
        """Bytes between DRAM rows, including singleton row padding."""
        return self.byte_strides[2]

    @property
    def channel(self) -> int:
        """Bytes between DRAM channel planes."""
        return self.byte_strides[1]

    @property
    def storage_bytes(self) -> int:
        """Bytes reserved for all batches and complete padded channel planes."""
        batch, plane, _, _ = self.byte_strides
        return (self.shape[0] - 1) * batch + self.channels * plane

    @property
    def declared_geometry(self) -> tuple[int, int, int, int]:
        """Declared row, plane, batch and storage sizes in bytes."""
        batch, plane, row, _ = self.byte_strides
        return row, plane, batch, self.storage_bytes

    @property
    def canonical_geometry(self) -> tuple[int, int, int, int]:
        """Canonical row, plane, batch and storage sizes in bytes."""
        batch, plane, row, _ = canonical_strides(self.dtype, self.shape)
        element = self.element_bytes
        return row * element, plane * element, batch * element, self.shape[0] * batch * element

    @property
    def is_canonical(self) -> bool:
        """Whether every non-singleton axis has the canonical stride."""
        return all(
            extent == 1 or actual == wanted
            for extent, actual, wanted in zip(
                self.shape, self.strides, canonical_strides(self.dtype, self.shape)
            )
        )

    def validate(self, layout: str = "canonical") -> None:
        """Reject unsupported dtypes, invalid extents and incompatible strides."""
        wanted = canonical_strides(self.dtype, self.shape)
        if len(self.strides) != 4 or any(
            type(n) is not int or not 0 <= n < 2**63 for n in self.strides
        ):
            raise ValueError("expected four nonnegative tensor strides")
        if layout == "canonical":
            if not self.is_canonical:
                raise ValueError(
                    f"tensor layout {self.strides} is not canonical for {self.shape}; "
                    f"declare strides {wanted}"
                )
        elif layout != "declared":
            raise ValueError(f"unknown layout policy {layout!r}")
        batch, plane, row, element = self.byte_strides
        if element != self.element_bytes or row < self.width * element or plane < self.height * row:
            raise ValueError("tensor strides do not cover the tensor")
        if self.shape[0] > 1 and batch < self.channels * plane:
            raise ValueError("batch stride does not cover the channels")
        if not 0 < self.storage_bytes < 2**63:
            raise ValueError("tensor storage is empty or overflows")

    def validate_feature(self) -> None:
        """Require a well-formed fp16 feature map with one batch for engine emission."""
        self.validate("declared")
        if self.dtype != "f16" or self.shape[0] != 1:
            raise ValueError("feature map dtype or batch is not decoded yet; expected fp16 N=1")

    @property
    def dram_row_stride(self) -> int:
        """The DMA row-stride field, left clear when any feature extent is one."""
        return 0 if min(self.shape[1:]) == 1 else self.row

    @property
    def l2_channel(self) -> int:
        """L2 channel pitch: one row, aligned to the L2 row unit."""
        return align_up(self.width * self.element_bytes, L2_ROW_ALIGN)

    @property
    def l2_row(self) -> int:
        """L2 row stride: every channel of one row, unpadded."""
        return self.channels * self.l2_channel

    @property
    def l2_extent(self) -> int:
        """Bytes the whole feature map occupies in L2."""
        return self.l2_row * self.height


__all__ = ["Engine", "NE_CHANNEL_GROUP", "SURFACE_ROW_ALIGN", "TensorDesc"]
