"""H14 field formulas and cross-field validation of descriptor register maps."""
from __future__ import annotations

from collections.abc import Mapping

from . import dma, fields

NE_CHANNEL_GROUP = 16
SHIFT_MAX = 4
UNICAST_MAX = 8


def ceil_log2(x: int) -> int:
    return (max(x, 1) - 1).bit_length()


def channel_groups(channels: int) -> int:
    groups = -(-channels // NE_CHANNEL_GROUP)
    return 0 if groups <= 1 else (groups - 1).bit_length()


def unicast_channels(channels: int) -> int:
    """The unicast count of a transposing pass: its channel groups rounded up to a power of two."""
    return min(UNICAST_MAX, 1 << ceil_log2(-(-channels // NE_CHANNEL_GROUP)))


def active_nodes(engine_nodes: int) -> int:
    return engine_nodes - 1


def shift(work_unit_dim: int) -> int:
    return min(ceil_log2(work_unit_dim), SHIFT_MAX)


class CheckError(ValueError):
    """A built descriptor violates a cross-field rule."""


def read(image: Mapping[int, int], field: fields.BitField) -> int | None:
    """The field's value when its word is present in the image, else ``None``."""
    word = image.get(field.addr)
    return None if word is None else field.get(word)


def check(image: Mapping[int, int]) -> None:
    """Check effective register relationships and verified tile memory intervals."""
    common, l2 = fields.Common, fields.L2

    mn = read(image, common.shift_minimum)
    pf = read(image, common.shift_preferred)
    mx = read(image, common.shift_maximum)
    if mn is not None and pf is not None and mx is not None:
        if not mn <= pf <= mx:
            raise CheckError(f"shift slices out of order: min={mn} pref={pf} max={mx}")
        if mx > SHIFT_MAX:
            raise CheckError(f"shift_maximum {mx} exceeds {SHIFT_MAX}")

    row = read(image, l2.l2_src_row_stride)
    depth = read(image, l2.l2_src_depth_stride)
    if row is not None and depth is not None and row < depth:
        raise CheckError(f"l2_src_row_stride {row} is below l2_src_depth_stride {depth}")

    if read(image, common.output_transpose) and read(image, fields.NE.op) == fields.NEOp.BYPASS:
        unicast = unicast_channels(read(image, common.cin))
        stated = (read(image, common.unicast_cin), read(image, common.fill_lower_nodes_first))
        if stated != (unicast, int(unicast > 1)):
            raise CheckError(f"transposing bypass unicast and fill-lower {stated} contradict "
                             f"its channels; expected {(unicast, int(unicast > 1))}")

    dma.check_l2(image, dma.geometry(image))


__all__ = [
    "CheckError",
    "NE_CHANNEL_GROUP",
    "SHIFT_MAX",
    "UNICAST_MAX",
    "active_nodes",
    "ceil_log2",
    "channel_groups",
    "check",
    "read",
    "shift",
    "unicast_channels",
]
