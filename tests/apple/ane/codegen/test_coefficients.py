"""Coefficient blocks preserve OIHW values through lane reordering and LUT placement."""

import math
import struct
from dataclasses import replace

import pytest

from tilefoundry_apple.ane.codegen import hwx
from tilefoundry_apple.ane.codegen.hwx import coefficients


def test_three_blocks_roundtrip_by_lane():
    blocks = []
    base = 0
    table = struct.pack(f"<{len(coefficients.LUT_SLOTS)}H", *range(3000, 3000 + len(coefficients.LUT_SLOTS)))
    for index, (shape, lut) in enumerate((((32, 3, 2, 2), b""), ((16, 8, 1, 1), table), ((16, 64, 1, 1), b""))):
        values = tuple(index * 1000 + n + 1 for n in range(math.prod(shape)))
        layout = hwx.CoefficientLayout(shape, base, lut)
        blocks.append(hwx.Coefficient(layout, struct.pack(f"<{len(values)}H", *values)))
        base += layout.size
    section = coefficients.coefficient_section(tuple(blocks))

    for index, block in enumerate(blocks):
        layout = block.layout
        outputs, inputs, kh, kw = layout.shape
        per_lane = outputs // layout.lanes
        for output in range(outputs):
            lane, inner = divmod(output, per_lane)
            for channel in range(inputs):
                for row in range(kh):
                    for column in range(kw):
                        source = ((output * inputs + channel) * kh + row) * kw + column
                        within = ((channel * kh + row) * kw + column) * per_lane + inner
                        offset = layout.base + lane * layout.stride + layout.weight_offset + within * 2
                        assert struct.unpack_from("<H", section, offset)[0] == index * 1000 + source + 1
        if layout.lut:
            for lane in range(layout.lanes):
                for slot, offset in enumerate(coefficients.LUT_SLOTS):
                    assert struct.unpack_from("<H", section, layout.base + lane * layout.stride + offset)[0] == 3000 + slot
            assert section[layout.base + 4 : layout.base + 6] == b"\0\0"
        else:
            for lane in range(layout.lanes):
                padding = layout.base + lane * layout.stride + layout.lane_bytes
                assert section[padding : layout.base + (lane + 1) * layout.stride] == bytes(layout.stride - layout.lane_bytes)
    with pytest.raises(ValueError, match="gaps or overlaps"):
        coefficients.coefficient_section((blocks[0], replace(blocks[1], layout=replace(blocks[1].layout, base=0))))


@pytest.mark.parametrize("shape,lut", (((0, 1, 1, 1), b""), ((17, 1, 1, 1), b""), ((16, 65, 1, 1), bytes(len(coefficients.LUT_SLOTS) * 2)), ((16, 1, 1, 1), b"\0\0")))
def test_unsupported_coefficient_layout_is_rejected(shape, lut):
    with pytest.raises(ValueError):
        hwx.CoefficientLayout(shape, lut=lut)
