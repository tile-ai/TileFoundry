"""SM90 fragment stores checked against compiled CuTe copy traits."""

from dataclasses import replace

import pytest

from tilefoundry.ir.tir.cuda.memory.ldmatrix import LdMatrix
from tilefoundry.ir.tir.cuda.memory.stmatrix import StMatrix
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.cuda.nn.wgmma import WARPGROUP, Form, Wgmma
from tilefoundry.ir.types import DType, Layout, Mesh, StorageKind, TensorType, Topology
from tilefoundry.ir.types.int_tuple import compatible_profile
from tilefoundry.ir.types.layout import flatten, unflatten
from tilefoundry.ir.types.stride import crd2idx
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, check_call

from ._cute_reference import cute_reference
from ._operand_layout import _bound_fragment


def _operands(n=48, dtype=DType.bf16):
    src = TensorType(
        (64, n),
        DType.bf16,
        _bound_fragment(Wgmma(n=n, dtype="bf16", form=Form.SS), "C"),
        StorageKind.RMEM,
    )
    dst = TensorType((64, n), DType.bf16, Layout((64, n), (n, 1)), StorageKind.SMEM)
    return replace(src, dtype=dtype), replace(dst, dtype=dtype)


@pytest.mark.parametrize("dtype", (DType.bf16, DType.f16), ids=lambda dtype: dtype.name)
@pytest.mark.parametrize("n", (16, 48, 256))
def test_stmatrix_matches_cute_per_lane(n, dtype):
    src, dst = _operands(n, dtype)
    check_call(StMatrix(), (src, dst), WARPGROUP, frame=None)
    reference = cute_reference()["stmatrix"]

    def index(raw, lane, value):
        shape = raw["shape"][1]
        shape = shape if isinstance(shape, list) else [shape]
        strides = raw["strides"][1]
        strides = strides if isinstance(strides, list) else [strides]
        result = lane * raw["strides"][0]
        for extent, stride in zip(shape, strides, strict=True):
            result += (value % extent) * stride
            value //= extent
        return result

    destination = {
        index(reference["dst"], lane, value): (lane, value // 16)
        for lane in range(32)
        for value in range(0, 128, 16)
    }
    primitive = flatten(src.layout.layout)
    coords = compatible_profile(primitive.shape, src.shape)
    digit_layouts = tuple(
        unflatten(Layout(primitive.shape, tuple(int(i == p) for i in range(len(primitive.shape)))), coords)
        for p in range(len(primitive.shape))
    )
    for warp in range(4):
        for issue in range(n // 16):
            for lane in range(32):
                for register in range(4):
                    for half in range(2):
                        address = index(reference["src"], lane, 32 * register + 16 * half)
                        row_lane, column = destination[address]
                        row = 16 * warp + row_lane % 8 + 8 * ((row_lane // 8) % 2)
                        column += 16 * issue + 8 * (row_lane // 16)
                        digits = tuple(
                            crd2idx((row, column), layout.shape, layout.strides)
                            for layout in digit_layouts
                        )
                        assert tuple(digits[position] for position in (0, 2, 4)) == (
                            warp,
                            lane // 4,
                            lane % 4,
                        )
                        assert tuple(digits[position] for position in (1, 3, 5)) == (
                            register % 2,
                            2 * issue + register // 2,
                            half,
                        )
                        assert (
                            sum(i * s for i, s in zip(digits, primitive.strides, strict=True))
                            == row + 64 * column
                        )


@pytest.mark.parametrize("failure", ("f32", "odd_c", "shape", "dtype", "two_groups"))
def test_stmatrix_rejects_incompatible_operands(failure):
    src, dst = _operands(40 if failure == "odd_c" else 48)
    mesh = WARPGROUP
    if failure == "f32":
        src = replace(src, dtype=DType.f32)
    elif failure == "shape":
        dst = replace(dst, shape=(64, 64), layout=Layout((64, 64), (64, 1)))
    elif failure == "dtype":
        dst = replace(dst, dtype=DType.f16)
    elif failure == "two_groups":
        mesh = Mesh((Topology("thread", 256),), Layout((2, 4, 8, 4), (128, 32, 4, 1)))
        src = replace(src, layout=replace(src.layout, mesh=mesh))
    with pytest.raises(ScheduleRefusal, match="src|dst|mesh|shape|dtype"):
        check_call(StMatrix(), (src, dst), mesh, frame=None)


def test_ldmatrix_rejects_dtype_mismatch():
    src = TensorType((16, 16), DType.f16, Layout((16, 16), (16, 1)), StorageKind.SMEM)
    dst = TensorType((16, 16), DType.bf16, _bound_fragment(Mma(), "A"), StorageKind.RMEM)
    with pytest.raises(ScheduleRefusal, match="dtype"):
        check_call(LdMatrix(), (src, dst), Mma.execution_mesh, frame=None)
