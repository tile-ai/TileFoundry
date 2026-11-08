"""Shared pure-layout patterns and binding for Metal primitives."""
from tilefoundry.evaluator.value import EvalError, to_torch_dtype
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.pattern import (
    LayoutPattern,
    MeshPattern,
    OrPattern,
    RangePattern,
    ShardLayoutPattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.types import Broadcast, DType, Split, StorageKind, UnitType

G = WildcardPattern("groups", predicates=(RangePattern(lo=1, hi=32),))
DT = WildcardPattern("dtype")
INT = (DType.i8, DType.u8, DType.i16, DType.u16, DType.i32, DType.u32)
NUM = (DType.f16, DType.f32, *INT)
SIMD_MESH = MeshPattern(("thread",), LayoutPattern(((G, 32),), ((32, 1),)))
MATRIX_MESH = MeshPattern(("thread",), LayoutPattern(
    ((G, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),)))

MATRIX_EXECUTION = MeshPattern(("thread",), OrPattern(SIMD_MESH.layout, MATRIX_MESH.layout))

def attribute(annotation, domain, default, *, name=None):
    if name is not None:
        domain = WildcardPattern(name, predicates=(domain,))
    return ParamDef(kind="attribute", annotation=annotation, pattern=domain, default=default)

def dtype_attribute(types, default):
    return attribute(DType, WildcardPattern("dtype", predicates=(OrPattern(*types),)), default)

def lanes(dtype=DT, *, broadcast=False):
    shape = (G, 1 if broadcast else 32)
    strides = (0, 0)
    return TensorPattern(shape=shape, dtype=dtype, storage=StorageKind.RMEM,
        layout=ShardLayoutPattern(LayoutPattern(shape, strides),
            (Split(0), Broadcast() if broadcast else Split(1)), SIMD_MESH))

def matrix(dtype=DT):
    return TensorPattern(shape=(G, 8, 8), dtype=dtype, storage=StorageKind.RMEM,
        layout=ShardLayoutPattern(
            LayoutPattern((G, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))),
            (Split(0), Split(4), Split(2), Split(3), Split(1)), MATRIX_MESH))

def memory_matrix():
    row = WildcardPattern("row_stride", predicates=(RangePattern(lo=8),))
    batch = WildcardPattern("batch_stride", predicates=(WildcardPattern("batch_stride") >= 8 * row,))
    return TensorPattern(shape=(G, 8, 8), dtype=DT,
        storage=OrPattern(StorageKind.GMEM, StorageKind.SMEM),
        layout=LayoutPattern((G, 8, 8), (batch, row, 1)))


def vectors(dtype=DT, *, broadcast=False, width=32, groups=G, predicates=()):
    """A scalar or native vector per lane, entirely in ordinary layout modes."""
    choices = {}
    mesh = MeshPattern(("thread",), LayoutPattern(((groups, width),), ((width, 1),),
        predicates=(groups * width <= 1024,)))
    for slots in (1, 2, 4):
        shape = (groups, 1 if broadcast else width) + (() if slots == 1 else (slots,))
        strides = (0, 0) + (() if slots == 1 else (1,))
        choices[slots] = TensorPattern(shape=shape, dtype=dtype, storage=StorageKind.RMEM,
            layout=ShardLayoutPattern(LayoutPattern(shape, strides),
                (Split(0), Broadcast() if broadcast else Split(1)), mesh), predicates=predicates)
    return SwitchPattern("slots", choices)


def selector():
    return SwitchPattern("uniform", {
        True: TensorPattern(shape=(), dtype=DType.u16, storage=StorageKind.RMEM,
                            layout=LayoutPattern((), ())),
        False: lanes(DType.u16),
    })

def infer(call, ctx):
    """Instruction calls are statements; admission belongs to check_call."""
    return UnitType()


def cpu_args(ctx):
    """Require concrete CPU data of the declared shape and dtype."""
    for value in ctx.args:
        if (value.data.device.type != "cpu"
                or value.data.dtype != to_torch_dtype(value.type.dtype)
                or tuple(value.data.shape) != tuple(value.type.shape)):
            raise EvalError("Metal evaluator requires CPU data matching its declared type")
    return tuple(value.data for value in ctx.args)


def write_result(ctx, result):
    ctx.args[0].data.copy_(result.to(ctx.args[0].data.dtype))
    return ctx.args[0]
