"""How ANE instructions see their operands.

A feature map is [1, C, H, W] in unified memory (GMEM) or in the on-chip L2
buffer (SMEM); an engine task reads and writes feature maps in SMEM. Its W stride
is 1: the engine addresses a feature map by row, channel, depth and group strides
and has no element stride. Weights stay in unified memory. A matrix is [rows,
columns] with rows contiguous (column major) or columns contiguous (row major).
A mesh layout holds one mode per topology.
"""

from dataclasses import dataclass

from tilefoundry.ir import expr, pattern, types

ENGINE = pattern.Mesh(("engine",), pattern.Layout(((1,),), ((1,),)))
FLOATS = (expr.DType.f16,)
GMEM_ONLY = pattern.Or(expr.StorageKind.GMEM)
SMEM_ONLY = pattern.Or(expr.StorageKind.SMEM)
STRIDES = ((1, 1), (1, 2), (2, 1), (2, 2))
_ = pattern.Wildcard()

MAX_CHANNELS = 65536
MAX_HEIGHT = 16384
MAX_WIDTH = 16384
MAX_KERNEL = 32
ACTIVATION_LIMITS = (MAX_WIDTH, MAX_CHANNELS)
WEIGHT_LIMITS = (MAX_CHANNELS, MAX_CHANNELS)


def extent(name: str, limit: int) -> pattern.Wildcard:
    """An extent of at most *limit*, captured as *name*."""
    return pattern.Wildcard(name, predicates=(pattern.Range(hi=limit),))


def rows(operand: str) -> dict:
    """The strides of feature map *operand*: a W stride of 1 unless W has extent 1."""
    width, step = pattern.Wildcard(f"{operand}.shape[3]"), pattern.Wildcard(f"{operand}.strides[3]")
    return {"strides": (_, _, _, step), "predicates": ((width == 1) | (step == 1),)}


def stored(operand: str, choices) -> pattern.Wildcard:
    """The storage of *operand*, captured as its name and limited to *choices*."""
    return pattern.Wildcard(f"{operand}.storage", predicates=(choices,))


def feature(operand: str, storage) -> pattern.Tensor:
    """[1, C, H, W] in *storage* with a W stride of 1, within the architecture limits.

    Extents and storage are captured under *operand*; every operand shares one dtype.
    """
    return pattern.Tensor(
        shape=(
            1,
            extent(f"{operand}.shape[1]", MAX_CHANNELS),
            extent(f"{operand}.shape[2]", MAX_HEIGHT),
            extent(f"{operand}.shape[3]", MAX_WIDTH),
        ),
        dtype=one_of("dtype", FLOATS),
        storage=stored(operand, storage),
        layout=pattern.Layout(**rows(operand)),
    )


def bounded(
    operand: str, leading: tuple, *limits: int, strides: tuple | None = None, predicates: tuple = ()
) -> pattern.Layout:
    """The layout of a tensor shaped *leading*, then extents of at most *limits*.

    *strides* and *predicates* add a rule on the strides; without them any strides match.

    Each extent after *leading* is captured as ``<operand>.shape[axis]``.
    """
    names = tuple(
        pattern.Wildcard(f"{operand}.shape[{len(leading) + axis}]") for axis in range(len(limits))
    )
    return pattern.Layout(
        (*leading, *names),
        strides,
        predicates=(*(name <= limit for name, limit in zip(names, limits)), *predicates),
    )


def feature_layout(operand: str) -> pattern.Layout:
    """The layout of a [1, C, H, W] feature map within the limits; its W stride is 1."""
    return bounded(operand, (1,), MAX_CHANNELS, MAX_HEIGHT, MAX_WIDTH, **rows(operand))


def kernel_layout(operand: str) -> pattern.Layout:
    """The layout of [O, I, KH, KW] weights, KH and KW at most MAX_KERNEL."""
    return bounded(operand, (), MAX_CHANNELS, MAX_CHANNELS, MAX_KERNEL, MAX_KERNEL)


def matrix_layout(operand: str, limits: tuple[int, int]) -> pattern.Layout:
    """The layout of an [R, C] matrix whose extents stay within *limits*."""
    return bounded(operand, (), *limits)


def strided(type_: types.TensorType, *candidates: tuple) -> bool:
    """Whether the strides of *type_* equal one of *candidates*, ignoring modes of extent 1."""
    plain = types.plain_layout(type_.layout)
    return any(
        all(
            size == 1 or actual == step
            for size, actual, step in zip(plain.shape, plain.strides, wanted)
        )
        for wanted in candidates
    )


def require_strides(call, ctx, name: str, type_: types.TensorType, *candidates: tuple) -> None:
    """Refuse *call* unless the strides of operand *name* equal one of *candidates*."""
    if not strided(type_, *candidates):
        ctx.error(
            call,
            f"{name} layout strides {tuple(types.plain_layout(type_.layout).strides)} "
            f"must be {' or '.join(map(str, candidates))}",
        )


def require_strides_or_dense(
    call, ctx, name: str, type_: types.TensorType, *candidates: tuple
) -> None:
    """Refuse *call* unless *name* matches a candidate or has a dense innermost axis."""
    if strided(type_, *candidates):
        return
    require_innermost_contiguous(call, ctx, name, type_)


def require_innermost_contiguous(call, ctx, name: str, type_: types.TensorType) -> None:
    """Refuse *call* unless the innermost stride of *name* is one.

    A weight view of a larger matrix keeps the parent's outer strides; only the
    innermost axis has to be dense for the coefficient DMA to read it.
    """
    plain = types.plain_layout(type_.layout)
    if plain.shape[-1] != 1 and plain.strides[-1] != 1:
        ctx.error(call, f"{name} innermost stride {plain.strides[-1]} must be 1")


def feature_dma_strides(type_: types.TensorType) -> tuple[int, int]:
    """Read static feature map row and channel strides in bytes."""
    plain = type_.layout
    if not isinstance(plain, types.Layout) or len(plain.strides) != 4:
        raise ValueError("feature map requires a flat layout")
    if type_.storage not in (expr.StorageKind.GMEM, expr.StorageKind.SMEM):
        raise ValueError("feature map requires gmem or smem storage")
    alignment = 64 if type_.storage is expr.StorageKind.GMEM else 16
    values: list[int] = []
    for axis, label in ((2, "row"), (1, "channel")):
        step = expr.static_dim_value(plain.strides[axis])
        if step is None or step < 0 or (step == 0 and type_.shape[axis] != 1):
            raise ValueError(f"{label} stride must be static and positive for a non-singleton axis")
        step_bytes = 0 if type_.shape[axis] == 1 else step * (type_.dtype.bit_width // 8)
        if step_bytes % alignment:
            raise ValueError(f"{label} stride {step_bytes} bytes must be a multiple of {alignment}")
        values.append(step_bytes)
    return values[0], values[1]


def forms(operand: str, layouts: tuple, storage) -> pattern.Tensor:
    """An operand in *storage* with the shared dtype whose layout is one of *layouts*.

    Each layout states its own shape, so the rank selects the form. Pairing checks
    only dtype and storage; a shape or layout mismatch is reported against the
    instruction.
    """
    return pattern.Tensor(
        dtype=one_of("dtype", FLOATS), storage=stored(operand, storage), layout=pattern.Or(*layouts)
    )


def activation(operand: str, storage) -> pattern.Tensor:
    """A feature map or an [R, C] activation matrix in *storage*."""
    layouts = (feature_layout(operand), matrix_layout(operand, ACTIVATION_LIMITS))
    return forms(operand, layouts, storage)


@dataclass(frozen=True)
class SameFeatureConstraint(pattern.Constraint):
    """Two shapes that are one feature map, or a feature map and the matrix it is the view of.

    Row m of an [R, C] matrix is row m % H of channel m // H of the [1, R / H, H, C]
    feature map, so both read the same elements in row-major order.
    """

    def holds(self, operands: dict) -> bool:
        pair = self.pair(operands)
        if pair is None:
            return True
        left, right = (tuple(shape) for shape in pair)
        if len(left) == 2:
            left, right = right, left
        if left == right:
            return True
        return (
            len(left) == 4 and len(right) == 2
            and left[0] == 1 and left[3] == right[1] and left[1] * left[2] == right[0]
        )

    def written(self) -> str:
        return (
            f"{self.left}.{self.field} = {self.right}.{self.field}, or one is an [R, C] matrix "
            "and the other its [1, R / H, H, C] feature map"
        )


def one_of(name: str, values) -> pattern.Wildcard:
    """A wildcard admitting exactly *values*."""
    return pattern.Wildcard(name, predicates=(pattern.Or(*values),))


__all__ = [
    "ACTIVATION_LIMITS",
    "activation",
    "ENGINE",
    "FLOATS",
    "GMEM_ONLY",
    "MAX_CHANNELS",
    "MAX_HEIGHT",
    "MAX_KERNEL",
    "MAX_WIDTH",
    "SMEM_ONLY",
    "SameFeatureConstraint",
    "STRIDES",
    "WEIGHT_LIMITS",
    "bounded",
    "extent",
    "feature",
    "feature_layout",
    "forms",
    "feature_dma_strides",
    "kernel_layout",
    "matrix_layout",
    "one_of",
    "require_strides",
    "require_strides_or_dense",
    "require_innermost_contiguous",
    "rows",
    "stored",
    "strided",
]
