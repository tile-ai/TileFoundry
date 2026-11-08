"""Metal declarations; each owning module registers its semantics."""
from .matrix import LdMatrix, MatMul, Mma8x8x8, StMatrix
from .memory import (
    AtomicCas,
    AtomicKind,
    AtomicLoad,
    AtomicOrder,
    AtomicRmw,
    AtomicStore,
    CoherentLoad,
    CoherentStore,
)
from .nax import LdMatrix16, MatMul16x32x16, Mma16x32x16, StMatrix16
from .simd import (
    Ballot,
    PrefixProduct,
    PrefixSum,
    Reduce,
    ReductionKind,
    Shuffle,
    ShuffleFill,
    ShuffleKind,
)
from .sync import Barrier, Fence, MemoryFence, ThreadScope

METAL_OPS = (LdMatrix, StMatrix, MatMul, Mma8x8x8, Reduce, Shuffle, ShuffleFill,
    Ballot, PrefixSum, PrefixProduct, Barrier, Fence, AtomicLoad, AtomicStore,
    AtomicRmw, AtomicCas, CoherentLoad, CoherentStore, LdMatrix16, StMatrix16,
    MatMul16x32x16, Mma16x32x16)
__all__ = ["METAL_OPS", "ReductionKind", "ShuffleKind", "MemoryFence", "ThreadScope",
    "AtomicKind", "AtomicOrder", *(op.__name__ for op in METAL_OPS)]
