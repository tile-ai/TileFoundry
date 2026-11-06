"""Metal declarations; each owning module registers its semantics."""
from .matrix import LdMatrix, StMatrix, MatMul, Mma8x8x8
from .simd import Reduce, ReductionKind, Shuffle, ShuffleFill, ShuffleKind, Ballot, PrefixSum, PrefixProduct
from .memory import AtomicLoad, AtomicStore, AtomicRmw, AtomicCas, CoherentLoad, CoherentStore, AtomicKind, AtomicOrder
from .sync import Barrier, Fence, MemoryFence, ThreadScope
from .nax import LdMatrix16, StMatrix16, MatMul16x32x16, Mma16x32x16

METAL_OPS = (LdMatrix, StMatrix, MatMul, Mma8x8x8, Reduce, Shuffle, ShuffleFill,
    Ballot, PrefixSum, PrefixProduct, Barrier, Fence, AtomicLoad, AtomicStore,
    AtomicRmw, AtomicCas, CoherentLoad, CoherentStore, LdMatrix16, StMatrix16,
    MatMul16x32x16, Mma16x32x16)
__all__ = ["METAL_OPS", "ReductionKind", "ShuffleKind", "MemoryFence", "ThreadScope",
    "AtomicKind", "AtomicOrder", *(op.__name__ for op in METAL_OPS)]
