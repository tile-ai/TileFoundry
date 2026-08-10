"""``T.cuda`` — the CUDA (NVIDIA-GPU) platform sub-namespace of ``dsl.T``."""

from __future__ import annotations

from tilefoundry.ir.tir.cuda.nn import mma as _cuda_mma
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom, MmaOpSpec


class _MmaNamespace:
    """``T.cuda.mma`` — named MMA instructions + the ``atom(op=...)`` builder."""

    SM80_16x8x16_F32BF16BF16F32_TN: MmaOpSpec = _cuda_mma.SM80_16x8x16_F32BF16BF16F32_TN

    @staticmethod
    def atom(op: MmaOpSpec) -> MmaAtom:
        """Realize an :class:`MmaAtom` from a named op (CuTe ``make_tiled_mma``)."""
        return _cuda_mma.make_atom(op)


class _CudaNamespace:
    """``T.cuda`` — the CUDA platform namespace."""

    mma = _MmaNamespace()


cuda = _CudaNamespace()


PLATFORM_NAMESPACES = {"cuda": cuda}

__all__ = ["cuda", "PLATFORM_NAMESPACES"]
