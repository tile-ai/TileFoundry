"""CUDA HIR-to-TIR candidate pairings."""

from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry.ir.tir.cuda.memory.ldmatrix import LdMatrix
from tilefoundry.ir.tir.cuda.nn.mma import TiledMma

from . import register_candidates

register_candidates(MatMul, (TiledMma,))
register_candidates(Reshard, (CopyAsync, CopyAsyncTensor, LdMatrix))
