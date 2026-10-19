"""Apple AMX matrix instruction candidates."""

from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.tir.amx.instructions import Fma16

from . import register_candidates

register_candidates(MatMul, (Fma16,))
