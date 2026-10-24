"""Apple AMX matrix instruction candidates."""

from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.visitor_registry.candidates import register_candidates
from tilefoundry_apple.amx.ir.tir.instructions import Fma16

register_candidates(MatMul, (Fma16,))
