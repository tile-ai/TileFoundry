"""Target-neutral HIR-to-TIR candidate pairings."""

from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.tir.memory.copy import Copy

from . import register_candidates

register_candidates(Reshard, (Copy,))
