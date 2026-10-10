"""Target-neutral HIR-to-TIR candidate pairings."""

from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.nn.relu import ReLU as HirReLU
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.where import Where as HirWhere
from tilefoundry.ir.tir.arith import Binary as TirBinary
from tilefoundry.ir.tir.arith import Unary as TirUnary
from tilefoundry.ir.tir.cast import Cast as TirCast
from tilefoundry.ir.tir.clamp import Clamp as TirClamp
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.nn.relu import ReLU as TirReLU
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.tir.where import Where as TirWhere

from . import register_candidates

register_candidates(Reshard, (Copy,))
register_candidates(HirBinary, (TirBinary,))
register_candidates(HirUnary, (TirUnary,))
register_candidates(HirClamp, (TirClamp,))
register_candidates(HirReLU, (TirReLU,))
register_candidates(HirCast, (TirCast,))
register_candidates(HirReduce, (TirReduce,))
register_candidates(HirWhere, (TirWhere,))
