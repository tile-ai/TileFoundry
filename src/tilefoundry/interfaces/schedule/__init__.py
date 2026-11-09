"""Schedule interface: HIR-to-TIR candidates and instruction checks."""

from . import instruction
from .candidates import (
    HirMatch,
    Pairing,
    candidate_bindings,
    candidate_operands,
    candidate_ops,
    instruction_from_hir,
    pairings,
    register_candidates,
    sole_candidate,
)
from .instruction import (
    Projection,
    RefusalError,
    admit_fields,
    call_params,
    check_call,
    operand_relations,
    project,
)
