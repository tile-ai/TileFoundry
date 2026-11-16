"""The concrete transforms of the pass pipeline."""

from __future__ import annotations

from .convert_hir_to_tir import (
    BINDING,
    INDEX,
    ConvertHIRToTIR,
    Lowering,
    LoweringError,
    StoragePlan,
    TirBuilder,
    TirScope,
    View,
    accepted_aliases,
    copy_mesh,
    issue_owner,
    label,
    lowers_to_dim,
    operand_types,
    scratch_buffers,
    selected_op,
    transfer_meshes,
)
from .host_entry import InsertHostEntryPass
