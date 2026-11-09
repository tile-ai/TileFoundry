"""CUDA analyses: the ``sync`` family and the completion registry it reads."""

from .completion import Group
from .emission import (
    copies_carry_backedge,
    copies_carry_entry,
    copies_insert_slice_base,
    copies_insert_slice_update,
    issue_mesh,
    operand_roles,
    required_execution_mesh,
    result_param,
    tensor_map_load,
)
from .sync import StepKind, SyncMetadata, SyncPlanMetadata, analyze_sync
