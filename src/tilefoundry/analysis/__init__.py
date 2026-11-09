"""Direct public Analysis fact contract."""

from . import allocation, footprint, reach, registry
from .access import (
    Access,
    AliasKind,
    BufferAlias,
    buffer_alias,
    eliminate_parameters,
    forwarded_leaves,
)
from .allocation import (
    AllocationSummaryMetadata,
    AllocationValue,
    aligned,
    covers_result,
    storage_bytes,
)
from .api import Result, analyze, build_context
from .check import check_program, program_dim_vars
from .compute_cost import inflight_bytes, local_duration_ns
from .errors import Error
from .iteration_scope import IterationScope, build_scopes
from .liveness import LivenessMetadata, result_copies
from .loop_domain import iteration_domain
from .memory import MemoryOptions, bytes_for, call_traffic, owned_bytes, reached_bytes
from .metadata import (
    ComputeCostMetadata,
    ExecutionMeshMetadata,
    MemoryLevelPeak,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    ReuseWindow,
    RooflineMetadata,
    TimelineMetadata,
    Traffic,
    ValueLifetime,
    shifted,
)
from .precision import Precision
from .quantity import Breakdown, Spread, shares
from .reach import Footprint, ReachedAddresses, footprint_of, merged
from .visitor import AnalyzeContext
