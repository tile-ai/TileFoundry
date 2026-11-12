"""Valuation: the lexical environment and the visitor every analysis builds on."""

from .call_paths import collect_functions
from .context import Context, module_slot
from .isolation import verify_region_isolated
from .specialize import (
    BOUND_DIMS,
    DISPLAY_NAME,
    PROVENANCE,
    InstantiateDimensionCloner,
    SpecializationError,
    bound_dims_of,
    canonical_specialization_signature,
    dim_vars_reached,
    display_name,
    instantiate,
    origin_of,
    residual_dims,
    variant_for,
)
from .visitor import Visitor, definitions, index_contexts
