"""Compatibility exports for the dim <-> isl bridge owned by IR."""

from tilefoundry.ir.isl_interop import dim_range, isl_to_dim, shape_to_isl_domain

__all__ = ["dim_range", "isl_to_dim", "shape_to_isl_domain"]
