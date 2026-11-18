"""Reverse-mode differentiation: rules registered per Op."""

from .context import Context, build, gradient_layout, scalar
from .registry import register, registry
