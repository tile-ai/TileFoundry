"""Reverse-mode differentiation: rules registered per Op."""

from .context import Context, gradient_layout
from .registry import register, registry
