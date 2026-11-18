"""Reverse-mode rules keyed by the Op class they differentiate."""

from __future__ import annotations

from tilefoundry import dispatch

__all__ = ["register", "registry"]

registry = dispatch.Registry("autodiff_rule")
register = registry.decorator()
