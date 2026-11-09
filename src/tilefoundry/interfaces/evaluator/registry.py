"""Reference-evaluation rules for HIR Ops and for the TIR Ops a schedule selects."""

from __future__ import annotations

from tilefoundry import dispatch

__all__ = ["register", "registry"]

registry = dispatch.Registry("eval")
register = registry.decorator()
