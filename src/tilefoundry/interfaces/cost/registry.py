"""Cost evaluators keyed by the Op class they price."""

from __future__ import annotations

from tilefoundry import dispatch

__all__ = ["register", "registry"]

registry = dispatch.Registry("cost_evaluator")
register = registry.decorator()
