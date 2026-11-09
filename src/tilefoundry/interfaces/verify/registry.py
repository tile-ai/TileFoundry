"""Verification rules keyed by the Op or Stmt class they check."""

from __future__ import annotations

from tilefoundry import dispatch

__all__ = ["register", "registry"]

registry = dispatch.Registry("verify_stmt")
register = registry.decorator()
