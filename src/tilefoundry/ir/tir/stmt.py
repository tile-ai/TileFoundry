"""Backwards-compat shim: ``Stmt`` base now lives in ``tilefoundry.ir.core.stmt``.

Backwards-compat shim: ``Stmt`` base now lives in
``tilefoundry.ir.core.stmt``.

Existing imports ``from tilefoundry.ir.tir.stmt import Stmt`` keep
working. New code should import from ``tilefoundry.ir.core.stmt``.

The relocation breaks the historical
``tilefoundry.visitor_registry.contexts`` ↔ ``tilefoundry.ir.tir`` import
cycle that the ``Stmt`` base used to anchor.
"""

from __future__ import annotations

from tilefoundry.ir.core.stmt import Stmt

__all__ = ["Stmt"]
