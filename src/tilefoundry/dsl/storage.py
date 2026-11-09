"""DSL storage constants.

User-facing short names for :class:`tilefoundry.ir.expr.StorageKind`,
imported directly (``from tilefoundry.dsl.storage import gmem, smem, ...``) rather
than hung under the auto-generated ``T`` op namespace.
"""

from __future__ import annotations

from tilefoundry.ir import expr

host = expr.StorageKind.HOST
gmem = expr.StorageKind.GMEM
smem = expr.StorageKind.SMEM
rmem = expr.StorageKind.RMEM
tmem = expr.StorageKind.TMEM
umat = expr.StorageKind.UMAT

__all__ = ["host", "gmem", "smem", "rmem", "tmem", "umat"]
