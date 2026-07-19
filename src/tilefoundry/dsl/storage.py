"""DSL storage constants.

User-facing short names for :class:`tilefoundry.ir.types.storage.StorageKind`,
imported directly (``from tilefoundry.dsl.storage import gmem, smem, ...``) rather
than hung under the auto-generated ``T`` op namespace.
"""
from __future__ import annotations

from tilefoundry.ir.types.storage import StorageKind

host = StorageKind.HOST
gmem = StorageKind.GMEM
smem = StorageKind.SMEM
rmem = StorageKind.RMEM
tmem = StorageKind.TMEM

__all__ = ["host", "gmem", "smem", "rmem", "tmem"]
