"""Shared shard-layout constructors for authored TIR fixtures."""

from __future__ import annotations

from tilefoundry.ir.types.shard import Layout, ShardLayout, Split
from tilefoundry.ir.types.shard.shard_layout import Broadcast


def bcast(shape, strides, mesh) -> ShardLayout:
    """A tile held whole by every instance, so ``local()`` returns that tile."""
    return ShardLayout(
        layout=Layout(shape=shape, strides=strides),
        attrs=tuple(Broadcast() for _ in mesh.positions.shape),
        mesh=mesh,
    )


def split_rows(mesh) -> ShardLayout:
    return ShardLayout(Layout((128, 4), (4, 1)), (Split(0),), mesh)


def split_pairs(mesh) -> ShardLayout:
    return ShardLayout(Layout((128, 2), (2, 1)), (Split(0),), mesh)


def split_short_rows(mesh) -> ShardLayout:
    return ShardLayout(Layout((32, 4), (4, 1)), (Split(0),), mesh)


def broadcast_run(mesh) -> ShardLayout:
    return ShardLayout(Layout((4,), (1,)), (Broadcast(),), mesh)


__all__ = ["bcast", "broadcast_run", "split_pairs", "split_rows", "split_short_rows"]
