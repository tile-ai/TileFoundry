"""Shared shard-layout constructors for authored TIR fixtures."""

from __future__ import annotations

from tilefoundry.ir import types


def bcast(shape, strides, mesh) -> types.ShardLayout:
    """A tile held whole by every instance, so ``local()`` returns that tile."""
    return types.ShardLayout(
        layout=types.Layout(shape=shape, strides=strides),
        attrs=tuple(types.Broadcast() for _ in types.flatten(mesh.layout).shape),
        mesh=mesh,
    )


def split_rows(mesh) -> types.ShardLayout:
    return types.ShardLayout(types.Layout((128, 4), (4, 1)), (types.Split(0),), mesh)


def split_pairs(mesh) -> types.ShardLayout:
    return types.ShardLayout(types.Layout((128, 2), (2, 1)), (types.Split(0),), mesh)


def split_short_rows(mesh) -> types.ShardLayout:
    return types.ShardLayout(types.Layout((32, 4), (4, 1)), (types.Split(0),), mesh)


def broadcast_run(mesh) -> types.ShardLayout:
    return types.ShardLayout(types.Layout((4,), (1,)), (types.Broadcast(),), mesh)


__all__ = ["bcast", "broadcast_run", "split_pairs", "split_rows", "split_short_rows"]
