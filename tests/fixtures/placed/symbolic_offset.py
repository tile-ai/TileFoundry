"""Literal and symbolic insert-slice destination offsets and their result."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, DimVar, Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare tile in authored bodies
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_HIDDEN = 2048
_GRID = 128
_HEAD = 128
_HEADS = 16
_STRIP = 32
_MESH_TILE = 4
_ROW_TILE = 256
_GROUPS = 3
_OUTPUTS = _GROUPS * _STRIP
_SEQ = DimVar("seq_len", 0, 8193)
_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", _GRID)


@module(entry="project", target=_H200, topologies=(_CTA,))
class _LiteralStoreOffset:
    """The control whose destination window starts at a literal zero."""

    @func
    def project(
        x: Tensor[(_HEADS, _SEQ, _HEAD), "bf16"],
        weight: ConstTensor[(_OUTPUTS, _HIDDEN, _HEAD), "bf16"],
        out: Tensor[(_OUTPUTS, _SEQ, _HEAD), "bf16"],
    ) -> Tensor[(_OUTPUTS, _SEQ, _HEAD), "bf16"]:
        with Mesh(
            ("cta",), layout=(_STRIP, _MESH_TILE), names=("strip", "tile")
        ) as mesh:
            result = out
            for position in tile(_SEQ, _ROW_TILE):
                base = position + 0
                for group in range(_GROUPS):
                    rows = tf.reshard(
                        x[0:1, base : base + _ROW_TILE, :],
                        (1, _ROW_TILE @ mesh.tile, _HEAD),
                        "smem",
                    )
                    weights = tf.reshard(
                        weight[0 : 0 + _STRIP, 0:_HEAD, :],
                        (_STRIP @ mesh.strip, _HEAD, _HEAD),
                        "smem",
                    )
                    product = tf.matmul(rows, weights)
                    for head in range(1, _HEADS, 1):
                        head_index = head + 0
                        weight_base = head_index * _HEAD
                        next_rows = tf.reshard(
                            x[head_index : head_index + 1, base : base + _ROW_TILE, :],
                            (1, _ROW_TILE @ mesh.tile, _HEAD),
                            "smem",
                        )
                        next_weights = tf.reshard(
                            weight[
                                0 : 0 + _STRIP,
                                weight_base : weight_base + _HEAD,
                                :,
                            ],
                            (_STRIP @ mesh.strip, _HEAD, _HEAD),
                            "smem",
                        )
                        product = product + tf.matmul(next_rows, next_weights)
                    result = tf.insert_slice(result, product, (0, base, 0))
            return result


@module(entry="project", target=_H200, topologies=(_CTA,))
class _SymbolicStoreOffset:
    """The same program with group_index * 32 as its destination offset."""

    @func
    def project(
        x: Tensor[(_HEADS, _SEQ, _HEAD), "bf16"],
        weight: ConstTensor[(_OUTPUTS, _HIDDEN, _HEAD), "bf16"],
        out: Tensor[(_OUTPUTS, _SEQ, _HEAD), "bf16"],
    ) -> Tensor[(_OUTPUTS, _SEQ, _HEAD), "bf16"]:
        with Mesh(
            ("cta",), layout=(_STRIP, _MESH_TILE), names=("strip", "tile")
        ) as mesh:
            result = out
            for position in tile(_SEQ, _ROW_TILE):
                base = position + 0
                for group in range(_GROUPS):
                    group_index = group + 0
                    store_base = group_index * _STRIP
                    rows = tf.reshard(
                        x[0:1, base : base + _ROW_TILE, :],
                        (1, _ROW_TILE @ mesh.tile, _HEAD),
                        "smem",
                    )
                    weights = tf.reshard(
                        weight[0 : 0 + _STRIP, 0:_HEAD, :],
                        (_STRIP @ mesh.strip, _HEAD, _HEAD),
                        "smem",
                    )
                    product = tf.matmul(rows, weights)
                    for head in range(1, _HEADS, 1):
                        head_index = head + 0
                        weight_base = head_index * _HEAD
                        next_rows = tf.reshard(
                            x[head_index : head_index + 1, base : base + _ROW_TILE, :],
                            (1, _ROW_TILE @ mesh.tile, _HEAD),
                            "smem",
                        )
                        next_weights = tf.reshard(
                            weight[
                                0 : 0 + _STRIP,
                                weight_base : weight_base + _HEAD,
                                :,
                            ],
                            (_STRIP @ mesh.strip, _HEAD, _HEAD),
                            "smem",
                        )
                        product = product + tf.matmul(next_rows, next_weights)
                    result = tf.insert_slice(result, product, (store_base, base, 0))
            return result
