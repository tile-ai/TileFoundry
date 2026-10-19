"""Qwen HIR with explicit AMX projection layouts and worker schedules."""

from dataclasses import replace
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, ConstTensor, Mesh, T, Topology, tf
from tilefoundry.ir.types import Layout, DType
from tilefoundry.ir.tir.amx.instructions import Z_LAYOUT
from model import build

BM = 32
WORKERS = 8


def scheduled_projection(M, N, K):
    if K <= 0 or N <= 0 or K % 32 or N % 32:
        raise ValueError(
            "The native AMX runtime requires positive K and N that are multiples of 32"
        )
    B = 64 if N % 64 == 0 else 32
    if M != 1 and (M <= 0 or M % 32):
        raise ValueError(
            "This schedule requires a multiple of 32 prompt tokens, or M=1 for the last-token head"
        )
    if M == 1:

        @module(entry="forward", topologies=(Topology("core", 8), Topology("amx", 1)))
        class Head:
            @func
            def forward(x: Tensor[(1, K), "f32"], w: ConstTensor[(K, N), "f16"]):
                return tf.matmul(tf.cast(x, dtype="f16"), w, out_dtype="f32")

            @forward.converter("w")
            def _(raw: ConstTensor[(N, K), "f16"]):
                return tf.transpose(raw, perm=(1, 0))

        return Head

    @module(entry="forward", topologies=(Topology("core", 8), Topology("amx", 1)))
    class Linear:
        @func
        def forward(x: Tensor[(M, K), "f32"], w: ConstTensor[(N // B, K, B), "f16"]):
            with Mesh(("core",), layout=(1,), names=("packer",)) as _packer:
                # One CPU pack, outside all K issues. The physical 32-row
                # panel makes each AMX Y load a contiguous 64-byte vector.
                half = tf.cast(x, dtype="f16")
                view = tf.transpose(
                    tf.reshape(half, new_shape=(M // BM, BM, K)), perm=(0, 2, 1)
                )
                packed = tf.schedule(
                    (
                        view,
                        tf.empty(
                            Tensor[
                                (M // BM, K, BM),
                                "f16",
                                Layout((M // BM, K, BM), (K * BM, BM, 1)),
                                "gmem",
                            ]
                        ),
                    ),
                    op=T.copy(),
                )
                out = tf.zeros(Tensor[(M, N), "f32"])
            with Mesh(("core",), layout=(WORKERS,), names=("worker",)) as cores:
                for panel in range(cores.worker, N // B, WORKERS):
                    for half_n in range(B // 32):
                        for mb in range(M // BM):
                            with Mesh(("amx",), layout=(1,), names=("unit",)) as _amx:
                                acc = tf.zeros(
                                    Tensor[(BM, 32), "f32", Z_LAYOUT, "rmem"]
                                )
                                for k in range(K):
                                    a = tf.schedule(
                                        (
                                            tf.reshape(
                                                packed[mb : mb + 1, k : k + 1, :],
                                                new_shape=(BM, 1),
                                            ),
                                            tf.empty(
                                                Tensor[(BM, 1), "f16", None, "rmem"]
                                            ),
                                        ),
                                        op=T.amx_load_y(),
                                    )
                                    b = tf.schedule(
                                        (
                                            tf.reshape(
                                                w[
                                                    panel : panel + 1,
                                                    k : k + 1,
                                                    half_n * 32 : half_n * 32 + 32,
                                                ],
                                                new_shape=(1, 32),
                                            ),
                                            tf.empty(
                                                Tensor[(1, 32), "f16", None, "rmem"]
                                            ),
                                        ),
                                        op=T.amx_load_x(),
                                    )
                                    acc = tf.schedule(
                                        (acc, a, b),
                                        op=T.amx_fma16(),
                                        repeat=(1, 1, 1),
                                        order=(0, 1, 2),
                                    )
                                tile = tf.schedule(
                                    (
                                        acc,
                                        tf.empty(Tensor[(BM, 32), "f32", None, "gmem"]),
                                    ),
                                    op=T.amx_store(),
                                )
                                out = tf.insert_slice(
                                    out, tile, (mb * BM, panel * B + half_n * 32)
                                )
                return out

        @forward.converter("w")
        def _(raw: ConstTensor[(N, K), "f16"]):
            return tf.transpose(
                tf.reshape(raw, new_shape=(N // B, B, K)), perm=(0, 2, 1)
            )

    return Linear


def make_scheduled(config, sequence=128):
    return replace(
        build(config, sequence, linear=scheduled_projection),
        topologies=(Topology("core", 8), Topology("amx", 1)),
    )
