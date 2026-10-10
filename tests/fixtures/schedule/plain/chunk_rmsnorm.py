"""Chunk-parallel gated RMS normalization over a symbolic sequence length.

The (head, chunk) mesh gives the head axis a symbolic stride (#201).
Scheduled scalar and lhs size-one broadcasts share a program with automatic
rank-one and scalar broadcasts (#200). This remains a plain scheduling fixture:
only two binaries select instructions; the other sites still need candidates.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

H = 4
BT = 64
D = 128
EPS = 1e-6
NC = DimVar("chunks", 1, 32)
T_LEN = NC * BT


@module(
    entry="chunk_rmsnorm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 128)),
)
class ChunkRmsNorm:
    @func
    def chunk_rmsnorm(
        x: Tensor[(H, T_LEN, D), "f32"],
        gate: Tensor[(H, T_LEN, D), "f32"],
        gamma: Tensor[(D,), "f32"],
        beta: Tensor[(D,), "f32"],
    ) -> Tensor[(H, T_LEN, D), "f32"]:
        with Mesh(("cta",), layout=(H, NC), names=("head", "chunk")) as cta:
            tile = tf.reshard(x, (H @ cta.head, T_LEN @ cta.chunk, D), "smem")
            gates = tf.reshard(gate, (H @ cta.head, T_LEN @ cta.chunk, D), "smem")
            with Mesh(("thread",), layout=(BT,), names=("row",)) as lanes:
                row = tf.reshard(tile, (H @ cta.head, T_LEN @ (cta.chunk, lanes.row), D), "rmem")
                gate_row = tf.reshard(
                    gates, (H @ cta.head, T_LEN @ (cta.chunk, lanes.row), D), "rmem"
                )
                scale = tf.reshard(gamma, ((D,), (1,), {}), "rmem")
                bias = tf.reshard(beta, ((D,), (1,), {}), "rmem")
                mean = tf.reduce(tf.square(row), axes=(-1,), keepdim=True, kind=ReduceKind.MEAN)
                shifted = tf.schedule((mean, EPS), op=T.binary(kind=BinaryKind.ADD))
                inv = tf.rsqrt(shifted)
                normed = tf.schedule((inv, row), op=T.binary(kind=BinaryKind.MUL))
                scaled = normed * scale
                affine = bias + scaled
                mixed = affine * (1.0 - gate_row)
                out = mixed * 0.5
            return tf.reshard(out, ((H, T_LEN, D), (D * T_LEN, D, 1), {}), "gmem")
