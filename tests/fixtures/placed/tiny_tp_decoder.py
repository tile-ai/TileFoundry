"""A two-card tensor-parallel decoder whose twin judges its own weight shards.

The projection weight is split by column across the ``gpu`` level and the
decode weight carries no distribution at all, so one checkpoint exercises both
answers a card can be owed. Every element of the projection weight is distinct,
so a twin that was handed the wrong column says which one it got.
"""

from __future__ import annotations

import torch

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf
from tilefoundry.ir.types.shard import Layout, Split, canonical_shard_layout
from tilefoundry.ir.types.shard import Mesh as ShardMesh
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CudaTarget

GPUS, R, C = 2, 4, 4
COLUMNS = C // GPUS
GPU, CTA = Topology("gpu", GPUS), Topology("cta", R)

_MESH = ShardMesh(topologies=(GPU,), layout=Layout((GPUS,), (1,)), names=("g",))
_BY_COLUMN = canonical_shard_layout((R, C), _MESH, (Split(1),))

PROJECT_FULL = torch.arange(R * C, dtype=torch.float32).reshape(R, C)
DECODE_FULL = torch.arange(100.0, 100.0 + R, dtype=torch.float32)
PROJECT_SHARDS = {
    gpu: PROJECT_FULL[:, gpu * COLUMNS : (gpu + 1) * COLUMNS] for gpu in range(GPUS)
}


@module(entry="project", topologies=(GPU, CTA))
class DecoderLayer:
    """One projection sharded by column and one replicated decode weight."""

    @func
    def project(
        x: Tensor[(R, C), "f32"],
        project_weight: ConstTensor[(R, C), "f32", _BY_COLUMN],
    ) -> Tensor[(R, C), "f32", _BY_COLUMN]:
        with Mesh(("gpu",), layout=(GPUS,), names=("g",)) as g:
            return tf.reshard(project_weight, (R, C @ g.g), "gmem")

    @func
    def decode(
        x: Tensor[(R,), "f32"],
        decode_weight: ConstTensor[(R,), "f32"],
    ) -> Tensor[(R,), "f32"]:
        with Mesh(("cta",), layout=(R,), names=("b",)) as cta:
            xs = tf.reshard(x, (R @ cta.b,), "rmem")
            ws = tf.reshard(decode_weight, (R @ cta.b,), "rmem")
            return tf.reshard(tf.mul(xs, ws), (R @ cta.b,), "gmem")


@module(target=CudaTarget("nvidia.h200_sxm"), topologies=(GPU, CTA))
class TinyTPDecoderLM:
    """The root the checkpoint is prepared from, reaching the layer as a child."""

    layer = DecoderLayer

    def forward(self, x, row, steps):
        """Run the decode loop *steps* times, as a host dispatch loop would."""
        for _ in range(steps):
            projected = self.layer.project(x)
            replicated = self.layer.decode(row)
        return projected, replicated


@runtime_module(DecoderLayer)
class DecoderLayerTwin:
    """The twin is the assertion.

    A body is where the runtime hands one program its data, so the judgement
    that it is this card's data and no other belongs right here rather than
    back out in a test that would have to rebuild the expectation.
    """

    @runtime_func
    def project(self, x, project_weight):
        """Hold the column block against the one this card's program id owes it."""
        want = PROJECT_SHARDS[torch.cuda.current_device()]
        assert tuple(project_weight.shape) == (R, COLUMNS), tuple(project_weight.shape)
        assert torch.equal(project_weight.cpu(), want), project_weight.cpu()
        return project_weight

    @runtime_func
    def decode(self, x, decode_weight):
        """A weight with no distribution is every card's whole tensor."""
        assert torch.equal(decode_weight.cpu(), DECODE_FULL), decode_weight.cpu()
        return x * decode_weight


@runtime_module(TinyTPDecoderLM)
class TinyTPDecoderLMTwin:
    layer = DecoderLayerTwin
