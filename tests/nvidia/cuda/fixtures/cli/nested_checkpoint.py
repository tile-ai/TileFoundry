from dataclasses import replace

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.models.qwen3_5_35b_a3b.model import (
    _DT,
    _H,
    Qwen3_5MoE,
    Qwen3_5Router,
    S,
)


@module(
    entry="routing",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 512)),
)
class Wrapper:
    router = Qwen3_5Router

    @func
    def routing(tokens: Tensor[(S, _H), _DT]):
        return router(tokens)


Root = replace(Qwen3_5MoE, functions=(*Qwen3_5MoE.functions, Wrapper.routing))
