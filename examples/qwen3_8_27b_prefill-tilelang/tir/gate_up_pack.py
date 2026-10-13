# analysis target=nvidia.h200_sxm module=GateUp function=pack topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r5.00MB/w345.00MB@logical,r660.00MB/w44.47GB@total,r5.00MB/w345.00MB@cta,r5.00MB/w345.00MB@thread;rmem:r224B/w0@logical,r28.88KB/w0@total,r224B/w0@cta,r224B/w0@thread footprint=v10:228:165.00MB;v5:227:165.00MB;w_gate:165.00MB;w_up:165.00MB footprint-precision=exact peak=gmem:680.00MB;rmem:0 persistent=gmem:340.00MB
#   error="l2 working set 660.00MB at the first iteration of a 132-unit wave exceeds capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def pack(
    w_gate: Tensor[(17408, 5120), "bf16"], w_up: Tensor[(17408, 5120), "bf16"], out: Tensor[(34816, 5120), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        with Mesh((Topology("thread", 384),), layout=(384,), names=("d0",)) as scope:
            for j in range(cta.d0, 136, 132):
                tile = T.tensor_view(
                    T.ptr_of(w_gate[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                window = T.tensor_view(
                    T.ptr_of(out[j * 256:j * 256 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(tile, window)
                tile_1 = T.tensor_view(
                    T.ptr_of(w_up[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                window_1 = T.tensor_view(
                    T.ptr_of(out[(j * 256) + 128:(j * 256) + 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(tile_1, window_1)
