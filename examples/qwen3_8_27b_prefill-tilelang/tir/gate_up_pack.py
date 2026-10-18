# analysis target=nvidia.h200_sxm module=GateUp function=pack topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r5.00MB/w345.00MB@logical,r660.00MB/w44.47GB@total,r5.00MB/w345.00MB@cta,r5.00MB/w345.00MB@thread;rmem:r224B/w0@logical,r28.88KB/w0@total,r224B/w0@cta,r224B/w0@thread footprint=v10:246:165.00MB;v5:245:165.00MB;w_gate:165.00MB;w_up:165.00MB footprint-precision=exact peak=gmem:680.00MB;rmem:0 persistent=gmem:340.00MB
#   error="l2 working set 660.00MB at the first iteration of a 132-unit wave exceeds capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def pack(
    w_gate: ConstTensor[(17408, 5120), "bf16"], w_up: ConstTensor[(17408, 5120), "bf16"], out: Tensor[(34816, 5120), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(384,), names=("t",)) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(34816, 5120), shape=(34816, 5120))
            T.fill(value, 0.0)
            for j in range(cta.persistent, 136, 132):
                value_1 = T.tensor_view(
                    T.ptr_of(w_gate), layout=(17408, 5120), shape=(17408, 5120)
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value_1[j * 128:j * 128 + 1, 0:0 + 1]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(out), layout=(34816, 5120), shape=(34816, 5120)
                )
                value_4 = T.tensor_view(
                    T.ptr_of(value_3), layout=(34816, 5120), shape=(34816, 5120)
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_4), layout=(34816, 5120), shape=(34816, 5120)
                )
                window = T.tensor_view(
                    T.ptr_of(value_5[j * 256:j * 256 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(value_2, window)
                value_6 = T.tensor_view(
                    T.ptr_of(w_up), layout=(17408, 5120), shape=(17408, 5120)
                )
                value_7 = T.tensor_view(
                    T.ptr_of(value_6[j * 128:j * 128 + 1, 0:0 + 1]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                value_8 = T.tensor_view(
                    T.ptr_of(value_5), layout=(34816, 5120), shape=(34816, 5120)
                )
                window_1 = T.tensor_view(
                    T.ptr_of(value_8[(j * 256) + 128:(j * 256) + 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(value_7, window_1)
