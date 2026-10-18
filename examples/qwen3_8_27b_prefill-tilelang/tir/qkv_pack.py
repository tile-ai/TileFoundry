# analysis target=nvidia.h200_sxm module=QKV function=pack topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r3.75MB/w143.75MB@logical,r495.00MB/w18.53GB@total,r3.75MB/w143.75MB@cta,r3.75MB/w143.75MB@thread;rmem:r272B/w0@logical,r35.06KB/w0@total,r272B/w0@cta,r272B/w0@thread footprint=v12:474:10.00MB;v21:475:10.00MB;v5:472:120.00MB;w_k:10.00MB;w_q:120.00MB;w_v:10.00MB footprint-precision=exact peak=gmem:280.00MB;rmem:0 persistent=gmem:140.00MB
#   error="l2 working set 280.00MB at the first iteration of a 132-unit wave exceeds capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def pack(
    w_q: ConstTensor[(12288, 5120), "bf16"], w_k: ConstTensor[(1024, 5120), "bf16"], w_v: ConstTensor[(1024, 5120), "bf16"], out: Tensor[(14336, 5120), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(384,), names=("t",)) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(14336, 5120), shape=(14336, 5120))
            T.fill(value, 0.0)
            for j in range(cta.persistent, 96, 132):
                value_1 = T.tensor_view(
                    T.ptr_of(w_q), layout=(12288, 5120), shape=(12288, 5120)
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value_1[j * 128:j * 128 + 1, 0:0 + 1]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(out), layout=(14336, 5120), shape=(14336, 5120)
                )
                value_4 = T.tensor_view(
                    T.ptr_of(value_3), layout=(14336, 5120), shape=(14336, 5120)
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_4), layout=(14336, 5120), shape=(14336, 5120)
                )
                window = T.tensor_view(
                    T.ptr_of(value_5[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(value_2, window)
            for j_2 in range(cta.persistent, 8, 132):
                value_6 = T.tensor_view(T.ptr_of(w_k), layout=(1024, 5120), shape=(1024, 5120))
                value_7 = T.tensor_view(
                    T.ptr_of(value_6[j_2 * 128:j_2 * 128 + 1, 0:0 + 1]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                value_8 = T.tensor_view(
                    T.ptr_of(out), layout=(14336, 5120), shape=(14336, 5120)
                )
                value_9 = T.tensor_view(
                    T.ptr_of(value_8), layout=(14336, 5120), shape=(14336, 5120)
                )
                value_10 = T.tensor_view(
                    T.ptr_of(value_9), layout=(14336, 5120), shape=(14336, 5120)
                )
                value_11 = T.tensor_view(
                    T.ptr_of(value_10), layout=(14336, 5120), shape=(14336, 5120)
                )
                value_12 = T.tensor_view(
                    T.ptr_of(value_11), layout=(14336, 5120), shape=(14336, 5120)
                )
                window_1 = T.tensor_view(
                    T.ptr_of(value_12[((24 * 2) * 256) + (j_2 * 128):((24 * 2) * 256) + (j_2 * 128) + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(value_7, window_1)
                value_13 = T.tensor_view(T.ptr_of(w_v), layout=(1024, 5120), shape=(1024, 5120))
                value_14 = T.tensor_view(
                    T.ptr_of(value_13[j_2 * 128:j_2 * 128 + 1, 0:0 + 1]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                value_15 = T.tensor_view(
                    T.ptr_of(value_12), layout=(14336, 5120), shape=(14336, 5120)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_15[(((24 * 2) * 256) + (4 * 256)) + (j_2 * 128):(((24 * 2) * 256) + (4 * 256)) + (j_2 * 128) + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(value_14, window_2)
