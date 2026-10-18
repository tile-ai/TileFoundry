# analysis target=nvidia.h200_sxm module=InProj function=pack topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r3.44MB/w165.94MB@logical,r453.75MB/w21.39GB@total,r3.44MB/w165.94MB@cta,r3.44MB/w165.94MB@thread;rmem:r144B/w0@logical,r18.56KB/w0@total,r144B/w0@cta,r144B/w0@thread footprint=v10:395:60.00MB;v11:397:480.00KB;v12:398:480.00KB;v5:393:100.00MB;w_a:480.00KB;w_b:480.00KB;w_qkv:100.00MB;w_z:60.00MB footprint-precision=exact peak=gmem:323.44MB;rmem:0 persistent=gmem:160.94MB
#   buffer=w_b holds=321.88MB time=none space=cta.persistent reuse=61.41MB fits=no precision=exact
#   buffer=w_a holds=321.88MB time=none space=cta.persistent reuse=61.41MB fits=no precision=exact
#   error="l2 reuse window cta.persistent holds 321.88MB at a 132-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def pack(
    w_qkv: ConstTensor[(10240, 5120), "bf16"], w_z: ConstTensor[(6144, 5120), "bf16"], w_b: ConstTensor[(48, 5120), "bf16"], w_a: ConstTensor[(48, 5120), "bf16"], out: Tensor[(16640, 5120), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(384,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            for j in range(cta.d0, 80, 132):
                value = T.tensor_view(
                    T.ptr_of(w_qkv[j * 128:j * 128 + 1, 0:0 + 1]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                value_1 = T.tensor_view(
                    T.ptr_of(out), layout=(16640, 5120), shape=(16640, 5120)
                )
                window = T.tensor_view(
                    T.ptr_of(value_1[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(value, window)
            for j_2 in range(cta.d0, 48, 132):
                value_2 = T.tensor_view(
                    T.ptr_of(w_z[j_2 * 128:j_2 * 128 + 1, 0:0 + 1]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(out), layout=(16640, 5120), shape=(16640, 5120)
                )
                window_1 = T.tensor_view(
                    T.ptr_of(value_3[10240 + (j_2 * 128):10240 + (j_2 * 128) + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(value_2, window_1)
            for j_3 in range(cta.d0, 1, 132):
                value_4 = T.tensor_view(T.ptr_of(w_b), layout=(48, 5120), shape=(48, 5120))
                value_5 = T.tensor_view(
                    T.ptr_of(out), layout=(16640, 5120), shape=(16640, 5120)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_5[16384:16384 + 48, 0:0 + 5120]),
                    layout=(48, 5120),
                    shape=(48, 5120),
                )
                T.copy(value_4, window_2)
                value_6 = T.tensor_view(T.ptr_of(w_a), layout=(48, 5120), shape=(48, 5120))
                value_7 = T.tensor_view(
                    T.ptr_of(out), layout=(16640, 5120), shape=(16640, 5120)
                )
                window_3 = T.tensor_view(
                    T.ptr_of(value_7[16432:16432 + 48, 0:0 + 5120]),
                    layout=(48, 5120),
                    shape=(48, 5120),
                )
                T.copy(value_6, window_3)
