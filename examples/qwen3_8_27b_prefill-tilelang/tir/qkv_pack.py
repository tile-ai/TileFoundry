# analysis target=nvidia.h200_sxm module=QKV function=pack topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r3.75MB/w143.75MB@logical,r495.00MB/w18.53GB@total,r3.75MB/w143.75MB@cta,r3.75MB/w143.75MB@thread;rmem:r272B/w0@logical,r35.06KB/w0@total,r272B/w0@cta,r272B/w0@thread footprint=v12:455:10.00MB;v21:456:10.00MB;v5:453:120.00MB;w_k:10.00MB;w_q:120.00MB;w_v:10.00MB footprint-precision=exact peak=gmem:280.00MB;rmem:0 persistent=gmem:140.00MB
#   error="l2 working set 280.00MB at the first iteration of a 132-unit wave exceeds capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def pack(
    w_q: Tensor[(12288, 5120), "bf16"], w_k: Tensor[(1024, 5120), "bf16"], w_v: Tensor[(1024, 5120), "bf16"], out: Tensor[(14336, 5120), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        with Mesh((Topology("thread", 384),), layout=(384,), names=("d0",)) as scope:
            for j in range(cta.d0, 96, 132):
                tile = T.tensor_view(
                    T.ptr_of(w_q[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                window = T.tensor_view(
                    T.ptr_of(out[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(tile, window)
            for j in range(cta.d0, 8, 132):
                tile_1 = T.tensor_view(
                    T.ptr_of(w_k[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                window_1 = T.tensor_view(
                    T.ptr_of(out[((24 * 2) * 256) + (j * 128):((24 * 2) * 256) + (j * 128) + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(tile_1, window_1)
                tile_2 = T.tensor_view(
                    T.ptr_of(w_v[j * 128:j * 128 + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                window_2 = T.tensor_view(
                    T.ptr_of(out[(((24 * 2) * 256) + (4 * 256)) + (j * 128):(((24 * 2) * 256) + (4 * 256)) + (j * 128) + 128, 0:0 + 5120]),
                    layout=(128, 5120),
                    shape=(128, 5120),
                )
                T.copy(tile_2, window_2)
