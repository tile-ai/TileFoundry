from __future__ import annotations

from tilefoundry.dsl import *


@func
def ins(dst: Tensor[(2, 5, 3), "f32"], upd: Tensor[(2, 1, 3), "f32"]):
    res = tf.insert_slice(dst, upd, (0, 1, 0))
    return res
