from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="ins", target=CpuTarget())
class Ins:
    @func
    def ins(dst: Tensor[(2, 5, 3), "f32"], upd: Tensor[(2, 1, 3), "f32"]):
        res = tf.insert_slice(dst, upd, (0, 1, 0))
        return res
