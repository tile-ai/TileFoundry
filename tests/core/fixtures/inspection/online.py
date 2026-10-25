from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="online", target=CpuTarget())
class Online:
    @func
    def online(x: Tensor[(4,), "f32"]):
        m = tf.zeros(Tensor[(4,), "f32"])
        o = tf.zeros(Tensor[(4,), "f32"])
        for i in range(4):
            m_new = tf.maximum(m, x)
            corr = tf.sub(m, m_new)
            o = tf.add(o, corr)
            m = m_new
        return (m, o)
