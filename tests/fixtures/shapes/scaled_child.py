"""A child Module holding one weight of its own, so a caller passes activations alone.

Three directories reach it from a different side: ``tests/analysis`` checks that
two attachments keep their resources distinct, ``tests/evaluator`` runs it against
two readings, and ``tests/runtime`` checks what a reading supplies and from where.
The weight parameter is named ``w`` because callers address it as ``<binding>.w``.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf


@module(entry="run")
class ScaledChild:
    @func
    def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.mul(x, w)
