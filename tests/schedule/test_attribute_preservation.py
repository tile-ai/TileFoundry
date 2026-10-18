"""Lowering must retain authored choices that differ from instruction defaults."""

import torch

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.ir.tir.arange import Arange
from tilefoundry.ir.visitor import StmtVisitor, walk_prim_function
from tilefoundry.schedule import finalize
from tilefoundry.target import CudaTarget


@module(
    entry="run",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class NondefaultRange:
    @func
    def run():
        with Mesh(("thread",), layout=(1,), names=("lane",)) as _thread:
            output = tf.empty(Tensor[(4,), "i64", None, "rmem"])
            return tf.schedule((output,), op=T.arange(start=5, step=2))


def test_nondefault_instruction_attributes_survive_lowering():
    torch.testing.assert_close(NondefaultRange(), torch.tensor([5, 7, 9, 11]))

    class Find(StmtVisitor):
        def __init__(self):
            self.operations = []

        def visit_Evaluate(self, statement):
            if isinstance(statement.callable, Arange):
                self.operations.append(statement.callable)

    visitor = Find()
    lowered = finalize(NondefaultRange)
    walk_prim_function(visitor, lowered.entry_function())
    assert [(op.start, op.step) for op in visitor.operations] == [(5, 2)]
