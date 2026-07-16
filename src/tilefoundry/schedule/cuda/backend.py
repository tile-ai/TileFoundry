from __future__ import annotations

from .cost import CudaCtaCostModel
from .materialize import materialize_cuda_schedule
from .solver import CudaCtaSolver
from .space import build_cuda_space


class CudaCtaBackend:
    def build_space(self, graph, context):
        return build_cuda_space(graph, context)

    def cost_model(self, context):
        return CudaCtaCostModel()

    def solver(self, context):
        return CudaCtaSolver()

    def materialize(self, problem, solution, context):
        return materialize_cuda_schedule(problem, solution, context)


__all__ = ["CudaCtaBackend"]
