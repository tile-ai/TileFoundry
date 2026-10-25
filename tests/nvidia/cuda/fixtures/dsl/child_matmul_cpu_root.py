"""A CPU root that holds the same target-less children as the CUDA root.

Both roots come from one execution of this file, so a test can compare what
each root does to the very child objects they share.
"""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.logical.child_matmul_inherited_target import (
    K_LEN,
    ChildMatmul,
    ChildMatmulDirect,
    ChildMatmulRoot,  # noqa: F401 -- the CUDA root that holds the same children
    ChildMatmulStaged,
    K,
    M,
    N,
)
from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="gemm", target=CpuTarget(), topologies=(Topology("cta", 1),))
class ChildMatmulCpuRoot:
    child = ChildMatmul
    direct = ChildMatmulDirect
    staged = ChildMatmulStaged

    @func
    def gemm(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]):
        return child(a, b)  # noqa: F821 -- class-body binding

    @func
    def gemm_on_chip(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]):
        return direct(a, b)  # noqa: F821 -- class-body binding

    @func
    def gemm_staged(a: Tensor[(M, K_LEN), "bf16"], b: Tensor[(K_LEN, N), "bf16"]):
        return staged(a, b)  # noqa: F821 -- class-body binding
