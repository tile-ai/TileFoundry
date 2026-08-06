"""``tilefoundry.jit()``: what a caller can observe about it.

Two things, and the rest is gone for stated reasons. What compiling really does is
witnessed by `tests/e2e/`, which builds kernels with nvcc and runs them; repeating a
compile here bought a second compile and no second fact, and it was one of the
slowest cases in the suite. The cache key's structure went with it -- the key is
private, and what a caller can observe is that a second `jit` of the same program
hands back the same module and that clearing makes it stop doing so.

What is left is the part no end-to-end test can localise: which inputs are refused,
and that the cache is a cache rather than something that merely returns an answer.
"""


import pytest

from tests.fixtures.demo_ir import build_demo
from tilefoundry import jit
from tilefoundry.target import CudaTarget


def test_jit_rejects_non_ir_inputs_and_string_targets() -> None:
    """Raw Python, authored string Targets, and unexpected kwargs all raise."""

    def raw_fn(a):
        return a

    with pytest.raises(TypeError, match="expected Function or Module"):
        jit(raw_fn, target=CudaTarget("nvidia.h200_sxm"))

    fn, _, _ = build_demo()
    with pytest.raises(TypeError, match="Target instance, not a string"):
        jit(fn, target="vulkan")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        jit(fn, target=CudaTarget("nvidia.h200_sxm"), foo=1)

def test_jit_caches_and_clears() -> None:
    """Same IR → same RuntimeModule; ``cache_clear()`` evicts."""
    jit.cache_clear()
    fn, _, _ = build_demo()
    rt1 = jit(fn, target=CudaTarget("nvidia.h200_sxm"))
    fn2, _, _ = build_demo()
    assert jit(fn2, target=CudaTarget("nvidia.h200_sxm")) is rt1
    jit.cache_clear()
    assert jit(fn, target=CudaTarget("nvidia.h200_sxm")) is not rt1
