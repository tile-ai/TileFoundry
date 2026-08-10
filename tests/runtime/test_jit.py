"""Pin caller-visible ``tilefoundry.jit`` behavior.

End-to-end tests cover compilation. These tests localize rejected inputs and the
observable cache contract: repeated JIT of one program returns the same module,
while clearing the cache ends that identity. Private key structure is not part of
the contract.
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
