"""Pin caller-visible ``tilefoundry.jit`` behavior.

End-to-end tests cover compilation. These tests localize rejected inputs and the
observable cache contract: repeated JIT of one program returns the same module,
while clearing the cache ends that identity. Private key structure is not part of
the contract.
"""

import pytest

from tests.fixtures.placed.rmsnorm import RmsnormModule
from tilefoundry import jit
from tilefoundry.target import CudaTarget


def test_jit_rejects_non_ir_inputs_and_string_targets() -> None:
    """Raw Python, authored string Targets, and unexpected kwargs all raise."""

    def raw_fn(a):
        return a

    with pytest.raises(TypeError, match="expected Function or Module"):
        jit(raw_fn, target=CudaTarget("nvidia.h200_sxm"))

    fn = RmsnormModule.entry_function()
    with pytest.raises(TypeError, match="Target instance, not a string"):
        jit(fn, target="vulkan")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        jit(fn, target=CudaTarget("nvidia.h200_sxm"), foo=1)


