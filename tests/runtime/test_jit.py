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


def test_jit_rejects_non_ir_inputs_and_unknown_targets() -> None:
    """Raw Python / non-IR / unsupported target / unexpected kwarg all raise."""

    def raw_fn(a):
        return a

    with pytest.raises(TypeError, match="expected Function or Module"):
        jit(raw_fn, target="cuda")

    fn, _, _ = build_demo()
    with pytest.raises(ValueError, match="not supported"):
        jit(fn, target="vulkan")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        jit(fn, target="cuda", foo=1)

def test_jit_caches_and_clears() -> None:
    """Same IR → same RuntimeModule; ``cache_clear()`` evicts."""
    jit.cache_clear()
    fn, _, _ = build_demo()
    rt1 = jit(fn, target="cuda")
    fn2, _, _ = build_demo()
    assert jit(fn2, target="cuda") is rt1
    jit.cache_clear()
    assert jit(fn, target="cuda") is not rt1
