"""``tilefoundry.jit()`` API — input validation + cache key + end-to-end (nvcc)."""

import hashlib

import pytest

from tests.fixtures.demo_canonical import build_demo_canonical
from tests.fixtures.demo_ir import build_demo
from tilefoundry import jit
from tilefoundry.compile import _canonical_module_text, _jit_cache_key_payload
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.runtime.module import RuntimeModule


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


def test_cache_key_is_deterministic_and_topology_target_sensitive() -> None:
    """Same IR → same payload; different topology / target → different payload."""


    fn = build_demo_canonical()
    t1, target1, opts1 = _jit_cache_key_payload(fn)
    t2, target2, opts2 = _jit_cache_key_payload(build_demo_canonical())
    assert (t1, target1) == (t2, target2)
    assert hashlib.sha256(f"{t1}\0{target1}\0{opts1}".encode()).hexdigest() \
        == hashlib.sha256(f"{t2}\0{target2}\0{opts2}".encode()).hexdigest()

    # Different topologies surface in canonical module text.
    fn_a, _, _ = build_demo()
    mod1 = Module(name=fn_a.name, functions=(fn_a,), entry=fn_a.name,
                  topologies=(Topology("cta", 128),))
    mod2 = Module(name=fn_a.name, functions=(fn_a,), entry=fn_a.name,
                  topologies=(Topology("cta", 64),))
    assert _canonical_module_text(mod1) != _canonical_module_text(mod2)

    # Different target → different options text.
    _, _, opts_cuda = _jit_cache_key_payload(fn_a, target="cuda")
    _, _, opts_hip = _jit_cache_key_payload(fn_a, target="hip")
    assert opts_cuda != opts_hip


# ── compile-backed end-to-end ───────────────────────────────────────────


def test_jit_compiles_function_to_runtime_module() -> None:
    """``jit(fn)`` → ``RuntimeModule`` (callable)."""
    fn, _, _ = build_demo()
    rt = jit(fn, target="cuda")
    assert isinstance(rt, RuntimeModule) and callable(rt)


def test_jit_caches_and_clears() -> None:
    """Same IR → same RuntimeModule; ``cache_clear()`` evicts."""
    jit.cache_clear()
    fn, _, _ = build_demo()
    rt1 = jit(fn, target="cuda")
    fn2, _, _ = build_demo()
    assert jit(fn2, target="cuda") is rt1
    jit.cache_clear()
    assert jit(fn, target="cuda") is not rt1
