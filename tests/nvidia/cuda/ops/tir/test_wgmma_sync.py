"""The WGMMA group steps verify only on whole warpgroups; the proxy fence on any threads."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import *
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.ir.tir.sync.wgmma import (
    FenceProxyAsync,
    WgmmaCommitGroup,
    WgmmaFence,
    WgmmaWaitGroup,
)

_GROUP_STEPS = (WgmmaFence(), WgmmaCommitGroup(), WgmmaWaitGroup(n=0))
_CTAS = (None, 4)


def _threads(count: int, *, of: int | None = None, start: int = 0) -> Mesh:
    if of is None:
        return Mesh((Topology("thread", count),), Layout((count,), (1,)))
    return Mesh((Topology("thread", of),), ComposedLayout(None, start, Layout((count,), (1,))))


def _scope(mesh: Mesh, body) -> MeshScope:
    return MeshScope(mesh=mesh, binding=Var(name="scope", type=mesh), body=Sequential(body=body))


def _pf(op, threads: Mesh, ctas: int | None = None) -> PrimFunction:
    """*op* run by *threads*, inside a scope of *ctas* CTAs when given."""
    body = _scope(threads, (Evaluate(callable=op, args=()),))
    if ctas is not None:
        body = _scope(Mesh((Topology("cta", ctas),), Layout((ctas,), (1,))), (body,))
    return PrimFunction(
        target=CudaTarget("nvidia.h200_sxm"),
        name="fn",
        params=(),
        body=Sequential(body=(body, Return())),
    )


@pytest.mark.parametrize("ctas", _CTAS, ids=("one CTA", "four CTAs"))
@pytest.mark.parametrize("op", _GROUP_STEPS, ids=lambda op: type(op).__name__)
@pytest.mark.parametrize(
    "threads",
    (_threads(32), _threads(128, of=256, start=64)),
    ids=("one warp", "a warpgroup's worth from thread 64"),
)
def test_a_group_step_refuses_threads_that_are_not_whole_warpgroups(op, threads, ctas) -> None:
    with pytest.raises(VerifyError, match="needs whole warpgroups"):
        verify_prim_function(_pf(op, threads, ctas))


@pytest.mark.parametrize("ctas", _CTAS, ids=("one CTA", "four CTAs"))
@pytest.mark.parametrize("op", _GROUP_STEPS, ids=lambda op: type(op).__name__)
def test_a_group_step_runs_on_two_warpgroups(op, ctas) -> None:
    verify_prim_function(_pf(op, _threads(256), ctas))


def test_the_proxy_fence_runs_on_one_warp() -> None:
    verify_prim_function(_pf(FenceProxyAsync(), _threads(32)))
