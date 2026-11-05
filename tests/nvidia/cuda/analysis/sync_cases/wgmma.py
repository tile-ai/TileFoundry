"""Sync cases for WGMMA groups: where they are fenced, committed and waited on."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.nvidia.cuda.analysis.sync_cases import (
    FIXTURES,
    SyncCase,
    _analyzed,
    _attached,
    _planned,
)
from tilefoundry.analysis import AnalysisError
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.schedule import finalize
from tilefoundry_nvidia.cuda.analysis.completion import Group
from tilefoundry_nvidia.cuda.analysis.sync import StepKind
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.sync.wgmma import WgmmaCommitGroup, WgmmaWaitGroup

_PUBLISHED = ((StepKind.PROXY_FENCE, None), (StepKind.BARRIER, None))
_WAITED = ((StepKind.COMMIT, Group.WGMMA), (StepKind.WAIT, Group.WGMMA))
_FENCE = ((StepKind.FENCE, Group.WGMMA),)


def _steps(attached):
    """Each node's instruction, None for a region or a retained call, with each tag's steps."""
    return [
        (
            type(owner.target.op).__name__
            if isinstance(getattr(owner, "target", None), ScheduleOp)
            else None,
            tag,
            tuple((step.kind, step.group) for step in steps),
        )
        for owner, tagged in attached
        for tag, steps in tagged
    ]


@case
class WgmmaRetainedCall(SyncCase):
    """A retained call reading the accumulator is an access: the group is waited on before it."""

    path: Path = FIXTURES / "analysis/sync_wgmma_call.py"

    def guard(self, state, problems):
        _function, attached = _planned(state)
        assert _steps(attached) == [
            ("TiledMma", None, _PUBLISHED + _FENCE),
            (None, None, _WAITED),
        ]
        owner, _tagged = attached[1]
        assert isinstance(owner, Call), "the waits sit before a call, not before a region"


@case
class WgmmaPartialWarpgroups(SyncCase):
    """The wait for a WGMMA group cannot run over a loader warp beside one warpgroup.

    So the WGMMA completes right after it issues, in its warpgroup.
    """

    path: Path = FIXTURES / "analysis/sync_wgmma_partial.py"

    def guard(self, state, problems):
        result = _analyzed(state)
        attached = _attached(result.function)
        assert _steps(attached) == [
            (None, None, ((StepKind.BARRIER, None),)),
            ("TiledMma", None, _FENCE),
            ("TiledMma", "after", _WAITED),
            (
                None,
                None,
                ((StepKind.COMMIT, Group.CP_ASYNC), (StepKind.WAIT, Group.CP_ASYNC)) + _PUBLISHED,
            ),
        ]

        issuing = []

        class IssuingScopes(StmtVisitor):
            def visit_MeshScope(self, stmt):
                if isinstance(stmt.body, Sequential):
                    kinds = [type(s.callable) for s in stmt.body.body if isinstance(s, Evaluate)]
                    if TiledMma in kinds:
                        issuing.append(kinds)
                self.generic_visit(stmt)

        IssuingScopes().visit(finalize(result).lookup(result.function.name).body)
        (kinds,) = issuing
        last = max(index for index, kind in enumerate(kinds) if kind is TiledMma)
        assert kinds[last + 1 : last + 3] == [WgmmaCommitGroup, WgmmaWaitGroup]


@case
class WgmmaDivergentLoop(SyncCase):
    """A WGMMA in a loop bounded by the warp cannot be issued by the whole warpgroup together."""

    path: Path = FIXTURES / "analysis/sync_wgmma_divergent.py"

    def guard(self, state, problems):
        with pytest.raises(AnalysisError, match="collective step is needed inside loop '_j'"):
            _planned(state)


@case
class WgmmaGroupLoop(SyncCase):
    """A loop bounded by the warpgroup runs as often for each of its threads.

    So each warpgroup issues its chained WGMMAs together, and the loop holds no step.
    """

    path: Path = FIXTURES / "analysis/sync_wgmma_group_loop.py"

    def guard(self, state, problems):
        _function, attached = _planned(state)
        assert _steps(attached) == [
            ("TiledMma", None, _PUBLISHED + _FENCE),
            (None, "update", _WAITED),
        ]


@case
class WgmmaInFlightAtReturn(SyncCase):
    """A callee's WGMMA would still write the accumulator it returns, so it completes in place.

    Calls complete on return, so the caller needs no step.
    """

    path: Path = FIXTURES / "analysis/sync_wgmma_return.py"

    def guard(self, state, problems):
        functions = {function.name: function for function in _analyzed(state).functions}
        assert _steps(_attached(functions["gemm"])) == []
        assert _steps(_attached(functions["issue"])) == [
            ("TiledMma", None, _PUBLISHED + _FENCE),
            ("TiledMma", "after", _WAITED),
        ]


@case
class WgmmaShapes(SyncCase):
    """Equal shapes chain without steps; switching BF16 and E4M3 waits and fences again."""

    path: Path = FIXTURES / "analysis/sync_wgmma_shapes.py"

    def guard(self, state, problems):
        _function, attached = _planned(state)
        assert _steps(attached) == [
            ("TiledMma", None, _PUBLISHED + _FENCE),
            ("TiledMma", None, _WAITED + _PUBLISHED + _FENCE),
            ("TiledMma", None, _WAITED + _FENCE),
            (None, "update", _WAITED),
        ]
