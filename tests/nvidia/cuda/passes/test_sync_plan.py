"""Where the CUDA lowering orders shared memory, what it refuses, and how it checks itself."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import tilefoundry.passes.transforms.convert_hir_to_tir as convert
import tilefoundry_nvidia.cuda.passes.transforms.lowering as cuda_lowering
from tests.nvidia.cuda.schedule._support import _RetainedStatements
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.sync import SyncPlanMetadata
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Var, attach_metadata, get_metadata
from tilefoundry.ir.tir import CpAsyncCommit, CpAsyncWait
from tilefoundry.ir.tir.arith import Binary
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.memory import Copy, Fill
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.passes.transforms import LoweringError
from tilefoundry.schedule import finalize
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.passes.transforms import CudaLowering

FIXTURES = Path(__file__).parents[1] / "fixtures"
GEMM = FIXTURES / "schedule/hir/sm89_e4m3_gemm.py"
SCALARS = FIXTURES / "schedule/hir/scalar_binary.py"


def _analyzed(module):
    return analyze(*check_program(module, module.entry_function()), analysis=("memory", "sync"))


def test_each_row_lands_before_the_column_reads():
    """A commit, a wait and a barrier between each thread's row copy and the column reads.

    The windows start at the thread's coordinate and the loop's induction variable, so
    lowering bounds their bytes by those ranges.
    """
    result = _analyzed(load_authored_ir(str(FIXTURES / "passes/sync_row_windows.py")))
    lowered = _RetainedStatements()
    lowered.visit(finalize(result).lookup(result.function.name).body)
    operations = lowered.operations
    start = next(i for i, op in enumerate(operations) if isinstance(op, CopyAsync))
    end = next(i for i, op in enumerate(operations) if i > start and isinstance(op, Copy))
    assert [type(op) for op in operations[start + 1 : end]] == [CpAsyncCommit, CpAsyncWait, Sync]


def _drop(lowering, stmt, builder, seen):
    return None


def _twice(lowering, stmt, builder, seen):
    return (stmt, stmt)


def _retyped(lowering, stmt, builder, seen):
    return (replace(stmt, callable=Copy()),)


def _reattributed(lowering, stmt, builder, seen):
    cta = lowering.context.function.body.mesh
    return (replace(stmt, callable=TiledMma(atom=stmt.callable.atom, execution_mesh=cta)),)


def _arg(index, value):
    """*stmt* with argument *index* replaced by what *value* builds from it."""

    def rewrite(lowering, stmt, builder, seen):
        args = list(stmt.args)
        args[index] = value(lowering, args[index], builder, seen)
        return (replace(stmt, args=tuple(args)),)

    return rewrite


def _unrecorded(lowering, arg, builder, seen):
    return Var("unrecorded", type=arg.type)


def _earlier(lowering, arg, builder, seen):
    return seen[0].args[0]


def _recorded(**fields):
    """A fresh value carrying the argument's record with *fields* changed."""

    def value(lowering, arg, builder, seen):
        changed = Var("changed", type=arg.type)
        builder.note(changed, replace(builder.record_of(arg), **fields))
        return changed

    return value


def _prepared_for(name):
    """A fresh scalar recorded as filled from the open action's argument for *name*."""

    def value(lowering, arg, builder, seen):
        role = next(role for role in lowering.current.child.roles if role.name == name)
        return _recorded(source=role.argument, param=name)(lowering, arg, builder, seen)

    return value


@pytest.mark.parametrize(
    ("path", "kind", "nth", "rewrite", "message"),
    [
        (GEMM, CopyAsync, 0, _drop, r"CopyAsync\(.*\) at \(\d+, 0\) issued 0 of 1 planned issue"),
        (GEMM, CopyAsync, 0, _twice, r"CopyAsync\(.*\) at \(\d+, 0\) issued 2 of 1 planned issue"),
        (GEMM, CopyAsync, 0, _retyped, r"Copy\(.*\) at \(\d+, 0\) is not the planned CopyAsync"),
        (GEMM, TiledMma, 0, _reattributed, r"TiledMma\(.*\) at \(\d+, 0\) is not the planned"),
        (GEMM, Copy, 0, _arg(0, _unrecorded), "passes shared memory the plan does not hold"),
        (GEMM, Copy, 1, _arg(0, _earlier), "passes shared memory the plan does not hold"),
        (
            GEMM, Copy, 0, _arg(0, _recorded(covered=(512, 528))),
            r"passes 'src' bytes \(512, 528\), outside",
        ),
        (GEMM, Copy, 0, _arg(0, _recorded(covered=None)), "passes 'src' bytes None, outside"),
        (SCALARS, Binary, 0, _arg(1, _unrecorded), "passes 'rhs' unprepared"),
        (SCALARS, Fill, 0, _arg(0, _prepared_for("lhs")), "prepares 'lhs' unplanned"),
        (SCALARS, Fill, 0, _arg(0, _recorded(source=None)), "prepares 'rhs' unplanned"),
    ],
    ids=[
        "an issue missing",
        "an issue repeated",
        "another instruction issued",
        "the instruction issued with other attributes",
        "an operand without a record",
        "another allocation's operand",
        "bytes outside the plan's",
        "unknown bytes",
        "a staged parameter not prepared",
        "a parameter prepared but not staged",
        "a parameter prepared from another value",
    ],
)
def test_lowering_stops_when_emission_leaves_the_plan(path, kind, nth, rewrite, message):
    """Rewrite the *nth* statement issuing *kind* as lowering adds it."""
    result = _analyzed(load_authored_ir(str(path)))
    original = CudaLowering.check
    seen = []

    def check(self, stmt, builder):
        if not (isinstance(stmt, Evaluate) and isinstance(stmt.callable, kind)) or len(seen) > nth:
            return original(self, stmt, builder)
        seen.append(stmt)
        if len(seen) <= nth:
            return original(self, stmt, builder)
        for changed in rewrite(self, stmt, builder, seen) or ():
            original(self, changed, builder)

    with patch.object(CudaLowering, "check", check), pytest.raises(LoweringError, match=message):
        finalize(result)


def test_lowering_refuses_a_plan_made_from_other_records():
    result = _analyzed(load_authored_ir(str(GEMM)))
    summary = get_metadata(result.function, AllocationSummaryMetadata)
    attach_metadata(result.function, replace(summary))
    with pytest.raises(LoweringError, match="needs the result of analyze"):
        finalize(result)


def _removed(child, mesh):
    return None


def _moved(child, mesh):
    return replace(child, scope=(-1, -1))


def _issued_on(child, mesh):
    return replace(child, mesh=mesh)


def _executed_on(child, mesh):
    return replace(child, op=TiledMma(atom=child.op.atom, execution_mesh=mesh))


def _beside_unknown(child, mesh):
    """Each role's own bytes narrowed to one, beside an allocation whose bytes are unknown."""
    roles = tuple(
        replace(role, owned=tuple((root, (0, 1)) for root, _ in role.owned) + ((None, None),))
        for role in child.roles
    )
    return replace(child, roles=roles)


@pytest.mark.parametrize(
    ("kind", "change", "message"),
    [
        (CopyAsync, _removed, r"lowering issues CopyAsync\(.*\) at"),
        (TiledMma, _executed_on, r"lowering issues TiledMma\(.*\) at .* plan holds TiledMma"),
        (CopyAsync, _moved, r"emitted child .* where the sync plan holds \(\(\d+, 0\), \(-1, -1\)\)"),
        (CopyAsync, _issued_on, r"CopyAsync\(.*\) at \(\d+, 0\) is issued on"),
        (Copy, _beside_unknown, r"passes 'src' bytes \(\d+, \d+\), outside \(\(0, 1\),\)"),
    ],
    ids=[
        "a child the plan does not hold",
        "an instruction with other attributes",
        "a child in another scope",
        "an issue on another mesh",
        "bytes outside its allocation's, beside unknown ones",
    ],
)
def test_lowering_refuses_a_plan_it_does_not_follow(kind, change, message):
    """Apply *change* to each planned child issuing *kind*, with the CTA mesh at hand."""
    result = _analyzed(load_authored_ir(str(GEMM)))
    plan = get_metadata(result.function, SyncPlanMetadata)
    cta = result.function.body.mesh
    changed = (
        change(child, cta) if isinstance(child.op, kind) else child for child in plan.children
    )
    planned = tuple(child for child in changed if child is not None)
    attach_metadata(result.function, replace(plan, children=planned))
    with pytest.raises(LoweringError, match=message):
        finalize(result)


def test_lowering_refuses_an_issue_off_its_scope_mesh():
    """A copy the plan leaves on its scope's mesh is issued on half of those threads."""
    result = _analyzed(load_authored_ir(str(GEMM)))
    original = convert.issue_mesh

    def issue_mesh(op, owner, enclosing):
        mesh = original(op, owner, enclosing)
        return enclosing[:, :4] if mesh is None and isinstance(op, CopyAsync) else mesh

    with (
        patch.object(convert, "issue_mesh", issue_mesh),
        pytest.raises(LoweringError, match=r"CopyAsync\(.*\) at \(\d+, 0\) is issued on"),
    ):
        finalize(result)


def test_lowering_refuses_issue_points_it_did_not_expect():
    result = _analyzed(load_authored_ir(str(GEMM)))
    with (
        patch.object(cuda_lowering, "required_execution_mesh", lambda op: None),
        pytest.raises(LoweringError, match="issue points are enumerated outside a planned action"),
    ):
        finalize(result)
