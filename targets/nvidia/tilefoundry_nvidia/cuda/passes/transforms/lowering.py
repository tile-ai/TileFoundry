"""Lower scheduled HIR for CUDA with the ``sync`` plan's steps before each planned child.

See [passes §7.1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/passes.md#71-converthirtotir).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.emission import required_execution_mesh
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.analysis.sync import SyncChild, SyncPlanMetadata, scope_position
from tilefoundry.ir.core import Op, get_metadata
from tilefoundry.ir.tir import CpAsyncCommit, CpAsyncWait
from tilefoundry.ir.tir.memory import Fill
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.ir.types import StorageKind, TensorType
from tilefoundry.passes.transforms.convert_hir_to_tir import (
    ConvertHIRToTIR,
    IssuePlan,
    Lowering,
    LoweringError,
    OperandRecord,
    StagedRecord,
    TirBuilder,
)
from tilefoundry.target import bind
from tilefoundry.visitor_registry.instruction import call_params
from tilefoundry_nvidia.cuda import CudaTarget


def _same_op(a: Op, b: Op) -> bool:
    """Whether two instructions are the same class with the same attributes."""
    if type(a) is not type(b):
        return False
    names = (param.name for param in type(a)._op_schema.signature if param.kind == "attribute")
    return all(getattr(a, name) == getattr(b, name) for name in names)


@dataclass
class _Action:
    """The planned child whose statements lowering is emitting."""

    child: SyncChild
    issues: int | None
    issued: int = 0
    prepared: dict = field(default_factory=dict)


class _Builder(TirBuilder):
    """The core builder, showing each statement it adds to the lowering that owns it."""

    def __init__(self, blocks, *, lowering):
        super().__init__(blocks)
        self.lowering = lowering

    def add(self, stmt):
        self.lowering.check(stmt, self)
        super().add(stmt)


class _Issues(IssuePlan):
    """The core issue plan, telling the lowering how many points an issue loop covers."""

    def __init__(self, *, lowering):
        super().__init__()
        self.lowering = lowering

    def points(self, plan):
        points = tuple(super().points(plan))
        self.lowering.expect(len(points))
        yield from points


class CudaLowering(Lowering):
    """The core lowering with the plan's steps, checking what it issues belongs to the plan."""

    def __post_init__(self) -> None:
        super().__post_init__()
        ctx = self.context
        ctx.builder = _Builder(ctx.builder.blocks, lowering=self)
        ctx.issues = _Issues(lowering=self)
        self.plan = get_metadata(ctx.function, SyncPlanMetadata)
        self.planned = {} if self.plan is None else {c.position: c for c in self.plan.children}
        self.emitted: list[tuple[tuple[int, int], tuple[int, int] | None]] = []
        self.current: _Action | None = None
        self.stepping = False

    def refuse(self, message: str):
        raise LoweringError(f"{self.context.function.name!r}: {message}")

    def run(self) -> PrimFunction:
        function = self.context.function
        basis = (
            get_metadata(function, AllocationSummaryMetadata),
            get_metadata(function, LivenessMetadata),
        )
        if self.plan is None or len(self.plan.basis) != 2 or any(
            planned is not held for planned, held in zip(self.plan.basis, basis, strict=True)
        ):
            self.refuse("needs the result of analyze(..., analysis=('sync',)) on this program")
        lowered = super().run()
        self.close()
        expected = [(child.position, child.scope) for child in self.plan.children]
        if self.emitted != expected:
            index = next(
                (i for i, pair in enumerate(zip(self.emitted, expected, strict=False)) if pair[0] != pair[1]),
                min(len(self.emitted), len(expected)),
            )
            got = self.emitted[index] if index < len(self.emitted) else None
            want = expected[index] if index < len(expected) else None
            self.refuse(f"lowering emitted child {got} where the sync plan holds {want}")
        return lowered

    def before(self, position: tuple[int, int], ctx, op: Op | None = None) -> None:
        self.close()
        block = ctx.builder.blocks[-1]
        self.emitted.append((position, scope_position(self.liveness, block.region)))
        child = self.planned.get(position)
        if (op is None) != (child is None or child.op is None) or (
            op is not None and not _same_op(op, child.op)
        ):
            self.refuse(
                f"lowering issues {op!r} at {position}, where the sync plan holds "
                f"{None if child is None else child.op!r}"
            )
        self.stepping = True
        for step in self.plan.at(position):
            if step.kind == "commit":
                ctx.builder.add(Evaluate(CpAsyncCommit(), ()))
            elif step.kind == "wait":
                ctx.builder.add(Evaluate(CpAsyncWait(n=0), ()))
            else:
                ctx.builder.add(Evaluate(Sync(mesh=step.mesh), ()))
        self.stepping = False
        if op is not None:
            self.current = _Action(child, 1 if required_execution_mesh(op) is None else None)

    def expect(self, points: int) -> None:
        """Record how many issues the open action's projected instruction makes."""
        if self.current is None or self.current.issues is not None:
            self.refuse("issue points are enumerated outside a planned action")
        self.current.issues = points

    def close(self) -> None:
        """Require the open action to have issued what it planned."""
        action, self.current = self.current, None
        if action is not None and (action.issues is None or action.issued != action.issues):
            self.refuse(
                f"{action.child.op!r} at {action.child.position} issued {action.issued} "
                f"of {action.issues} planned issue(s)"
            )

    def check(self, stmt, builder: TirBuilder) -> None:
        """Require *stmt* to belong to the open action, or to touch no shared memory."""
        if self.stepping or not isinstance(stmt, Evaluate):
            return
        if not isinstance(stmt.callable, Op):
            self.close()
        action = self.current
        if action is None:
            if any(_shared(arg) for arg in stmt.args):
                self.refuse(f"{stmt.callable!r} takes shared memory outside a planned action")
            return
        child = action.child
        roles = {role.name: role for role in child.roles}
        if isinstance(stmt.callable, Fill) and stmt.args and isinstance(
            builder.record_of(stmt.args[0]), StagedRecord
        ):
            staged = builder.record_of(stmt.args[0])
            role = roles.get(staged.param)
            if role is None or not role.staged or staged.source is not role.argument:
                self.refuse(f"{child.op!r} at {child.position} prepares {staged.param!r} unplanned")
            action.prepared[staged.param] = stmt.args[0]
            self.reach(child, role, stmt.args[1], builder)
            return
        if not _same_op(stmt.callable, child.op):
            self.refuse(f"{stmt.callable!r} at {child.position} is not the planned {child.op!r}")
        held = next((block.mesh for block in reversed(builder.blocks) if block.mesh is not None), None)
        if held != child.mesh:
            self.refuse(f"{child.op!r} at {child.position} is issued on {held}, planned {child.mesh}")
        action.issued += 1
        for param, arg in zip(call_params(stmt.callable, len(stmt.args)), stmt.args, strict=True):
            role = roles.get(param.name)
            if role is not None and role.staged:
                if action.prepared.get(param.name) is not arg:
                    self.refuse(f"{child.op!r} at {child.position} passes {param.name!r} unprepared")
                continue
            self.reach(child, role, arg, builder)

    def reach(self, child, role, arg, builder) -> None:
        """Require a shared memory operand to belong to *role* and stay in its root's bytes."""
        if not _shared(arg):
            return
        where = f"{child.op!r} at {child.position}"
        record = builder.record_of(arg)
        spans = (
            ()
            if role is None or not isinstance(record, OperandRecord)
            else tuple(span for root, span in role.owned if root == record.leaf)
        )
        if not spans:
            self.refuse(f"{where} passes shared memory the plan does not hold for that parameter")
        if None in spans:
            return
        covered = record.covered
        if covered is None or not any(low <= covered[0] and covered[1] <= high for low, high in spans):
            self.refuse(f"{where} passes {role.name!r} bytes {covered}, outside {spans}")


def _shared(arg) -> bool:
    return isinstance(arg.type, TensorType) and arg.type.storage is StorageKind.SMEM


@bind(CudaTarget, ConvertHIRToTIR)
class CudaConvertHIRToTIR(ConvertHIRToTIR):
    """Lower scheduled HIR with the visitor that emits the sync plan."""

    visitor = CudaLowering


__all__ = ["CudaConvertHIRToTIR", "CudaLowering"]
