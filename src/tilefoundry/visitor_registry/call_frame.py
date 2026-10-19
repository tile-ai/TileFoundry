"""Call-path context and the shared entry into a called definition."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.core.module import ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.visitor import collect_exprs


@dataclass(frozen=True)
class CallFrame:
    """Current definition and its effective execution context on a call path."""

    module: Module
    function: Function | None
    target: Target | None = None
    topologies: tuple[Topology, ...] | None = None

    def __post_init__(self):
        if self.target is None:
            object.__setattr__(self, "target", self.module.target)
        if self.topologies is None:
            object.__setattr__(self, "topologies", tuple(self.module.topologies or ()))

    @staticmethod
    def resolve(call, frame) -> CallFrame | None:
        """Enter a definition through the current module's instance slots.

        Function and PrimFunction targets preserve the caller's module/context,
        or return None when there is no caller frame. Such unframed direct calls
        follow the caller's existing path; None therefore does not imply an Op.
        Instance targets require a frame. Op calls always return None.
        """
        target = call.target
        if isinstance(target, (Function, PrimFunction)):
            return None if frame is None else replace(frame, function=target)
        slot, index = CallFrame.module_slot(call)
        if slot is None:
            return None
        if frame is None:
            raise ValueError("instance call requires a CallFrame")
        slot = frame.module.children[slot.key]
        index = index.value if isinstance(index, Constant) else 0
        child = slot.elements[index]
        return CallFrame(
            child,
            child.entry_function(),
            child.target if child.target is not None else frame.target,
            child.topologies if child.topologies is not None else frame.topologies,
        )

    @staticmethod
    def module_slot(call):
        target = call.target
        if isinstance(target, ModuleList):
            return target, 0
        if (
            isinstance(target, Call)
            and isinstance(target.target, TupleGetItem)
            and isinstance(target.args[0], ModuleList)
        ):
            return target.args[0], target.args[1]
        return None, None

    def walk(self, *, postorder=False):
        """Reach HIR definitions once, in DFS first-arrival or dependency order.

        This walks calls from the supplied function, not all declarations in
        its Module. Per-call signature checks still inspect every call site.
        """
        seen = set()

        def visit(current):
            if id(current.function) in seen:
                return
            seen.add(id(current.function))
            if not postorder:
                yield current
            for call in collect_exprs(current.function.body):
                if (
                    isinstance(call, Call)
                    and (child := CallFrame.resolve(call, current)) is not None
                ):
                    yield from visit(child)
            if postorder:
                yield current

        yield from visit(self)
