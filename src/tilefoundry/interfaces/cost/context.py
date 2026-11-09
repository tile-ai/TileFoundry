"""Context and result values of cost evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types, utils


@dataclass
class Context(typeinfer.TypeInferContext):
    """Cost Evaluator context for one topology window.

    ``topology_level=None`` exposes the types as written. A named level projects
    them to what one unit of that level holds, letting the same registered
    evaluator answer both global and per-unit questions.
    """

    selected_types: Mapping[int, types.Type] = field(default_factory=dict)
    selected_output_type: types.Type | None = None
    topology_level: str | None = None

    def type_of(self, node: expr.Expr) -> types.Type:
        selected = self.selected_types.get(id(node))
        return selected if selected is not None else node.type

    def local_type_of(self, node: expr.Expr) -> types.Type:
        """Return ``expr``'s Type in this context's topology window."""
        selected = self.selected_types.get(id(node))
        type_ = selected if selected is not None else node.type
        if self.topology_level is None:
            return type_
        return utils.local_type_of(type_, topology_level=self.topology_level, topologies=self.topologies)

    def local_output_type(self, call: expr.Call) -> types.Type:
        """Return the selected candidate output in recursive-local form."""
        output = self.selected_output_type
        if output is None:
            output = call.type
        if self.topology_level is None:
            return output
        return utils.local_type_of(output, topology_level=self.topology_level, topologies=self.topologies)


@dataclass(frozen=True)
class TrafficBytes:
    """Bytes one operand moves, read and write kept apart."""

    read: int = 0
    write: int = 0

    @property
    def total_bytes(self) -> int:
        """Bytes moved in either direction."""
        return self.read + self.write


@dataclass(frozen=True)
class Cost:
    """Leaf-local logical work for one selected ``OpCandidate``.

    ``flops`` groups leaf-local logical work by compute ``DType`` so one Op can
    report mixed work without selecting an ALU/TensorCore implementation.
    ``service`` counts what is not floating point at all -- a comparison, a
    select, an integer add -- by the service it asks for, because a dtype is not
    a kind of work. ``traffic`` carries one entry per operand in call order with
    the result last, so an Op that reads part of an input says so where it knows
    it. No field names a hardware implementation or a memory level.
    """

    flops: Mapping[expr.DType, int]
    traffic: tuple[TrafficBytes, ...]
    service: Mapping[str, int] = field(default_factory=dict)
    sent: tuple[tuple[str, TrafficBytes], ...] = ()
    """Bytes that left the unit they were on, by the boundary they crossed.

    ``traffic`` says which storage level each operand touched. Data handed
    between two units of one topology level touches the same storage at both
    ends and has still gone somewhere, so it is stated against that level's
    name in addition to the storage traffic, from the sending unit's own view.
    """

    @property
    def bytes(self) -> int:
        """Every operand's traffic, in either direction."""
        return sum(moved.total_bytes for moved in self.traffic)

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 0 for value in self.flops.values()):
            raise ValueError("Cost flops must be non-negative integers")
        if any(
            not isinstance(value, int) or value < 0
            for moved in self.traffic
            for value in (moved.read, moved.write)
        ):
            raise ValueError("Cost traffic must be non-negative integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.service.values()
        ):
            raise ValueError("Cost service work must be non-negative integers")


__all__ = ["Context", "Cost", "TrafficBytes"]
