"""Define the common behavior of algorithm-owned schedule plans.

A plan is the typed result of one solve. Each subtype owns its decisions, JSON,
and human rendering because algorithms at different levels have no shared
decision schema. Plans are produced and consumed in the solving process.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSpecRef:
    """Stable identity of the installed target facts one plan relies on.

    Which hardware documents a decision was made against is the same question
    whatever was decided, so every plan states it the same way. A target that
    was constructed directly rather than installed from documents has a name but
    no digest, and says so with an empty one rather than a fabricated value.
    """

    architecture_id: str
    architecture_digest: str
    device_id: str
    device_digest: str

    @classmethod
    def of(cls, target: object) -> "TargetSpecRef":
        """The identity *target* publishes for its installed documents."""
        architecture = target.architecture  # type: ignore[attr-defined]
        device = target.device  # type: ignore[attr-defined]
        return cls(
            architecture_id=getattr(target, "architecture_id", None) or architecture.name,
            architecture_digest=getattr(target, "architecture_digest", None) or "",
            device_id=getattr(target, "device_id", None) or device.name,
            device_digest=getattr(target, "device_digest", None) or "",
        )


class SchedulePlan:
    """One solve's decisions, owned by the algorithm that made them.

    A subtype carries the typed decisions of its own algorithm, and owns the
    whole of its JSON object and its human rendering. Nothing here constrains
    what those decisions are.
    """

    def verify(self, module: "Module", function: "Function", topology: "Topology") -> None:
        """Check that this plan is internally consistent against its inputs.

        This is a structural check, not a second solve: it confirms that the
        plan refers only to things that exist and that its own references agree
        with each other. It states nothing about whether the schedule is good.
        """
        raise NotImplementedError

    def to_json(self) -> str:
        """The whole plan as JSON, owned entirely by the subtype."""
        raise NotImplementedError

    def render(self) -> str:
        """The whole plan as human-readable text, owned entirely by the subtype."""
        raise NotImplementedError


class PlanVerificationError(ValueError):
    """A plan referred to something that does not exist, or contradicted itself."""


__all__ = ["PlanVerificationError", "SchedulePlan", "TargetSpecRef"]
