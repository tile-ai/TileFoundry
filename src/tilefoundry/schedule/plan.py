"""What a scheduling algorithm hands back.

A Plan is the semantic result of one solve: the decisions an agent needs, in the
vocabulary of the algorithm that made them. The base names only what every plan
must be able to do, and deliberately not a shared shape -- two algorithms
scheduling different hardware at different levels do not decide the same things,
and a common schema would either describe none of them or force both to pretend.

There is no version field, no deserializer, and no renderer registry. A plan is
produced by the algorithm that solved for it, in the process that solved; reading
one back from text would mean trusting a document to describe decisions nobody
made in this run.
"""

from __future__ import annotations


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


__all__ = ["PlanVerificationError", "SchedulePlan"]
