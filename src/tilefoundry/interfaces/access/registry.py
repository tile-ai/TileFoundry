"""The registry of access relations: one handler per Op, and what is derived from its boundaries."""

from __future__ import annotations

from typing import Callable

import isl

from tilefoundry import dispatch
from tilefoundry.ir import dim

from .relation import Relation

relation_registry: dispatch.Registry = dispatch.Registry("access_relation")


def register_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register the one handler that states an Op's coordinates.

    The handler signature is ``(call, ctx) -> tuple[Relation, ...]``: one
    per argument, then one per result field. It answers before the Call has a
    Type, so it may read its operands, its Op's attributes and the values its
    parameters bind, and not the Call's own Type. Projecting that answer onto a
    reader's view is a separate step, `local_relations_of`.
    """

    def decorate(handler: Callable) -> Callable:
        relation_registry.register(op_cls, handler)
        return handler

    return decorate


def relations_of(call, ctx) -> tuple[Relation, ...]:
    """One Op's declared relations, before deriving what the Call returns.

    This is what type inference asks, so nothing here may consult the Type being
    derived. What is held is the one thing a caller counts on without that Type:
    one relation per operand, in argument order, and then at least one result.
    Each relation answers for its own parameters when it is built.
    """
    op_cls = type(call.target)
    handler = relation_registry.get(op_cls, None)
    if handler is None:
        raise ValueError(
            f"{op_cls.__name__} states no access relations, and there is no "
            "fallback: register one with register_relation"
        )
    relations = handler(call, ctx)
    if not isinstance(relations, tuple) or not all(
        isinstance(access, Relation) for access in relations
    ):
        raise ValueError(
            f"{op_cls.__name__} states its boundaries as one access.Relation each, "
            f"in a tuple; got {relations!r}"
        )
    if len(relations) <= len(call.args):
        raise ValueError(
            f"{op_cls.__name__} describes {len(relations)} boundar"
            f"{'y' if len(relations) == 1 else 'ies'} of a call with {len(call.args)} "
            "arguments; one per argument comes first, and then what it produces"
        )
    return relations


def iteration_universe(relations: tuple[Relation, ...]) -> "isl.set | None":
    """The whole space one Op walks, from the boundaries that answer about it.

    There is no separate place an Op declares this: its boundaries do, each on
    the part it answers on, so the whole is their union once their parameters are
    lined up. Every reader derives it here so that one Op is one space to all of
    them. It says what the Op walks, not that the Op was right about it.
    """
    walked = None
    for access in relations:
        own = access.relation.domain()
        walked = own if walked is None else walked.union(own)
    return None if walked is None else walked.coalesce()


def values_of(relations: tuple[Relation, ...]) -> dim.IslParamValues:
    """Every parameter this Op binds, by name, across all of its boundaries.

    One name is one value for the whole Op, so a relation that gains a parameter
    by being composed or restricted still knows what it stands for, and a name
    two boundaries bind to different values is refused rather than resolved by
    whichever came last.
    """
    merged: dim.IslParamValues = {}
    for access in relations:
        for name, value in access.values.items():
            if merged.setdefault(name, value) is not value:
                raise ValueError(
                    f"parameter {name!r} stands for two different values across one "
                    f"Op's boundaries; one name in one Op is one value"
                )
    return merged


__all__ = [
    "iteration_universe",
    "register_relation",
    "relation_registry",
    "relations_of",
    "values_of",
]
