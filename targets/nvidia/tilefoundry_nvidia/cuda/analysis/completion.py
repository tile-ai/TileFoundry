"""When each memory effect of an issued instruction is complete for later accesses.

A synchronisation plan orders a later access after an earlier one only once the
earlier one is complete. An Op without a registration completes on return. See
[analysis §2](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/analysis.md#2-completion-registry).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto

from tilefoundry import dispatch
from tilefoundry.ir import expr, tir, types
from tilefoundry_nvidia.cuda.ir.tir import memory, nn


class Mechanism(Enum):
    """When an effect completes: when its statement returns, or at its group's wait."""

    ON_RETURN = auto()
    GROUP = auto()


class Proxy(Enum):
    """The path an effect takes to shared memory."""

    GENERIC = auto()
    ASYNC = auto()


class Group(Enum):
    """A completion group: a commit closes the effects issued into it, a wait completes them."""

    CP_ASYNC = auto()
    WGMMA = auto()


@dataclass(frozen=True)
class Issued:
    """One instruction lowering issues: the Op, its parameters and their types."""

    op: object
    params: tuple
    types: tuple


@dataclass(frozen=True)
class Effect:
    """When one read or write of an issued parameter is complete.

    ``ON_RETURN`` effects are complete for the issuing threads when the statement
    returns, ``GROUP`` ones once those threads wait for ``group``. ``proxy`` is
    the path to shared or global memory. ``unit`` is the mesh of one group of
    threads that run the group's steps, and the issue itself, together. A
    ``fence`` effect on registers needs the group's fence since any other access to them, except an
    earlier one with the same ``chain``.
    """

    param: str
    write: bool
    mechanism: Mechanism
    group: Group | None = None
    proxy: Proxy = Proxy.GENERIC
    unit: types.Mesh | None = None
    fence: bool = False
    chain: tuple[int, expr.DType] | None = None


completion_registry: dispatch.Registry = dispatch.Registry("completion")


def effects(issued: Issued, **fields) -> tuple[Effect, ...]:
    """Each read, then each write, of *issued*'s parameters, all with *fields*."""
    return tuple(
        Effect(param.name, write, **fields)
        for param in issued.params
        for write in (False, True)
        if param.effect & (expr.MemoryEffect.WRITE if write else expr.MemoryEffect.READ)
    )


def completion_of(issued: Issued, *, key: type | None = None) -> tuple[Effect, ...]:
    """The effects declared for *issued*, looked up by *key* or its Op's exact class."""
    handler = completion_registry.get(type(issued.op) if key is None else key, None)
    return effects(issued, mechanism=Mechanism.ON_RETURN) if handler is None else handler(issued)


def wgmma_effects(issued: Issued) -> tuple[Effect, ...]:
    """Every operand completes at the warpgroup's wgmma wait.

    Shared operands go through the async proxy. Registers need ``wgmma.fence``,
    except between accumulator accesses of wgmmas of one shape (PTX wgmma.fence).
    """
    storage = dict(zip((param.name for param in issued.params), issued.types, strict=True))
    shape = (issued.op.atom.bindings["n"], issued.op.atom.bindings["dtype"])

    def declared(effect):
        if storage[effect.param].storage is expr.StorageKind.SMEM:
            return replace(effect, proxy=Proxy.ASYNC)
        return replace(effect, fence=True, chain=shape if effect.param == "acc" else None)

    return tuple(
        declared(effect)
        for effect in effects(issued, mechanism=Mechanism.GROUP, group=Group.WGMMA, unit=nn.WARPGROUP)
    )


def tensor_map_effects(issued: Issued) -> tuple[Effect, ...]:
    """Every operand completes on return, through the async proxy.

    The emitter finishes each copy before the operation ends, so no completion
    group stays in flight.
    """
    return effects(issued, mechanism=Mechanism.ON_RETURN, proxy=Proxy.ASYNC)


completion_registry.register(
    tir.async_copy.CopyAsync, lambda issued: effects(issued, mechanism=Mechanism.GROUP, group=Group.CP_ASYNC)
)
completion_registry.register(nn.Wgmma, wgmma_effects)
completion_registry.register(memory.CopyAsyncTensor, tensor_map_effects)
completion_registry.register(memory.CopyReduceAsyncTensor, tensor_map_effects)
completion_registry.register(
    nn.TiledMma, lambda issued: completion_of(issued, key=type(issued.op.atom))
)


__all__ = [
    "Effect",
    "Group",
    "Issued",
    "Mechanism",
    "Proxy",
    "completion_of",
    "completion_registry",
    "effects",
]
