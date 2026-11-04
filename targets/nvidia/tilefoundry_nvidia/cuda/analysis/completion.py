"""When each memory effect of an issued instruction is complete for later accesses.

A synchronisation plan orders a later access after an earlier one only once the
earlier one is complete. An Op without a registration completes on return. See
[analysis §2](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/analysis.md#2-completion-registry).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.registries import DispatchRegistry
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.nn.wgmma import WARPGROUP, Wgmma


@dataclass(frozen=True)
class Issued:
    """One instruction lowering issues: the Op, its parameters and their types."""

    op: object
    params: tuple
    types: tuple


@dataclass(frozen=True)
class Effect:
    """When one read or write of an issued parameter is complete.

    ``on_return`` effects are complete for the issuing threads when the statement
    returns, ``group`` ones once those threads wait for ``group``. ``proxy`` is
    the path to shared memory, ``generic`` or ``async``. ``unit`` is the mesh of
    one group of threads that run the group's steps, and the issue itself,
    together. A ``fence`` effect on registers needs the group's fence since any
    other access to them, except an earlier one with the same ``chain``.
    """

    param: str
    write: bool
    mechanism: str
    group: str | None = None
    proxy: str = "generic"
    unit: object = None
    fence: bool = False
    chain: object = None


completion_registry: DispatchRegistry = DispatchRegistry("completion")


def effects(issued: Issued, **fields) -> tuple[Effect, ...]:
    """Each read, then each write, of *issued*'s parameters, all with *fields*."""
    return tuple(
        Effect(param.name, write, **fields)
        for param in issued.params
        for write in (False, True)
        if param.effect & (MemoryEffect.WRITE if write else MemoryEffect.READ)
    )


def completion_of(issued: Issued, *, key: type | None = None) -> tuple[Effect, ...]:
    """The effects declared for *issued*, looked up by *key* or its Op's exact class."""
    handler = completion_registry.lookup(type(issued.op) if key is None else key)
    return effects(issued, mechanism="on_return") if handler is None else handler(issued)


def _wgmma_effects(issued: Issued) -> tuple[Effect, ...]:
    """Every operand completes at the warpgroup's wgmma wait.

    Shared operands go through the async proxy. Registers need ``wgmma.fence``,
    except between accumulator accesses of wgmmas of one shape (PTX wgmma.fence).
    """
    storage = dict(zip((param.name for param in issued.params), issued.types, strict=True))
    shape = (issued.op.atom.bindings["n"], issued.op.atom.bindings["dtype"])

    def declared(effect):
        if storage[effect.param].storage is StorageKind.SMEM:
            return replace(effect, proxy="async")
        return replace(effect, fence=True, chain=shape if effect.param == "acc" else None)

    return tuple(
        declared(effect)
        for effect in effects(issued, mechanism="group", group="wgmma", unit=WARPGROUP)
    )


completion_registry.register(
    CopyAsync, lambda issued: effects(issued, mechanism="group", group="cp_async")
)
completion_registry.register(Wgmma, _wgmma_effects)
completion_registry.register(
    TiledMma, lambda issued: completion_of(issued, key=type(issued.op.atom))
)


__all__ = ["Effect", "Issued", "completion_of", "completion_registry", "effects"]
