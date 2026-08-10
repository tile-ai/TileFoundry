"""Effect-form TIR Op ``tir.Sync`` — a mesh-scoped barrier emitted by ``T.sync(m)``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilefoundry.ir.core import Op, VerifyError
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.shard import product
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.ir.types.shard.layout_algebra import apply as _apply
from tilefoundry.ir.types.shard.layout_algebra import size as _size
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

_WARP_SIZE = 32


@register_op(dialect="T", category="sync")
class Sync(Op):
    """Mesh-scoped barrier — emitted by ``T.sync(m)``."""

    mesh = ParamDef(kind="attribute", annotation=Mesh)


@register_typeinfer(Sync)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


def _legal_slice_of(m: Mesh, e: Mesh) -> bool:
    """Return whether *m* reconstructs as a constant slice of full mesh *e*.

    Topology, names, strides, bounds, and decomposed offset must agree. The
    final proof rebuilds the slice and compares it structurally.
    """
    if isinstance(e.layout, ComposedLayout):
        return False
    if e.topologies != m.topologies or e.names != m.names:
        return False
    region = m.layout
    if not isinstance(region, ComposedLayout):
        return False
    outer = region.outer
    p = e.layout
    if not isinstance(outer, Layout) or outer.strides != p.strides:
        return False
    sub, pshape = outer.shape, p.shape
    if len(sub) != len(pshape):
        return False
    if any(not isinstance(s, int) for s in sub) or any(not isinstance(s, int) for s in pshape):
        return False
    if not isinstance(region.offset, int) or any(s > ps for s, ps in zip(sub, pshape)):
        return False

    rem = region.offset
    starts = [0] * len(sub)
    for i in sorted(range(len(sub)), key=lambda k: -p.strides[k]):
        st = p.strides[i]
        if not isinstance(st, int) or st <= 0:
            return False
        starts[i] = rem // st
        rem -= starts[i] * st
    if rem != 0 or any(
        not (0 <= starts[i] and starts[i] + sub[i] <= pshape[i]) for i in range(len(sub))
    ):
        return False
    key = tuple(slice(starts[i], starts[i] + sub[i]) for i in range(len(sub)))
    try:
        return e[key] == m
    except (ValueError, IndexError):
        return False


@register_verify_stmt(Sync)
def _(call: "Call", ctx: "VerifyContext") -> None:
    """Verify a Sync references an enclosing mesh or its legal constant slice.

    The participant set must classify to a supported barrier; dynamic,
    non-contiguous, and cross-warp-unaligned subsets are rejected.

    See [tir §1.5](docs/spec/tir.md#15-sync).
    """
    m = call.target.mesh
    if not isinstance(m, Mesh):
        raise VerifyError(
            f"T.sync expects a Mesh argument (m or a slice m[...]), got {type(m).__name__}"
        )
    scope = ctx.mesh_scope
    if not isinstance(m.layout, ComposedLayout):
        ok = any(m == e for e in scope)
    else:
        ok = any(_legal_slice_of(m, e) for e in scope)
    if not ok:
        raise VerifyError(
            "T.sync(m): not inside a MeshScope binding the synced mesh; T.sync "
            "must reference an enclosing `with Mesh(...) as m` (m or a legal "
            "m[slice])"
        )

    classify(m)


class SyncBarrier(Enum):
    """Hardware barrier selected for whole-block, warp, subset, or grid sync."""

    SYNCTHREADS = "syncthreads"
    SYNCWARP = "syncwarp"
    BAR_SYNC = "bar_sync"
    GRID = "grid"


@dataclass(frozen=True)
class Participation:
    """Describe a contiguous participant interval and its barrier properties."""

    base: int
    count: int
    block_domain: int
    single_warp: bool
    full_cta: bool
    lane_mask: int


def _participant_layout(mesh: Mesh) -> "tuple[Layout, int]":
    """The (outer layout, offset) describing which threads participate.

    For a sliced mesh ``layout`` is a ``ComposedLayout`` whose ``outer`` is the
    participating sub-box and ``offset`` the slice origin; for an un-sliced mesh
    the whole plain-``Layout`` ``layout`` participates at offset 0.
    """
    ly = mesh.layout
    if isinstance(ly, ComposedLayout):
        outer = ly.outer
        if not isinstance(outer, Layout):
            raise VerifyError("T.sync: mesh slice must be a plain-Layout affine scope")
        return outer, ly.offset
    return ly, 0


def participation(mesh: Mesh) -> Participation:
    """Derive the participating thread set of ``mesh``.

    Raises ``VerifyError`` for a malformed mesh (dynamic extent) or an
    unsupported slice (non-contiguous / overlapping).
    """
    domain = product(mesh.topologies)
    if domain is None:
        raise VerifyError(
            "T.sync: a mesh with a dynamic topology extent cannot be classified; "
            "only a static thread count is supported"
        )
    outer, offset = _participant_layout(mesh)
    shape = outer.shape
    strides = outer.strides
    if (
        not isinstance(offset, int)
        or any(not isinstance(s, int) for s in shape)
        or strides is None
        or any(not isinstance(s, int) for s in strides)
    ):
        raise VerifyError("T.sync: a mesh with a dynamic layout cannot be classified")

    count = _size(outer)

    lins = {offset + _apply(outer, c) for c in range(count)}
    if len(lins) != count:
        raise VerifyError("T.sync: mesh layout maps several coords to one thread (overlap)")
    base = min(lins)
    if lins != set(range(base, base + count)):
        raise VerifyError(
            "T.sync: the sliced mesh is not a contiguous thread interval; only a "
            "single-warp lane subset or a contiguous warp-aligned multi-warp "
            "range is supported"
        )
    if base + count > domain:
        raise VerifyError("T.sync: participant range exceeds the block thread domain")

    full_cta = base == 0 and count == domain
    single_warp = count <= _WARP_SIZE and (base // _WARP_SIZE == (base + count - 1) // _WARP_SIZE)
    lane_mask = (((1 << count) - 1) << (base % _WARP_SIZE)) & 0xFFFFFFFF if single_warp else 0
    return Participation(
        base=base,
        count=count,
        block_domain=domain,
        single_warp=single_warp,
        full_cta=full_cta,
        lane_mask=lane_mask,
    )


def classify(mesh: Mesh) -> SyncBarrier:
    """Pick the hardware barrier for ``mesh``.

    Pick the hardware barrier for ``mesh``. Raises ``VerifyError`` for a
    cross-warp subset that is not warp-aligned.
    """
    topos = mesh.topologies
    if all(t.name == "cta" for t in topos):
        if isinstance(mesh.layout, ComposedLayout):
            raise VerifyError("T.sync: a partial grid sync (cta mesh slice) is unsupported")
        return SyncBarrier.GRID
    p = participation(mesh)
    if p.full_cta:
        return SyncBarrier.SYNCWARP if p.count == _WARP_SIZE else SyncBarrier.SYNCTHREADS
    if p.single_warp:
        return SyncBarrier.SYNCWARP
    if p.base % _WARP_SIZE != 0 or p.count % _WARP_SIZE != 0:
        raise VerifyError(
            "T.sync: a cross-warp subset must be warp-aligned — both the base "
            "and the count must be multiples of 32"
        )
    return SyncBarrier.BAR_SYNC


__all__ = [
    "Sync",
    "SyncBarrier",
    "Participation",
    "participation",
    "classify",
]
