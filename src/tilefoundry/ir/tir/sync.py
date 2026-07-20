"""Effect-form TIR Op ``tir.Sync`` — a mesh-scoped barrier emitted by ``T.sync(m)``."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilefoundry.ir.core import Op, VerifyError
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.ir.types.shard.layout_algebra import apply as _apply
from tilefoundry.ir.types.shard.layout_algebra import size as _size
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

# A warp is 32 lanes — the granularity of ``__syncwarp`` / ``bar.sync`` counts.
_WARP_SIZE = 32


@register_op(dialect="T", category="sync")
class Sync(Op):
    """Mesh-scoped barrier — emitted by ``T.sync(m)``."""
    mesh = ParamDef(kind="attribute", annotation=Mesh)


@register_typeinfer(Sync)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


def _legal_slice_of(m: Mesh, e: Mesh) -> bool:
    """Is ``m`` (a sliced mesh, ``m.layout`` a ``ComposedLayout``) a legal
    constant slice of the enclosing full mesh ``e``?

    Mechanically checkable, not inferred from a few equal fields: ``e`` must be
    un-sliced, share ``m``'s full topology tuple and axis names, and ``m``'s
    slice must reconstruct as ``e[start_i : start_i+sub_i]`` — same strides,
    per-axis sub-extents bounded by ``e``'s shape, and an offset that decomposes
    into in-range per-axis starts. The proof ends by rebuilding ``e[key]`` and
    comparing it to ``m`` (so a forged slice cannot pass)."""
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
    # Recover per-axis starts from the offset over the parent strides.
    rem = region.offset
    starts = [0] * len(sub)
    for i in sorted(range(len(sub)), key=lambda k: -p.strides[k]):
        st = p.strides[i]
        if not isinstance(st, int) or st <= 0:
            return False
        starts[i] = rem // st
        rem -= starts[i] * st
    if rem != 0 or any(not (0 <= starts[i] and starts[i] + sub[i] <= pshape[i]) for i in range(len(sub))):
        return False
    key = tuple(slice(starts[i], starts[i] + sub[i]) for i in range(len(sub)))
    try:
        return e[key] == m
    except (ValueError, IndexError):
        return False


@register_verify_stmt(Sync)
def _(call: "Call", ctx: "VerifyContext") -> None:
    """A ``Sync`` must reference an enclosing ``with Mesh(...) as m`` — either
    ``m`` itself (full mesh) or a legal constant slice ``m[...]`` — and its
    participant set must lower to a supported barrier.

    A full mesh (plain-``Layout`` ``layout``) is accepted only by equality with
    an enclosing mesh; a sliced mesh (``ComposedLayout`` ``layout``) only through
    the slice-derived-from-enclosing proof (:func:`_legal_slice_of`). The
    enclosing ``MeshScope`` stack arrives on ``ctx.mesh_scope``. ``classify``
    then rejects a dynamic / non-contiguous / cross-warp-unaligned participant
    set."""
    m = call.target.mesh
    if not isinstance(m, Mesh):
        raise VerifyError(
            f"T.sync expects a Mesh argument (m or a slice m[...]), got "
            f"{type(m).__name__}"
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
    # Feasibility: raises for a dynamic / non-contiguous / unaligned slice.
    classify(m)


class SyncBarrier(Enum):
    """The hardware barrier a sync lowers to."""
    SYNCTHREADS = "syncthreads"  # whole block, more than one warp
    SYNCWARP = "syncwarp"        # whole block of one warp, or a single-warp subset
    BAR_SYNC = "bar_sync"        # named barrier — a warp-aligned multi-warp subset
    GRID = "grid"                # grid-wide software barrier (cta-scope mesh)


@dataclass(frozen=True)
class Participation:
    """The participating thread set of a (possibly sliced) sync mesh."""
    base: int           # linear CTA thread index of the first participant
    count: int          # number of participating threads
    block_domain: int   # total threads in the block (topology product)
    single_warp: bool   # the participants live inside one warp
    full_cta: bool       # the participants are the whole block
    lane_mask: int      # __syncwarp mask (only meaningful when single_warp)


def _topology_domain(mesh: Mesh) -> "int | None":
    """Block thread count = product of the mesh's topology extents. ``None`` if
    any topology extent is dynamic (launch-provided)."""
    topos = mesh.topologies or (mesh.topology,)
    domain = 1
    for t in topos:
        if not isinstance(t.size, int):
            return None
        domain *= t.size
    return domain


def _participant_layout(mesh: Mesh) -> "tuple[Layout, int]":
    """The (outer layout, offset) describing which threads participate.

    For a sliced mesh ``layout`` is a ``ComposedLayout`` whose ``outer`` is the
    participating sub-box and ``offset`` the slice origin; for an un-sliced mesh
    the whole plain-``Layout`` ``layout`` participates at offset 0."""
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
    unsupported slice (non-contiguous / overlapping)."""
    domain = _topology_domain(mesh)
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
    # Linear thread index of each participant: offset + outer(coord).
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
    single_warp = count <= _WARP_SIZE and (
        base // _WARP_SIZE == (base + count - 1) // _WARP_SIZE
    )
    lane_mask = (
        (((1 << count) - 1) << (base % _WARP_SIZE)) & 0xFFFFFFFF if single_warp else 0
    )
    return Participation(
        base=base,
        count=count,
        block_domain=domain,
        single_warp=single_warp,
        full_cta=full_cta,
        lane_mask=lane_mask,
    )


def classify(mesh: Mesh) -> SyncBarrier:
    """Pick the hardware barrier for ``mesh``. Raises ``VerifyError`` for a
    cross-warp subset that is not warp-aligned."""
    topos = mesh.topologies or (mesh.topology,)
    if topos and all(getattr(t, "name", None) == "cta" for t in topos):
        # A cta-scope mesh maps to the grid-wide barrier; only the full mesh
        # (no cta slice) has a supported barrier.
        if isinstance(mesh.layout, ComposedLayout):
            raise VerifyError(
                "T.sync: a partial grid sync (cta mesh slice) is unsupported"
            )
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
