"""Effect-form TIR Op ``tir.Sync`` — a mesh-scoped barrier emitted by ``T.sync(m)``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, types, utils
from tilefoundry.ir.types import int_tuple

WARP_SIZE = 32


@expr.register_op(dialect="T", category="sync")
class Sync(expr.Op):
    """Mesh-scoped barrier — emitted by ``T.sync(m)``."""

    execution_mesh = pattern.Or(
        pattern.thread_execution_mesh(),
        pattern.Mesh(("cta",), pattern.thread_execution_mesh().layout),
    )

    mesh = expr.ParamDef(kind="attribute", annotation=types.Mesh)


@typeinfer.register(Sync)
def _(call: "Call", ctx: "TypeInferContext") -> types.TupleType:
    return types.TupleType.Void


def legal_slice_of(m: types.Mesh, e: types.Mesh) -> bool:
    """Return whether *m* reconstructs as a constant slice of full mesh *e*.

    ``types.sub_box`` decodes the box from the first and last unit of *m*. The
    final proof rebuilds the slice and compares it structurally, names included.
    """
    if isinstance(e.layout, types.ComposedLayout) or e.names != m.names:
        return False
    box = types.sub_box(e, m)
    if box is None:
        return False
    try:
        return e[box] == m
    except (ValueError, IndexError):
        return False


def mesh_str(mesh: types.Mesh) -> str:
    """A mesh as its topologies and the shape they are viewed through."""
    topologies = ", ".join(f"{t.name}({t.size})" for t in mesh.topologies)
    return f"({topologies})[{', '.join(str(d) for d in types.flatten_to_tuple(mesh.layout.shape))}]"


def no_enclosing_mesh_error(m: types.Mesh, scope: "tuple[types.Mesh, ...]") -> expr.VerifyError:
    """Name which of the three ways a sync failed to reach an enclosing mesh.

    All three land on the same check, so saying only "not inside a MeshScope"
    describes the first and misleads about the other two: there may well be an
    enclosing scope that the synced mesh simply does not come from.
    """
    if not scope:
        return expr.VerifyError(
            f"T.sync({mesh_str(m)}): no enclosing mesh scope — a sync must name a mesh "
            "bound by an enclosing `with Mesh(...) as m`"
        )
    enclosing = ", ".join(mesh_str(e) for e in scope)
    if not isinstance(m.layout, types.ComposedLayout):
        return expr.VerifyError(
            f"T.sync({mesh_str(m)}): no enclosing scope binds that mesh; the scopes in "
            f"force bind {enclosing}"
        )
    return expr.VerifyError(
        f"T.sync({mesh_str(m)}): that sub-box is not a slice any enclosing mesh can "
        f"produce; the scopes in force bind {enclosing}, and a sliced sync must name a "
        "sub-box `m[...]` builds from one of them"
    )


@verify.register(Sync)
def _(call: "Call", ctx: "verify.Context") -> None:
    """Verify a Sync references an enclosing mesh or its legal constant slice.

    The unit set must classify to a supported barrier; dynamic,
    non-contiguous, and cross-warp-unaligned subsets are rejected.

    See [tir §1.5](docs/spec/tir.md#15-sync).
    """
    m = call.target.mesh
    if not isinstance(m, types.Mesh):
        raise expr.VerifyError(
            f"T.sync expects a Mesh argument (m or a slice m[...]), got {type(m).__name__}"
        )
    scope = ctx.enclosing_meshes()
    if not isinstance(m.layout, types.ComposedLayout):
        ok = any(m == e for e in scope)
    else:
        ok = any(legal_slice_of(m, e) for e in scope)
    if not ok:
        raise no_enclosing_mesh_error(m, scope)

    classify(m)


class SyncBarrier(Enum):
    """Hardware barrier selected for whole-block, warp, subset, or grid sync."""

    SYNCTHREADS = "syncthreads"
    SYNCWARP = "syncwarp"
    BAR_SYNC = "bar_sync"
    GRID = "grid"


@dataclass(unsafe_hash=True)
class SyncScope:
    """The contiguous run of threads one barrier synchronizes, and its barrier properties."""

    base: int
    count: int
    block_domain: int
    single_warp: bool
    full_cta: bool
    lane_mask: int


def sync_scope(mesh: types.Mesh) -> SyncScope:
    """Derive the run of threads that one barrier over ``mesh`` synchronizes.

    Raises ``VerifyError`` for a malformed mesh (dynamic extent) or an
    unsupported slice (non-contiguous / overlapping).
    """
    domain = int_tuple.product(mesh.topologies)
    if domain is None:
        raise expr.VerifyError(
            "T.sync: a mesh with a dynamic topology extent cannot be classified; "
            "only a static thread count is supported"
        )
    try:
        outer, offset = types.plain_layout(utils.flat_layout(mesh)), types.offset(mesh)
    except ValueError as error:
        raise expr.VerifyError(f"T.sync: {error}") from error
    shape = outer.shape
    strides = outer.strides
    if (
        not isinstance(offset, int)
        or any(not isinstance(s, int) for s in shape)
        or any(not isinstance(s, int) for s in strides)
    ):
        raise expr.VerifyError("T.sync: a mesh with a dynamic layout cannot be classified")

    count = types.size(outer)

    lins = {offset + outer(c) for c in range(count)}
    if len(lins) != count:
        raise expr.VerifyError("T.sync: mesh layout maps several coords to one thread (overlap)")
    base = min(lins)
    if lins != set(range(base, base + count)):
        raise expr.VerifyError(
            "T.sync: the sliced mesh is not a contiguous thread interval; only a "
            "single-warp lane subset or a contiguous warp-aligned multi-warp "
            "range is supported"
        )
    if base + count > domain:
        raise expr.VerifyError("T.sync: the thread run exceeds the block thread domain")

    full_cta = base == 0 and count == domain
    single_warp = count <= WARP_SIZE and (base // WARP_SIZE == (base + count - 1) // WARP_SIZE)
    lane_mask = (((1 << count) - 1) << (base % WARP_SIZE)) & 0xFFFFFFFF if single_warp else 0
    return SyncScope(
        base=base,
        count=count,
        block_domain=domain,
        single_warp=single_warp,
        full_cta=full_cta,
        lane_mask=lane_mask,
    )


def classify(mesh: types.Mesh) -> SyncBarrier:
    """Pick the hardware barrier for ``mesh``.

    Pick the hardware barrier for ``mesh``. Raises ``VerifyError`` for a
    cross-warp subset that is not warp-aligned.
    """
    topos = mesh.topologies
    if all(t.name == "cta" for t in topos):
        if isinstance(mesh.layout, types.ComposedLayout):
            raise expr.VerifyError("T.sync: a partial grid sync (cta mesh slice) is unsupported")
        return SyncBarrier.GRID
    p = sync_scope(mesh)
    if p.full_cta:
        return SyncBarrier.SYNCWARP if p.count == WARP_SIZE else SyncBarrier.SYNCTHREADS
    if p.single_warp:
        return SyncBarrier.SYNCWARP
    if p.base % WARP_SIZE != 0 or p.count % WARP_SIZE != 0:
        raise expr.VerifyError(
            "T.sync: a cross-warp subset must be warp-aligned — both the base "
            "and the count must be multiples of 32"
        )
    return SyncBarrier.BAR_SYNC


__all__ = [
    "Sync",
    "SyncBarrier",
    "SyncScope",
    "sync_scope",
    "classify",
    "WARP_SIZE",
]


access.register_relation(Sync)(access.identity_relations)
