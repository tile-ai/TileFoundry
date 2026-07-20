"""CUDA emitter handler autodiscovery + shared codegen helpers.

Importing this module loads every registered per-Op emitter under ``cuda/tir/``
so its ``@register_codegen_cuda`` handler is active before codegen runs, and
exposes the launch-config / ABI helpers shared by the split-pipeline emitters.
"""
from __future__ import annotations

# Trigger emitter registration via autodiscovery.
import importlib
import logging
import os
import pkgutil

from tilefoundry.ir.core import Call
from tilefoundry.ir.tir.stmts import LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.runtime.module import ParamABI

_log = logging.getLogger(__name__)
_tir_path = os.path.dirname(__file__)

def _discover(subdir: str, prefix: str) -> None:
    full = os.path.join(_tir_path, subdir)
    if not os.path.isdir(full):
        return
    for _finder, _name, _ispkg in pkgutil.iter_modules([full], prefix=prefix):
        try:
            importlib.import_module(_name)
        except Exception:
            _log.debug("codegen autodiscovery: skip %s", _name, exc_info=True)

# Order matters for stmts/ subpackage — import parent first.
_discover("tir/stmts", "tilefoundry.codegen.cuda.tir.stmts.")
_discover("tir/memory", "tilefoundry.codegen.cuda.tir.memory.")
_discover("tir/nn", "tilefoundry.codegen.cuda.tir.nn.")
_discover("tir", "tilefoundry.codegen.cuda.tir.")


def _topology_shape_specializations(
    grid: tuple[int, int, int], block: tuple[int, int, int]
) -> list[dict[str, str]]:
    def _shape_args(dims: tuple[int, int, int]) -> str:
        return ", ".join(f"cute::Int<{d}>{{}}" for d in dims)

    # Emit a ``program_shape`` specialization per static program topology level
    # (cta, thread). ``warp`` is not a program topology level — it is expressed
    # inside a mesh layout, so no ``program_shape<warp>`` is emitted.
    specializations = []
    # A dynamic (launch-provided) CTA extent has no compile-time program_shape;
    # the device reads its count via the codegen-emitted ``program_dim<cta>()``
    # specialization. Emit no constexpr cta program_shape in that case.
    if grid[0] is not None:
        specializations.append(
            {
                "scope": "tilefoundry::TopologyScope::cta",
                "shape_args": _shape_args(grid),
            }
        )
    specializations.append(
        {
            "scope": "tilefoundry::TopologyScope::thread",
            "shape_args": _shape_args(block),
        }
    )
    return specializations


def _output_count_from_fn(fn) -> int:
    """Read output_count from the lowered PrimFunction metadata.

    The HIR-to-TIR lowering pass records output_count on the PrimFunction so
    codegen can pass it through to CallableType without guessing.
    """
    return getattr(fn, "output_count", 1)


def _param_abi(var) -> ParamABI:
    ty = var.type
    assert isinstance(ty, TensorType), f"PrimFunction param {var.name!r} must be TensorType"

    def _abi_dim(s):
        static = static_dim_value(s)
        if static is not None:
            return static
        # Dynamic dim (e.g. DimVar) — host wrapper resolves the real
        # extent from the runtime tensor; the ABI shape entry stays
        # symbolic so the launcher can detect "this axis is runtime".
        return -1

    return ParamABI(
        name=var.name,
        dtype=ty.dtype.name,
        shape=tuple(_abi_dim(s) for s in ty.shape),
        storage=ty.storage,
    )


def _derive_launch_config(
    body: Sequential,
) -> tuple[tuple[int | None, int, int], tuple[int, int, int]]:
    """Derive ``(grid, block)`` dims from the mesh topologies the kernel
    body opens via ``MeshScope``.

    Walk every ``MeshScope`` reached from *body* and accumulate:

    - ``grid.x`` = product of sizes for all ``Topology(name='cta', ...)``
      entries (every cta topology axis lives at the CTA / grid level)
    - ``block.x`` = product of sizes for the ``thread`` topology — the
      per-thread launch grain inside each CTA. (``warp`` is a mesh layout
      axis, not a program topology level, so it is not a launch contributor.)

    A launch-provided (dynamic) cta extent is reported as ``grid.x = None``;
    callers that need a static grid check for it. ``block.y`` / ``block.z`` /
    ``grid.y`` / ``grid.z`` stay at 1 — there is no user-visible convention for
    them yet.
    """
    grid_x = 1
    block_x = 1
    cta_dynamic = False

    def _topo_dims(mesh) -> tuple[int, int]:
        """Return ``(grid_size_contribution, block_size_contribution)``
        for *mesh*'s full topology list."""
        nonlocal cta_dynamic
        g = 1
        b = 1
        for t in mesh.topologies or (mesh.topology,):
            if t is None:
                continue
            tname = t.name if hasattr(t, "name") else str(t)
            size = getattr(t, "size", 1)
            if not isinstance(size, int):
                if tname == "cta":
                    # Launch-provided (dynamic) CTA extent: the grid is supplied
                    # by the host launch and read on device via
                    # ``program_dim<cta>()``. Report it as a dynamic grid
                    # (``grid.x = None``); a caller that needs a static grid
                    # checks for ``None`` and errors at its own site.
                    cta_dynamic = True
                    continue
                raise ValueError(
                    f"_derive_launch_config: topology {tname!r} has a "
                    f"dynamic/scalar extent ({size!r}) that cannot be converted "
                    f"to a static launch config; only a 'cta' level may be "
                    f"launch-provided"
                )
            tsize = size
            if tname == "cta":
                g *= tsize
            else:
                b *= tsize
        return g, b

    def _harvest_from_layout(layout) -> None:
        nonlocal grid_x, block_x
        if isinstance(layout, ShardLayout):
            g, b = _topo_dims(layout.mesh)
            grid_x = max(grid_x, g)
            block_x = max(block_x, b)

    def _walk_expr(e):
        if isinstance(e, Call):
            _harvest_from_layout(getattr(e.type, "layout", None))
            for a in e.args:
                _walk_expr(a)
        else:
            _harvest_from_layout(getattr(getattr(e, "type", None), "layout", None))

    def walk(stmt) -> None:
        nonlocal grid_x, block_x
        if isinstance(stmt, MeshScope):
            g, b = _topo_dims(stmt.mesh)
            # Use max so nested MeshScopes (e.g. cta + thread sequence)
            # land at the union footprint rather than overwriting.
            grid_x = max(grid_x, g)
            block_x = max(block_x, b)
            walk(stmt.body)
        elif isinstance(stmt, Sequential):
            for s in stmt.body:
                walk(s)
        elif isinstance(stmt, LetStmt):
            # Inspect the bound value's type — Reshard / sharded ops
            # carry ``ShardLayout`` here even when no MeshScope is
            # present (rmsnorm path).
            if hasattr(stmt, "value"):
                _walk_expr(stmt.value)
            if hasattr(stmt, "var") and getattr(stmt.var, "type", None) is not None:
                _harvest_from_layout(getattr(stmt.var.type, "layout", None))
            walk(stmt.body)

    walk(body)
    grid = (None, 1, 1) if cta_dynamic else (grid_x, 1, 1)
    return grid, (block_x, 1, 1)
