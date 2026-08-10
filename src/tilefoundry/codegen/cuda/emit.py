"""CUDA emitter handler autodiscovery + shared codegen helpers.

Importing this module loads every registered per-Op emitter under ``cuda/tir/``
so its ``@register_codegen_cuda`` handler is active before codegen runs, and
exposes the launch-config helper shared by the split-pipeline emitters. Param
ABI derivation lives in ``runtime.function.param_abi_of`` (shared with
``entry_abi_of``), not here.
"""
from __future__ import annotations

import importlib
import logging
import os
import pkgutil

from tilefoundry.ir.core import Call
from tilefoundry.ir.tir.stmts import LetStmt, MeshScope, Sequential
from tilefoundry.ir.types.shard.shard_layout import ShardLayout

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


_discover("tir/stmts", "tilefoundry.codegen.cuda.tir.stmts.")
_discover("tir/memory", "tilefoundry.codegen.cuda.tir.memory.")
_discover("tir/nn", "tilefoundry.codegen.cuda.tir.nn.")
_discover("tir", "tilefoundry.codegen.cuda.tir.")


def _topology_shape_specializations(
    grid: tuple[int, int, int], block: tuple[int, int, int]
) -> list[dict[str, str]]:
    def _shape_args(dims: tuple[int, int, int]) -> str:
        return ", ".join(f"cute::Int<{d}>{{}}" for d in dims)




    specializations = []



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
    codegen can pass it through to EntryABI without guessing.
    """
    return getattr(fn, "output_count", 1)


def _derive_launch_config(
    body: Sequential,
) -> tuple[tuple[int | None, int, int], tuple[int, int, int]]:
    """Derive grid and block dimensions from body mesh topologies.

    CTA topology sizes multiply into ``grid.x`` and thread sizes into
    ``block.x``; warp axes do not contribute. A launch-provided CTA extent
    yields ``grid.x = None`` for static callers to reject. Other dimensions
    remain one until a public convention exists.
    """
    grid_x = 1
    block_x = 1
    cta_dynamic = False

    def _topo_dims(mesh) -> tuple[int, int]:
        """Return grid and block contributions for *mesh*.

        Return ``(grid_size_contribution, block_size_contribution)``
        for *mesh*'s full topology list.
        """
        nonlocal cta_dynamic
        g = 1
        b = 1
        for t in mesh.topologies:
            tname = t.name
            size = t.size
            if not isinstance(size, int):
                if tname == "cta":





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
        match stmt:
            case MeshScope():
                g, b = _topo_dims(stmt.mesh)


                grid_x = max(grid_x, g)
                block_x = max(block_x, b)
                walk(stmt.body)
            case Sequential():
                for s in stmt.body:
                    walk(s)
            case LetStmt():



                if hasattr(stmt, "value"):
                    _walk_expr(stmt.value)
                if hasattr(stmt, "var") and getattr(stmt.var, "type", None) is not None:
                    _harvest_from_layout(getattr(stmt.var.type, "layout", None))
                walk(stmt.body)

    walk(body)
    grid = (None, 1, 1) if cta_dynamic else (grid_x, 1, 1)
    return grid, (block_x, 1, 1)
