"""Give a device-only module the CPU entry that calls into it.

The entry is synthesized, never retargeted: which target a function records is
what its author wrote, and a pass that rewrites it decides in the wrong place.
Launch geometry is settled here and written into the ``Launch``, so nothing
downstream derives it a second time. A launch-provided (dynamic) grid is not
supported by the synthesized entry.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.tir.launch import launch_call
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import LetStmt, MeshScope, Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.ir.visitor import ExprWalker
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.target import CpuTarget, CudaTarget

_DEFAULT_ENTRY_NAME = "main"


def _fresh_entry_name(taken: set[str]) -> str:
    if _DEFAULT_ENTRY_NAME not in taken:
        return _DEFAULT_ENTRY_NAME
    candidate = f"_tilefoundry_{_DEFAULT_ENTRY_NAME}"
    i = 0
    while candidate in taken:
        i += 1
        candidate = f"_tilefoundry_{_DEFAULT_ENTRY_NAME}_{i}"
    return candidate


def _device_callee(module: Module) -> PrimFunction:
    """The one device function a synthesized entry can be written against.

    Refuses rather than guesses: a second device function would leave the
    entry's parameters undecided, and a CPU function that is not the entry
    means the author meant some other function to be called first.
    """
    if any(isinstance(fn.target, CpuTarget) for fn in module.functions):
        raise ValueError(
            "insert_default_host_entry: module has a CPU function that is not "
            "the entry; refusing to guess the host entry"
        )
    device_fns = [
        fn
        for fn in module.functions
        if isinstance(fn, PrimFunction) and isinstance(fn.target, CudaTarget)
    ]
    if len(device_fns) != 1:
        raise ValueError(
            f"insert_default_host_entry: expected exactly one CUDA device "
            f"prim_function and no CPU entry, found {len(device_fns)} device "
            f"functions"
        )
    return device_fns[0]


def _launch_config_from_body(
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
                    f"_launch_config_from_body: topology {tname!r} has a "
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

    class _LayoutVisitor(ExprWalker[None]):
        def _record(self, expr) -> None:
            _harvest_from_layout(getattr(getattr(expr, "type", None), "layout", None))

        def visit_Call(self, expr: Call, ctx=None) -> None:
            self._record(expr)
            self.visit_operands(expr, ctx)

        def visit_Tuple(self, expr: Tuple, ctx=None) -> None:
            self._record(expr)

        def visit_LoopRegion(self, expr: LoopRegion, ctx=None) -> None:
            self._record(expr)

        def visit_Var(self, expr: Var, ctx=None) -> None:
            self._record(expr)

        def visit_Constant(self, expr: Constant, ctx=None) -> None:
            self._record(expr)

        def visit_SymbolRef(self, expr: SymbolRef, ctx=None) -> None:
            self._record(expr)

        def visit_ShapeOf(self, expr: ShapeOf, ctx=None) -> None:
            self._record(expr)

        def default_visit(self, expr, ctx=None) -> None:
            self._record(expr)

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
                    _LayoutVisitor().visit(stmt.value)
                if hasattr(stmt, "var") and getattr(stmt.var, "type", None) is not None:
                    _harvest_from_layout(getattr(stmt.var.type, "layout", None))
                walk(stmt.body)

    walk(body)
    grid = (None, 1, 1) if cta_dynamic else (grid_x, 1, 1)
    return grid, (block_x, 1, 1)


def _launch_config_of(
    fn: PrimFunction,
) -> tuple[tuple[int | None, int, int], tuple[int, int, int]]:
    """The geometry one launch of *fn* runs at.

    A specialization prototype has variants and no body of its own, so the
    variants state it -- and they have to agree, because a translation unit
    states its program dimensions once for every kernel it holds.
    """
    if fn.variants and not fn.body.body:
        by_variant = {
            variant.name: _launch_config_from_body(variant.body) for variant in fn.variants
        }
        distinct = set(by_variant.values())
        if len(distinct) > 1:
            raise ValueError(
                f"insert_default_host_entry: prototype {fn.name!r} has variants "
                f"launching at different geometries ({by_variant}); one "
                f"translation unit runs every kernel in it at one geometry"
            )
        return distinct.pop()
    return _launch_config_from_body(fn.body)


def insert_default_host_entry(module: Module) -> Module:
    """Return a module whose entry is host-callable, synthesizing one if it is not.

    A CPU entry passes through. Otherwise the device entry gains a CPU entry
    that launches it with its own parameters. Whether that call reads as one
    launch or as a dispatch over variants is the callee's shape to state, so
    a prototype and a lone kernel take the same route here.
    """
    if isinstance(module.entry_function().target, CpuTarget):
        return module

    device_fn = _device_callee(module)
    entry_params = tuple(Var(type=p.type, name=p.name) for p in device_fn.params)
    grid, block = _launch_config_of(device_fn)
    if grid[0] is None:
        raise ValueError(
            f"insert_default_host_entry: device function {device_fn.name!r} has "
            f"a launch-provided (dynamic) CTA extent; the implicit host entry "
            f"cannot derive its grid — launch it from an explicit host entry"
        )
    launch = launch_call(device_fn, entry_params, grid, block)
    name = _fresh_entry_name({fn.name for fn in module.functions})
    entry = PrimFunction(
        name=name,
        params=entry_params,
        body=Sequential(body=(launch,)),
        output_count=device_fn.output_count,
        target=CpuTarget(),
    )
    return replace(module, functions=(*module.functions, entry), entry=name)


class InsertHostEntryPass(ModulePass):
    """Give a device-only module a host-callable entry.

    The only pass a TIR module needs: everything else it is made of was
    settled before it became TIR.
    """

    name: str = "insert_host_entry"
    requires: tuple[str, ...] = ()

    def run(self, module: Module) -> Module:
        return insert_default_host_entry(module)


__all__ = ["InsertHostEntryPass", "insert_default_host_entry"]
