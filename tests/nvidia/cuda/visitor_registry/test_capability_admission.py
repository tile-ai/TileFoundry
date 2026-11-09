"""Admission refuses an instruction the target's architecture does not list.

See [target §4](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/target.md#4-cudatarget).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.nvidia.cuda.ops.tir.targets import retarget
from tilefoundry import cli, target, verifier
from tilefoundry.interfaces import schedule, verify
from tilefoundry.ir import core, expr, hir, types
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda.ir.tir import memory, nn, sync

SM89 = cuda.Target("nvidia.rtx_6000_ada")
FIXTURES = Path(__file__).parents[1] / "fixtures"
MM_HANDWRITTEN = f"{FIXTURES / 'tir' / 'mma.py'}:MmHandwritten"

_SMEM_BAR = types.TensorType.from_shape((1,), expr.DType.from_name("i64"), storage="smem")
_PHASE = types.TensorType.from_shape((), expr.DType.from_name("i32"), storage="rmem")
_GMEM_TILE = types.TensorType.from_shape((32,), expr.DType.f32, storage="gmem")
_SMEM_TILE = types.TensorType.from_shape((32,), expr.DType.f32, storage="smem")


def _one_call(op, *type_list, level: str = "thread", size: int = 32) -> core.PrimFunction:
    """A function whose whole body is one call of *op* on a one-level mesh."""
    args = tuple(expr.Var(type=t, name=f"a{i}") for i, t in enumerate(type_list))
    mesh = types.Mesh((types.Topology(level, size),), types.Layout((size,), (1,)))
    return core.PrimFunction(
        target=cuda.Target("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=core.Sequential(
            body=(
                core.MeshScope(
                    mesh=mesh,
                    binding=expr.Var(name="thread", type=mesh),
                    body=core.Sequential(body=(core.Evaluate(callable=op, args=args),)),
                ),
                core.Return(),
            )
        ),
    )


def _verify(function: core.PrimFunction, tgt: target.Target) -> None:
    """Verify *function* as the entry of a module on *target*."""
    module = core.Module(
        name="admission", functions=(function,), entry=function.name, target=tgt
    )
    verifier.verify_prim_function(function, module_fns=module)


_REFUSED = [
    pytest.param(
        _one_call(sync.MBarrierArriveExpectTx(tx_bytes=16), _SMEM_BAR),
        "mbarrier.expect_tx",
        id="mbarrier_arrive_expect_tx",
    ),
    pytest.param(
        _one_call(sync.MBarrierWaitParity(), _SMEM_BAR, _PHASE),
        "mbarrier.try_wait",
        id="mbarrier_wait_parity",
    ),
    pytest.param(
        _one_call(memory.CopyAsyncBulk(), _GMEM_TILE, _SMEM_TILE, _SMEM_BAR),
        "cp.async.bulk",
        id="copy_async_bulk",
    ),
    pytest.param(
        f"{FIXTURES / 'schedule' / 'tir' / 'wgmma_b_nk.py'}:WgmmaBNk",
        "wgmma.mma_async",
        id="wgmma_atom",
    ),
]


@pytest.mark.parametrize(("function", "capability"), _REFUSED)
def test_missing_capability_refused_by_name(
    function: core.PrimFunction | str, capability: str
) -> None:
    """On sm_89 the call is refused, naming the capability it needs."""
    if isinstance(function, str):
        function = cli.load_authored_ir(function).entry_function()
    with pytest.raises(
        expr.VerifyError, match=f"needs {capability}, which nvidia.rtx_6000_ada does not list"
    ):
        _verify(function, SM89)


def test_selected_atom_only_checked() -> None:
    """The SM80 atom is admitted on sm_89, although the same op also declares WGMMA."""
    module = retarget(cli.load_authored_ir(MM_HANDWRITTEN), SM89)
    for function in module.functions:
        verifier.verify_prim_function(function, module_fns=module)


class _MmaSite(core.StmtVisitor[None]):
    """The one ``TiledMma`` call of a function, with the meshes enclosing it."""

    def __init__(self) -> None:
        self.meshes: tuple = ()
        self.site = None

    def visit_MeshScope(self, stmt: core.MeshScope) -> None:
        previous = self.meshes
        self.meshes = (*previous, stmt.mesh)
        try:
            self.visit(stmt.body)
        finally:
            self.meshes = previous

    def visit_Evaluate(self, stmt: core.Evaluate) -> None:
        if isinstance(stmt.callable, nn.TiledMma):
            self.site = (stmt.args, self.meshes)


def test_open_wgmma_refused_on_sm89() -> None:
    """Schedule matching names the missing capability for an atom given as a class."""
    device = next(
        fn for fn in cli.load_authored_ir(MM_HANDWRITTEN).functions if fn.name == "mm_device"
    )
    site = _MmaSite()
    site.visit(device.body)
    args, meshes = site.site
    mesh = types.Mesh.from_meshes(*meshes)
    frame = verify.Context(current_mesh=mesh, mesh_scope=meshes, target=SM89)
    operands = tuple(arg.type for arg in args)
    with pytest.raises(
        schedule.RefusalError, match="needs wgmma.mma_async, which nvidia.rtx_6000_ada does not list"
    ):
        hir.schedule.match_instruction(nn.TiledMma, {"atom": nn.Wgmma}, operands, mesh, frame=frame)
