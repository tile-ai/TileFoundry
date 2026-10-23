"""Admission refuses an instruction the target's architecture does not list.

See [target §4](targets/cuda/tilefoundry_cuda/docs/spec/target.md#4-cudatarget).
"""

from __future__ import annotations

import pytest

from tests.cuda.ops.tir.targets import retarget
from tests.fixtures.schedule.tir.wgmma_b_nk import gemm as wgmma_gemm
from tests.fixtures.tir.mma import MmHandwritten
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.schedule import match_instruction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.ir.types import DType, Layout, Mesh, Topology, make_tensor_type
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.target import Target
from tilefoundry.visitor_registry.contexts import VerifyContext
from tilefoundry.visitor_registry.instruction import ScheduleRefusal
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_cuda import CudaTarget
from tilefoundry_cuda.ir.tir.memory.copy_async_bulk import CopyAsyncBulk
from tilefoundry_cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_cuda.ir.tir.nn.wgmma import Wgmma
from tilefoundry_cuda.ir.tir.sync.mbarrier import MBarrierArriveExpectTx, MBarrierWaitParity

SM89 = CudaTarget("nvidia.rtx_6000_ada")

_SMEM_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="smem")
_PHASE = make_tensor_type((), DType.from_name("i32"), storage="rmem")
_GMEM_TILE = make_tensor_type((32,), DType.f32, storage="gmem")
_SMEM_TILE = make_tensor_type((32,), DType.f32, storage="smem")


def _one_call(op, *types, level: str = "thread", size: int = 32) -> PrimFunction:
    """A function whose whole body is one call of *op* on a one-level mesh."""
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    mesh = Mesh((Topology(level, size),), Layout((size,), (1,)))
    return PrimFunction(
        target=CudaTarget("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=Sequential(
            body=(
                MeshScope(
                    mesh=mesh,
                    binding=Var(name="thread", type=mesh),
                    body=Sequential(body=(Evaluate(callable=op, args=args),)),
                ),
                Return(),
            )
        ),
    )


def _verify(function: PrimFunction, target: Target) -> None:
    """Verify *function* as the entry of a module on *target*."""
    module = Module(
        name="admission", functions=(function,), entry=function.name, target=target
    )
    verify_prim_function(function, module_fns=module)


_REFUSED = [
    pytest.param(
        _one_call(MBarrierArriveExpectTx(tx_bytes=16), _SMEM_BAR),
        "mbarrier.expect_tx",
        id="mbarrier_arrive_expect_tx",
    ),
    pytest.param(
        _one_call(MBarrierWaitParity(), _SMEM_BAR, _PHASE),
        "mbarrier.try_wait",
        id="mbarrier_wait_parity",
    ),
    pytest.param(
        _one_call(CopyAsyncBulk(), _GMEM_TILE, _SMEM_TILE, _SMEM_BAR),
        "cp.async.bulk",
        id="copy_async_bulk",
    ),
    pytest.param(wgmma_gemm, "wgmma.mma_async", id="wgmma_atom"),
]


@pytest.mark.parametrize(("function", "capability"), _REFUSED)
def test_an_instruction_the_architecture_lacks_is_refused_naming_its_capability(
    function: PrimFunction, capability: str
) -> None:
    """On sm_89 the call is refused, naming the capability it needs."""
    with pytest.raises(
        VerifyError, match=f"needs {capability}, which nvidia.rtx_6000_ada does not list"
    ):
        _verify(function, SM89)


def test_only_the_selected_atom_is_checked() -> None:
    """The SM80 atom is admitted on sm_89, although the same op also declares WGMMA."""
    module = retarget(MmHandwritten, SM89)
    for function in module.functions:
        verify_prim_function(function, module_fns=module)


class _MmaSite(StmtVisitor[None]):
    """The one ``TiledMma`` call of a function, with the meshes enclosing it."""

    def __init__(self) -> None:
        self.meshes: tuple = ()
        self.site = None

    def visit_MeshScope(self, stmt: MeshScope) -> None:
        previous = self.meshes
        self.meshes = (*previous, stmt.mesh)
        try:
            self.visit(stmt.body)
        finally:
            self.meshes = previous

    def visit_Evaluate(self, stmt: Evaluate) -> None:
        if isinstance(stmt.callable, TiledMma):
            self.site = (stmt.args, self.meshes)


def test_an_open_wgmma_atom_is_refused_on_sm89() -> None:
    """Schedule matching names the missing capability for an atom given as a class."""
    device = next(fn for fn in MmHandwritten.functions if fn.name == "mm_device")
    site = _MmaSite()
    site.visit(device.body)
    args, meshes = site.site
    mesh = make_mesh(*meshes)
    frame = VerifyContext(current_mesh=mesh, mesh_scope=meshes, target=SM89)
    operands = tuple(arg.type for arg in args)
    with pytest.raises(
        ScheduleRefusal, match="needs wgmma.mma_async, which nvidia.rtx_6000_ada does not list"
    ):
        match_instruction(TiledMma, {"atom": Wgmma}, operands, mesh, frame=frame)
