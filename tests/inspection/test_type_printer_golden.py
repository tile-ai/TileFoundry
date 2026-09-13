from tilefoundry.inspection import PythonTypePrinter
from tilefoundry.inspection.print_context import HirPrintContext, TirPrintContext
from tilefoundry.inspection.printer_base import PythonPrinter
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind


def test_hir_and_tir_share_type_value_text() -> None:
    mesh = Mesh((Topology("thread", 4),), Layout((4,), (1,)), names=("lane",))
    value = TensorType(
        shape=(8,),
        dtype=DType.f32,
        layout=ShardLayout(Layout((8,), (1,)), (S(0),), mesh),
        storage=StorageKind.GMEM,
    )
    printer = PythonPrinter()
    hir = printer.type_printer.render(value, HirPrintContext({id(mesh): "m"}))
    tir_ctx = TirPrintContext()
    tir_ctx.push_mesh(mesh, "m")
    tir = printer.type_printer.render(value, tir_ctx)
    assert hir == tir == 'Tensor[(8,), "f32", (8 @ m.lane,)]'


def test_type_functor_is_the_shared_dispatch_implementation() -> None:
    assert isinstance(PythonPrinter().type_printer, PythonTypePrinter)
