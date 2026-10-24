"""TIR codegen emits dynamic bounds and long statement sequences."""

import tilefoundry_nvidia.cuda.codegen  # noqa: F401
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import For, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import DType, Layout, Mesh, StorageKind, TensorType, Topology
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext


def test_loop_bounds_emit_variables() -> None:
    scalar = TensorType.scalar(DType.i64)
    variable = Var(type=scalar, name="n")
    loop = For(
        induction_var=Var(type=scalar, name="i"),
        start=Constant(type=scalar, value=0),
        stop=variable,
        step=Constant(type=scalar, value=1),
        body=Sequential(()),
    )
    context = CudaCodegenContext()
    context.emit_node(loop)
    assert "i_1 < n_2" in context.source()


def test_long_sequence_emits_statements() -> None:
    type_ = TensorType.scalar(DType.f32, storage=StorageKind.RMEM)
    statements = tuple(
        LetStmt(
            Var(name=f"value_{i}", type=type_), Call(AllocTensor(tensor_type=type_), (), type=type_)
        )
        for i in range(1000)
    )
    mesh = Mesh((Topology("thread", 32),), layout=Layout((32,), (1,)))
    binding = Var(name="thread", type=TensorType.scalar(DType.i64, storage=StorageKind.RMEM))
    function = PrimFunction(
        name="sequence",
        params=(),
        body=Sequential((MeshScope(mesh, binding, Sequential(statements)),)),
        target=CudaTarget("nvidia.h200_sxm"),
        output_count=0,
    )
    module = Module(name="LongSequence", functions=(function,))
    printed = as_script(module)
    assert printed.count("T.alloc_tensor(") == 1000
    verify_module(module)
    context = CudaCodegenContext(target=module.functions[0].target)
    context.emit_node(module.functions[0].body)
    assert context.source()
