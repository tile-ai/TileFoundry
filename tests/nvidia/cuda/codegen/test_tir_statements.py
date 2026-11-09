"""TIR codegen emits dynamic bounds and long statement sequences."""

import pytest

from tilefoundry import inspection, verifier
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import tensor
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda import codegen


def test_loop_bounds_emit_variables() -> None:
    scalar = types.TensorType.from_scalar(expr.DType.i64)
    variable = expr.Var(type=scalar, name="n")
    loop = core.For(
        induction_var=expr.Var(type=scalar, name="i"),
        start=expr.Constant(type=scalar, value=0),
        stop=variable,
        step=expr.Constant(type=scalar, value=1),
        body=core.Sequential(()),
    )
    context = codegen.Context()
    context.emit_node(loop)
    assert "i_1 < n_2" in context.source()


def test_long_sequence_emits_statements() -> None:
    type_ = types.TensorType.from_scalar(expr.DType.f32, storage=expr.StorageKind.RMEM)
    statements = tuple(
        core.LetStmt(
            expr.Var(name=f"value_{i}", type=type_), expr.Call(memory.AllocTensor(tensor_type=type_), (), type=type_)
        )
        for i in range(1000)
    )
    mesh = types.Mesh((types.Topology("thread", 32),), layout=types.Layout((32,), (1,)))
    binding = expr.Var(name="thread", type=types.TensorType.from_scalar(expr.DType.i64, storage=expr.StorageKind.RMEM))
    function = core.PrimFunction(
        name="sequence",
        params=(),
        body=core.Sequential((core.MeshScope(mesh, binding, core.Sequential(statements)),)),
        target=cuda.Target("nvidia.h200_sxm"),
        output_count=0,
    )
    module = core.Module(name="LongSequence", functions=(function,))
    printed = inspection.as_script(module)
    assert printed.count("T.alloc_tensor(") == 1000
    verifier.verify_module(module)
    context = codegen.Context(target=module.functions[0].target)
    context.emit_node(module.functions[0].body)
    assert context.source()


@pytest.mark.parametrize("homogeneous,dynamic", [(True, False), (True, True), (False, False)])
def test_tuple_binding_emits_one_variable(homogeneous, dynamic):
    first = expr.Constant(1, type=types.TensorType.from_scalar(expr.DType.i64))
    second = expr.Constant(2 if homogeneous else 2.5, type=types.TensorType.from_scalar(expr.DType.i64 if homogeneous else expr.DType.f32))
    values = expr.Tuple((first, second), type=types.TupleType((first.type, second.type)))
    held = expr.Var("choices", type=values.type)
    index = expr.Var("index", type=first.type) if dynamic else expr.Constant(0, type=first.type)
    selected = expr.Var("selected", type=first.type)
    context = codegen.Context(target=cuda.Target("nvidia.h200_sxm"))
    context.emit_node(core.Sequential((
        core.LetStmt(held, values),
        core.LetStmt(selected, expr.Call(tensor.TupleGetItem(), (held, index), type=selected.type)),
    )))
    declaration, selection = context.source().strip().splitlines()
    name = context.name_for(held)
    if homogeneous:
        assert declaration == f"cute::array<decltype(1), 2> {name}{{1, 2}};"
    else:
        assert declaration == f"auto {name} = cute::make_tuple(1, 2.5);"
    chosen = f"{name}[{context.name_for(index)}]" if dynamic else f"cute::get<0>({name})"
    assert selection == f"auto {context.name_for(selected)} = {chosen};"
