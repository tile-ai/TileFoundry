"""Physical placement, semantic aliases, and call-boundary proofs."""

from pathlib import Path
from unittest.mock import patch

import isl
import pytest

from tilefoundry.analysis import MemoryMetadata, analyze
from tilefoundry.analysis.access import AliasKind, SSAValue
from tilefoundry.analysis.allocation import AllocationSummary, address_alias, covers_result, root
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError, TirBuilder
from tilefoundry.schedule import finalize
from tilefoundry.visitor_registry.verify import verify_module


def test_union_coverage_does_not_prove_complete_writes_per_iteration() -> None:
    """Complementary partial writes cannot justify destructive result aliasing."""
    outputs = isl.map(
        "{ [iteration, element] -> [element] : "
        "0 <= iteration <= 1 and "
        "((iteration = 0 and 0 <= element <= 1) or "
        "(iteration = 1 and 2 <= element <= 3)) }"
    )
    result_box = isl.set("{ [element] : 0 <= element <= 3 }")
    iterations = isl.set("{ [iteration] : 0 <= iteration <= 1 }")
    full_result = isl.set(
        "{ [iteration, element] : "
        "0 <= iteration <= 1 and 0 <= element <= 3 }"
    )

    assert outputs.range().is_equal(result_box)
    assert not covers_result(outputs.domain(), full_result, outputs, iterations, result_box)


@pytest.mark.parametrize("later_use", [None, "callee", "caller"])
@pytest.mark.parametrize("nested", [False, True])
def test_insert_slice_and_retained_call_keep_parameter_storage(tmp_path, later_use, nested):
    source = (
        Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    ).read_text()
    if later_use == "callee":
        source = source.replace(
            "return tf.insert_slice(out, row, (1, 0))",
            "updated = tf.insert_slice(out, row, (1, 0))\n                return updated + out",
        )
    elif later_use == "caller":
        source = source.replace("return update(out, x)", "return update(out, x), out")
    if nested:
        source = source.replace(
            "    @func\n    def gemm",
            '    @func\n    def relay(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):\n'
            "        return update(out, x)\n\n    @func\n    def gemm",
        )
        source = source.replace("return update(out, x), out", "return relay(out, x), out")
        source = source.replace("return update(out, x)  #", "return relay(out, x)  #")
    path = tmp_path / "update.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, Function)
    )
    callee = call.target
    if nested:
        callee = next(
            value.target
            for value in collect_exprs(callee.body)
            if isinstance(value, Call) and isinstance(value.target, Function)
        )
    insert = next(
        value
        for value in collect_exprs(callee.body)
        if isinstance(value, Call) and isinstance(value.target, InsertSlice)
    )
    inserted = get_metadata(insert, MemoryMetadata)
    called = get_metadata(call, MemoryMetadata)
    proofs = {value.value: value for value in get_metadata(callee, AllocationSummary).values}
    callers = {
        value.value: value for value in get_metadata(result.function, AllocationSummary).values
    }
    if later_use == "callee":
        assert not proofs[SSAValue(insert)].aliases
    else:
        link = next(
            alias
            for alias in proofs[SSAValue(insert)].aliases
            if alias.kind is AliasKind.INSERT_SLICE
        )
        assert link.target.expr is insert.args[0]
        assert link.relation.relation.is_identity()
        composed = address_alias(proofs[SSAValue(insert)], proofs)
        assert composed.target.expr is callee.params[0]
        assert composed.relation.relation.is_identity()
        assert inserted.buffer_bytes is None and inserted.offsets == ()
    if later_use is not None:
        assert not callers[SSAValue(call)].aliases
        assert called.offsets
    else:
        composed = address_alias(callers[SSAValue(call)], callers)
        assert composed.target.expr is result.function.params[0]
        assert composed.relation.relation.is_identity()
        assert called.buffer_bytes is None and called.offsets == ()
    if later_use == "caller":
        with patch.object(
            TirBuilder, "transfer", autospec=True, side_effect=TirBuilder.transfer
        ) as copy:
            with pytest.raises(LoweringError, match="cannot be placed"):
                finalize(module)
        copy.assert_not_called()


def test_retained_update_translates_an_actual_window(tmp_path):
    source = (
        Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    ).read_text()
    source = source.replace("def gemm(out: Tensor[(2, 4)", "def gemm(out: Tensor[(3, 4)")
    source = source.replace("return update(out, x)", "return update(out[1:3, :], x)")
    path = tmp_path / "window.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, Function)
    )
    memory = get_metadata(call, MemoryMetadata)
    values = {
        value.value: value for value in get_metadata(result.function, AllocationSummary).values
    }
    composed = address_alias(values[SSAValue(call)], values)
    assert composed.target.expr is result.function.params[0]
    expected = isl.map("{ [i, j] -> [i + 1, j] : 0 <= i < 2 and 0 <= j < 4 }")
    assert composed.relation.relation.is_equal(expected)
    assert memory.buffer_bytes is None and memory.offsets == ()


def test_insert_slice_keeps_its_address_through_sequential_carry_loops(tmp_path):
    source = """from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget
@module(entry='gemm', target=CudaTarget('nvidia.h200_sxm'),
        topologies=(Topology('cta', 1), Topology('thread', 1)))
class SequentialCarries:
    @func
    def gemm(x: Tensor[(2, 4), 'f32'], y: Tensor[(2, 4), 'f32'], z: Tensor[(2, 4), 'f32']):
        with Mesh(('cta',), layout=(1,), names=('block',)) as _cta:
            with Mesh(('thread',), layout=(1,), names=('lane',)) as _thread:
                out = tf.zeros(Tensor[(6, 4), 'f32'])
                for j in range(2):
                    out = tf.insert_slice(out, x[j:j + 1, :], (j, 0))
                for j in range(2):
                    out = tf.insert_slice(out, y[j:j + 1, :], (2 + j, 0))
                    out = tf.insert_slice(out, z[j:j + 1, :], (4 + j, 0))
                return out
"""
    path = tmp_path / "sequential.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    inserts = [
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, InsertSlice)
    ]
    assert len(inserts) == 3
    values = {
        value.value: value for value in get_metadata(result.function, AllocationSummary).values
    }
    for insert in inserts:
        memory = get_metadata(insert, MemoryMetadata)
        link = next(
            alias
            for alias in values[SSAValue(insert)].aliases
            if alias.kind is AliasKind.INSERT_SLICE
        )
        assert link.relation.relation.is_identity()
        assert link.target.expr is insert.args[0]
        assert memory.buffer_bytes is None and memory.offsets == ()
    verify_module(finalize(module))


@pytest.mark.parametrize("returned", ["out", "out[1:2, :]"])
def test_read_only_call_return_keeps_storage_despite_later_actual_use(tmp_path, returned):
    source = (
        Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    ).read_text()
    source = source.replace("return tf.insert_slice(out, row, (1, 0))", f"return {returned}")
    source = source.replace("return update(out, x)", "return update(out, x), out")
    path = tmp_path / "readonly.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, Function)
    )
    memory = get_metadata(call, MemoryMetadata)
    values = {
        value.value: value for value in get_metadata(result.function, AllocationSummary).values
    }
    composed = address_alias(values[SSAValue(call)], values)
    assert composed.target.expr is result.function.params[0]
    expected = (
        isl.map("{ [i, j] -> [i, j] : 0 <= i < 2 and 0 <= j < 4 }")
        if returned == "out"
        else isl.map("{ [i, j] -> [i + 1, j] : i = 0 and 0 <= j < 4 }")
    )
    assert composed.relation.relation.is_equal(expected)
    assert memory.buffer_bytes is None and memory.offsets == ()


@pytest.mark.parametrize("later_use", [False, True])
def test_retained_elementwise_return_overwrites_only_a_dead_actual(tmp_path, later_use):
    source = """from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget
@module(entry='gemm', target=CudaTarget('nvidia.h200_sxm'),
        topologies=(Topology('cta', 1), Topology('thread', 1)))
class FreshReturn:
    @func
    def doubled(x: Tensor[(2, 4), 'f32']):
        return x + x
    @func
    def gemm():
        with Mesh(('cta',), layout=(1,), names=('block',)) as _cta:
            a = tf.zeros(Tensor[(2, 4), 'f32'])
            return doubled(a)
"""
    if later_use:
        source = source.replace("return doubled(a)", "return doubled(a), a")
    path = tmp_path / "fresh.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(value for value in collect_exprs(result.function.body)
                if isinstance(value, Call) and isinstance(value.target, Function))
    actual = get_metadata(call.args[0], MemoryMetadata)
    returned = get_metadata(call, MemoryMetadata)
    assert actual.offsets and returned.offsets
    assert (returned.offsets == actual.offsets) is (not later_use)
    assert returned.buffer_bytes == actual.buffer_bytes == 32
    summary = get_metadata(result.function, AllocationSummary)
    values = {value.value: value for value in summary.values}
    assert (root(values[SSAValue(call)], values) is
            root(values[SSAValue(call.args[0])], values)) is (not later_use)
