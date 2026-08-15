"""What each parser program must parse to, and what a golden cannot say.

The golden is the whole assertion for anything the printed source shows. The
tests beside it are the ones a golden cannot carry: node identity, a target's
concrete ``Op`` class, a canonicalisation the printer renders back as the sugar
it was written as, and what a program does once it is evaluated or rebuilt.
"""

from __future__ import annotations

import ast
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests._source import import_dsl
from tests.fixtures.placed.mma_tile import MatmulModule
from tests.parser.conftest import GoldenFiles
from tests.parser.error_cases import literal_reshard_func, mesh_dims_reshard_func
from tests.parser.programs import (
    M_CTA,
    M_GPU,
    M_MULTI,
    M_STRIDED,
    PROGRAMS,
    SEQ_LEN,
    CustomParseAddSq,
    HirExpressions,
    HirGrid,
    HirModule,
    HirSharded,
    ParserProgram,
    Scaled,
    doubles_a_constant,
    returns_a_pair,
)
from tilefoundry import func, module
from tilefoundry.analysis.preflight import infer_authored_types
from tilefoundry.analysis.walk import postorder
from tilefoundry.dsl import ConstTensor, DimVarRangePat, Tensor
from tilefoundry.dsl._stub_gen import regen_stubs
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.inspection import as_script
from tilefoundry.ir.constraints import LayoutConstraint, constraint_metadata
from tilefoundry.ir.core import Call, Constant, Tuple, Var, get_metadata
from tilefoundry.ir.hir.function import Function, elaborate
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.specialize import origin_of, specialize_function
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type
from tilefoundry.ir.types.dim import DimAdd, DimVar
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.parser import hir_parser
from tilefoundry.parser.base import _ModuleCallee
from tilefoundry.parser.sugar import parse_shard_layout_sugar
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor


@pytest.mark.parametrize("program", PROGRAMS, ids=[program.name for program in PROGRAMS])
def test_a_feature_dense_program_parses_to_its_golden(
    program: ParserProgram, golden: GoldenFiles
) -> None:
    """Parsing the program yields exactly the recorded IR, printed back as source."""
    golden.check(f"{program.name}.py", as_script(program.parsed))


def test_converter_declared_before_variants_is_retained() -> None:
    seq = DimVar("converter_before_variant", 1, 8)

    @module()
    class ConverterThenVariants:
        @func
        def dispatch(
            x: Tensor[(seq,), "f32"], w: ConstTensor[(8,), "f32"]
        ) -> Tensor[(seq,), "f32"]:
            pass

        @dispatch.converter("w")
        def convert(w: ConstTensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return w

        @dispatch.specialize(DimVarRangePat("converter_before_variant", 1, 4))
        def small(x: Tensor[(seq,), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(seq,), "f32"]:
            return x

        @dispatch.specialize(DimVarRangePat("converter_before_variant", 4, 8))
        def large(x: Tensor[(seq,), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(seq,), "f32"]:
            return x

    dispatch = ConverterThenVariants.lookup("dispatch")
    assert len(dispatch.converters) == 1
    assert len(dispatch.variants) == 2


def test_annotation_sugar_lands_on_the_hand_written_layout() -> None:
    """The golden prints each annotation back as the sugar it was written as.

    What it therefore cannot show is what the sugar canonicalises *to*: the
    auto-filled C-order strides, and the ``Broadcast`` every mesh axis named in
    no ``Split`` falls back to.
    """
    expected = {
        "split_inline_and_default_broadcast": ShardLayout(
            layout=Layout((32, 2, 64), (128, 64, 1)),
            attrs=(Split(0), Split(1), Broadcast(), Broadcast()),
            mesh=M_GPU,
        ),
        "partial_brace_value_state": ShardLayout(
            layout=Layout((32, 64), (64, 1)),
            attrs=(Split(0), Broadcast(), Partial("sum"), Broadcast()),
            mesh=M_GPU,
        ),
        "multi_axis_split_with_remainder": ShardLayout(
            layout=Layout((1, 6, 32, 8), (1536, 256, 8, 1)),
            attrs=(Split(1), Split(2)),
            mesh=M_MULTI,
        ),
        "explicit_strides": ShardLayout(
            layout=Layout((12, 4), (4, 1)), attrs=(Split(0), Broadcast()), mesh=M_STRIDED
        ),
        "int_at_a_single_axis_mesh": ShardLayout(
            layout=Layout((1, 128, 64), (8192, 64, 1)), attrs=(Split(1),), mesh=M_CTA
        ),
    }
    for name, layout in expected.items():
        parsed = HirSharded.lookup(name).params[0].type
        assert parsed.storage is StorageKind.SMEM, name
        assert parsed.layout == layout, name


def test_a_reshard_split_axis_resolved_through_the_closure() -> None:
    """``32 @ cta`` keeps the un-factorised shape and defers strides to typeinfer."""
    body = HirSharded.lookup("reshard_with_a_dynamic_and_a_closure_axis").body
    assert isinstance(body, Call)
    assert body.type.shape == (1, DimVar("seq_len", 1, 4), 32, 128)
    assert any(isinstance(attr, Split) for attr in body.target.layout.attrs)
    assert body.target.layout.layout.strides is None


def test_a_symbolic_extent_on_both_sides_has_local_size_one() -> None:
    """``parse_shard_layout_sugar`` on a bare AST node, with no function around it."""
    dyn = DimVar("seq_len", 1, 4)
    cta = Mesh((Topology("cta", dyn),), Layout((dyn,), (1,)), names=("cta",))
    node = ast.parse("(1, S @ cta, 32, 128)", mode="eval").body

    actual = parse_shard_layout_sugar(node, lambda n: cta if n == "cta" else None, closure={"S": dyn})

    assert actual.layout.shape == (1, dyn, 32, 128)
    assert actual.attrs == (Split(1),)


def test_the_two_grid_spellings_share_one_loop_domain() -> None:
    """The domain fields and the induction var's storage are not printed.

    ``range`` and ``tile`` differ only in their step, which the golden shows as
    two loop headers rather than as one domain; and nothing in the source says
    the induction var is unmaterialized.
    """
    default_step = HirGrid.lookup("range_default_step").body
    assert isinstance(default_step, GridRegionExpr)
    assert (default_step.start, default_step.extent, default_step.step) == (0, 8, 1)
    assert default_step.carried_args == ()
    assert default_step.init_args == ()
    assert default_step.yield_values == ()

    with_step = HirGrid.lookup("tile_extent_step").body
    assert repr(default_step) == repr(replace(with_step, step=1))

    ranged = HirGrid.lookup("range_start_stop_step").body
    assert (ranged.start, ranged.extent, ranged.step) == (2, 8, 3)
    assert isinstance(ranged.induction_var, Var)
    assert ranged.induction_var.type.storage is StorageKind.UMAT

    assert isinstance(HirGrid.lookup("range_dim_expr_extent").body.extent, Call)
    assert HirGrid.lookup("tile_dimvar_extent").body.extent == SEQ_LEN
    assert HirGrid.lookup("inner_bindings_carry_nothing").body.carried_args == ()


def test_the_generated_tile_stub_requires_its_window_step() -> None:
    """``tile`` is positional-only at the IR level, so its stub declares no default.

    The stub is generated rather than parsed, so no program can carry this.
    """
    with tempfile.TemporaryDirectory() as directory:
        stub = ast.parse(regen_stubs(Path(directory))["tf"].read_text())

    tile_def = next(
        node for node in stub.body if isinstance(node, ast.FunctionDef) and node.name == "tile"
    )
    assert [arg.arg for arg in tile_def.args.args] == ["extent", "step"]
    assert tile_def.args.defaults == []


def test_a_carry_reuses_the_nodes_it_was_built_from() -> None:
    """Which node a carry *is* — not what it prints as.

    A rebinding yields the very node its RHS bound, the reader of the old value
    still points at the phi, and a carry initialized from a bare name adds no
    call of its own.
    """
    grid = HirGrid.lookup("carry_reads_old_and_new").body.args[0]
    assert isinstance(grid, GridRegionExpr)
    carried = {value.name: value for value in grid.carried_args}
    yielded = dict(zip((value.name for value in grid.carried_args), grid.yield_values))
    correction = yielded["o"].args[1]

    assert grid.body is yielded["m"]
    assert correction.args[0] is carried["m"]
    assert correction.args[1] is yielded["m"]

    from_parameter = HirGrid.lookup("carry_initialized_from_a_parameter")
    initialized = from_parameter.body
    assert initialized.init_args[0] is from_parameter.params[0]
    assert [expr for expr in postorder(initialized) if isinstance(expr, Call)] == [
        initialized.body
    ]

    outer = HirGrid.lookup("nested_for").body
    assert [v.name for v in outer.carried_args] == ["o"]
    assert [v.name for v in outer.yield_values[0].carried_args] == ["o"]


def test_a_where_annotation_attaches_to_the_existing_ssa_node() -> None:
    """A constraint is metadata on a node, so nothing about it is a printed line.

    Both readers of the annotated binding reach the same node, and the parser
    attaches once rather than rebuilding.
    """
    layout = constraint_metadata(HirGrid.lookup("where_on_a_binding").body).constraints[0]
    assert isinstance(layout, LayoutConstraint)
    assert repr(layout.layout.shape[0]) == "_"
    assert layout.layout.shape[1] == 16
    assert layout.bindings == (("cta", Split(1)),)

    partial = constraint_metadata(
        HirGrid.lookup("where_with_a_partial_value_state").body
    ).constraints[0]
    assert partial.bindings == (("cta", Partial("sum")),)

    for name in ("where_on_a_binding", "where_with_a_partial_value_state"):
        verify_function(HirGrid.lookup(name))


def test_a_where_layout_extent_is_read_out_of_the_globals(monkeypatch) -> None:
    """A named extent resolves through the function's globals, as an int or a DimVar.

    The subject has to be parsed inside the test to count the attachments, so
    this one stays on DSL source rather than reading a program.
    """
    attached = []
    original = hir_parser._HirBodyVisitor._attach_metadata

    def capture(expr, metadata):
        attached.append(expr)
        original(expr, metadata)

    monkeypatch.setattr(hir_parser._HirBodyVisitor, "_attach_metadata", staticmethod(capture))

    preamble = (
        "from tilefoundry.ir.types.shard import Layout, Mesh, Topology\n\n"
        'cta_mesh = Mesh((Topology("cta", 8),), Layout((8,), (1,)))\n'
    )
    source = (
        "from __future__ import annotations\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor, tf\n\n"
        f"{preamble}N = 16\n\n"
        "@func\n"
        'def candidate(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:\n'
        "    y: where(layout=(_, N @ cta)) = tf.add(x, x)\n"
        "    return y\n"
    )
    function = import_dsl(source)

    assert len(attached) == 1
    assert function.body is attached[0]
    assert constraint_metadata(attached[0]).constraints[0].layout.shape[1] == 16

    as_dim_var = import_dsl(source.replace("N = 16", 'N = DimVar("S", 1, 128)').replace(
        "from tilefoundry.dsl import Tensor, tf\n",
        "from tilefoundry.dsl import Tensor, tf\nfrom tilefoundry.ir.types.dim import DimVar\n",
    ))
    extent = constraint_metadata(as_dim_var.body).constraints[0].layout.shape[1]
    assert isinstance(extent, DimVar) and extent.name == "S"


def test_umat_is_an_accepted_surface_storage() -> None:
    """An explicit ``umat`` annotation preserves unresolved residency.

    Both spellings live in ``HirExpressions``, whose module declares meshes —
    which is the point: the third annotation slot holds a storage name or an
    empty layout, and a mesh being in scope does not make either one layout
    sugar. The golden shows the storage; that it resolved to ``UMAT`` rather
    than defaulting to ``GMEM`` is what is asserted here.
    """
    for name in ("unmaterialized_surface_storage", "storage_without_a_layout_slot"):
        parsed = HirExpressions.lookup(name)
        assert parsed.params[0].type.storage is StorageKind.UMAT, name
        assert parsed.params[0].type.layout is None, name
        assert parsed.return_type.storage is StorageKind.GMEM, name


def test_a_value_literal_takes_the_dtype_of_the_operand_it_meets() -> None:
    """The golden prints ``1e-06`` without saying what dtype it carries."""
    for name in ("literal_meets_bf16", "captured_float_meets_bf16"):
        fn = HirExpressions.lookup(name)
        assert fn.body.args[1].type.dtype == DType.bf16, name
        assert fn.body.type.dtype == DType.bf16, name


def test_a_dim_expression_stays_resolvable_arithmetic() -> None:
    """The golden prints the dim expression; that it still evaluates is separate."""
    padded = HirExpressions.lookup("dim_from_a_static_call").body.target.shape[0]
    assert resolve_dim(padded, {"CTX_LEN": 128}) == 128
    assert resolve_dim(padded, {"CTX_LEN": 130}) == 256

    anchored = HirExpressions.lookup("dim_anchored_twice")
    assert isinstance(anchored.params[1].type.shape[0], Call)
    verify_function(anchored)


def test_a_literal_tuple_return_folds_to_a_tuple_typed_body() -> None:
    """The caller's golden names the callee; the callee's own body is not in it."""
    assert isinstance(returns_a_pair.body, Tuple)
    assert len(returns_a_pair.body.elements) == 2
    assert isinstance(returns_a_pair.return_type, TupleType)
    assert all(field.dtype == DType.f32 for field in returns_a_pair.return_type.fields)


def test_a_custom_op_call_resolves_to_the_registered_op_class() -> None:
    """``as_script`` prints the call by name; the target's class is not visible there."""
    body = HirModule.lookup("uses_a_custom_op").body
    assert isinstance(body, Call)
    assert isinstance(body.target, CustomParseAddSq)


def test_each_child_binding_calls_its_own_attached_entry() -> None:
    """Two bindings of one Module print identically and are still different objects."""
    children = {child.name: child for child in HirModule.modules}
    assert set(children) == {"leaf", "first", "second", "mlp", "deep", "variant_leaf"}

    call = HirModule.lookup("calls_a_child").body
    assert isinstance(call.target, Function)
    assert call.target is children["leaf"].entry_function()
    assert get_metadata(call, _ModuleCallee) is None

    left, right = HirModule.lookup("two_bindings_under_a_reshard").body.args
    assert left.target is not right.target
    assert origin_of(left.target) is children["first"].entry_function()
    assert origin_of(right.target) is children["second"].entry_function()

    (variant,) = HirModule.lookup("dispatches_to_a_variant").variants
    assert variant.body.target is children["variant_leaf"].entry_function()
    assert variant.body.target is not Scaled.entry_function()

    ((weight, converter),) = HirModule.lookup("converts_its_weight").converters
    assert weight == "w"
    assert converter.body.target is children["leaf"].entry_function()


def test_a_child_call_carries_activations_and_leaves_the_constants_declared() -> None:
    """Which parameters are constants, and whose weights they are, is not printed."""
    (child,) = [c for c in HirModule.modules if c.name == "mlp"]
    call = HirModule.lookup("carries_activations_only").body
    assert len(call.args) == 1
    assert [(p.name, p.is_const) for p in call.target.params] == [("x", False), ("w", True)]
    assert call.target.params[1].type == child.weights["w"]
    assert set(child.weights) == {"w"}
    assert child.weights["w"].shape == (8, 8)
    assert HirModule.weights["w"].shape == (8,)

    grandchild = next(c for c in HirModule.modules if c.name == "deep").modules[0]
    inner = HirModule.lookup("through_the_grandchild").body.target.body
    assert [p.is_const for p in inner.target.params] == [False, True]
    assert origin_of(inner.target) is grandchild.entry_function()


def test_a_rebuilt_child_target_is_owned_only_by_that_child() -> None:
    """Ownership after a rebuild, which no printed program states."""
    one, two = [c for c in HirModule.modules if c.name in ("first", "second")][:2]
    sized = specialize_function(Scaled.entry_function(), {"N": 8})
    resharded = elaborate(
        sized, (make_shard_tensor_type((8,), mesh=make_mesh((4,)), attrs=(Split(0),)),)
    )
    assert resharded is not sized
    assert not one.owns(resharded, derived=True)
    assert not two.owns(resharded, derived=True)


def test_a_wildcard_chain_reelaborates_the_whole_nested_call() -> None:
    """Re-elaboration builds new Functions; the program it started from is unchanged."""
    x_split = make_shard_tensor_type((8, 64), mesh=make_mesh((4,)), attrs=(Split(0),))
    rebuilt = elaborate(doubles_a_constant, (x_split,))
    assert rebuilt is not doubles_a_constant
    assert rebuilt.params[0].is_const is True
    assert rebuilt.params[0].type == x_split


def test_the_printer_falls_back_to_verbose_when_a_mesh_has_no_names() -> None:
    """A mesh without ``names=`` cannot use ``@`` sugar, so the layout prints in full."""
    src = as_script(MatmulModule.entry_function())
    assert "@" not in src.split("@func")[1].split("def ")[0]
    assert "ShardLayout(" in src


def test_both_authoring_spellings_print_to_one_canonical_program() -> None:
    """A string and the descriptor it names are one surface over one IR."""
    header = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.dsl.tf import *\n"
    )
    string_form = header + (
        "@func\n"
        'def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:\n'
        '    return cast(x, dtype="bf16")\n'
    )
    descriptor_form = header + (
        "from tilefoundry.ir.types import DType\n"
        "@func\n"
        'def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:\n'
        "    return cast(x, dtype=DType.bf16)\n"
    )
    printed = as_script(import_dsl(string_form))
    assert printed == as_script(import_dsl(descriptor_form))
    assert 'dtype="bf16"' in printed


def test_a_program_still_evaluates_to_what_torch_would_give() -> None:
    """Subscript semantics are a runtime contract; the golden only shows the IR."""
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    for name, expected in (
        ("index_drops_its_axis", x[:, :, 3]),
        ("slice_keeps_its_axis", x[:, :, 3:4]),
        ("index_counted_from_the_end", x[:, :, -1]),
        ("slice_strided_and_clamped", x[:, :, 1:20:3]),
    ):
        torch.testing.assert_close(
            evaluate(HirExpressions.lookup(name), x, device="cpu"), expected, msg=name
        )


def test_symbolic_slice_endpoints_preserve_shape_and_bind_across_a_call() -> None:
    """Equivalent full windows retain the authored dimension at a call boundary."""
    prelude = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import DimVar, Tensor\n"
    )
    full_window = import_dsl(
        prelude
        + '\nCTX_LEN = DimVar("CTX_LEN", 1, 4097)\n'
        '@func\ndef f(x: Tensor[(CTX_LEN, 128), "f32"]) '
        '-> Tensor[(CTX_LEN, 128), "f32"]:\n'
        "    return x[:, 0:128]\n"
    )
    explicit_window = HirExpressions.lookup("slice_to_symbolic_extents")
    assert isinstance(explicit_window.body, Call)
    assert isinstance(explicit_window.body.target, Slice)
    assert explicit_window.body.target.sizes == full_window.body.target.sizes
    assert explicit_window.body.target.strides == full_window.body.target.strides
    assert explicit_window.body.type == full_window.body.type
    assert explicit_window.body.args[1] == full_window.body.args[1]
    authored_extent = explicit_window.params[0].type.shape[0]
    assert explicit_window.body.target.sizes[0] is authored_extent
    assert explicit_window.body.type.shape[0] is authored_extent

    param = Var(type=explicit_window.return_type, name="window")
    consumer = Function.build(
        name="consume_window",
        params=(param,),
        body=param,
        return_type=explicit_window.return_type,
    )
    call = Call(type=consumer.return_type, target=consumer, args=(explicit_window.body,))
    assert TypeInferVisitor(TypeInferContext()).visit(call) == consumer.return_type

    dimension_start = import_dsl(
        prelude
        + '\nS = DimVar("m2_start_seq", 1, 4097)\n'
        '@func\ndef f(x: Tensor[(S + 8, 128), "f32"]) -> Tensor[(8, 128), "f32"]:\n'
        "    return x[S:S + 8, 0:128]\n"
    )
    start = dimension_start.body.args[1].elements[0]
    assert isinstance(start, Call) and isinstance(start.target, DimAdd)
    assert any(start.args[0] is arg for arg in dimension_start.params[0].type.shape[0].args)
    assert isinstance(start.args[1], Constant) and start.args[1].value == 0


def test_a_packed_cache_can_keep_a_symbolic_capacity_axis() -> None:
    """A layer index and full symbolic windows can name every packed-cache axis."""
    packed = import_dsl(
        "from tilefoundry import func, module\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import DimVar, Tensor\n"
        '\nCAP = DimVar("m2_capacity", 1, 4097)\n'
        '@module(entry="run")\n'
        "class PackedCache:\n"
        "    @func\n"
        '    def run(kc: Tensor[(4, CAP, 8, 16), "f32"], '
        'seed: Tensor[(CAP, 8, 16), "f32"]) -> Tensor[(CAP, 8, 16), "f32"]:\n'
        "        out = relu(seed)\n"
        "        for i in range(4):\n"
        "            out = add(out, kc[i, 0:CAP, 0:8, 0:16])\n"
        "        return out\n"
    )
    entry = packed.entry_function()
    infer_authored_types((entry,), packed)
    capacity = entry.params[0].type.shape[1]

    assert entry.body.type.shape == (capacity, 8, 16)


def _fused_reference(gu, seed):
    out = seed * 2
    for lo in range(0, 4, 2):
        out = out + gu[:, lo : lo + 2] * gu[:, lo + 4 : lo + 4 + 2]
    return out


def test_a_moved_tile_window_evaluates_where_it_says_it_reads() -> None:
    """Both spellings of the same move land the same data, and re-import unchanged."""
    gu = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    seed = torch.ones((3, 2), dtype=torch.float32)
    expected = _fused_reference(gu, seed)

    for name in ("two_windows_a_fixed_distance_apart", "a_summed_offset_names_the_same_move"):
        fn = HirExpressions.lookup(name)
        torch.testing.assert_close(evaluate(fn, gu, seed, device="cpu"), expected, msg=name)

    script = as_script(HirExpressions.lookup("two_windows_a_fixed_distance_apart"))
    assert "gu[:, n + 4]" in script, script
    torch.testing.assert_close(
        evaluate(import_dsl(script), gu, seed, device="cpu"), expected
    )


def test_a_range_scalar_and_a_runtime_endpoint_drive_a_slice_window() -> None:
    """A window whose start is only known at run time keeps its static size."""
    prelude = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import Tensor\n"
    )
    runtime_start = import_dsl(
        prelude
        + "\nfrom tilefoundry.ir.types.shard import Layout\n"
        "plain_layout = Layout((8, 4), (4, 1))\n"
        '\n@func\ndef f(x: Tensor[(8, 4), "f32", plain_layout], '
        'start: Tensor[(), "i64"]) -> Tensor[(4, 4), "f32"]:\n'
        "    return x[start:start + 4, :]\n"
    )
    assert runtime_start.body.target.sizes == (4, 4)
    assert runtime_start.body.args[1].elements[0] is runtime_start.params[1]
    assert runtime_start.body.type.layout is None

    shifted_start = import_dsl(
        prelude
        + '\n@func\ndef f(x: Tensor[(16, 4), "f32"], '
        'start: Tensor[(), "i64"]) -> Tensor[(8, 4), "f32"]:\n'
        "    return x[start + 1:start + 9, :]\n"
    )
    assert shifted_start.body.target.sizes == (8, 4)
    assert shifted_start.body.type.shape == (8, 4)

    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    torch.testing.assert_close(
        evaluate(runtime_start, x, torch.tensor(2, dtype=torch.int64), device="cpu"), x[2:6, :]
    )

    sharded = import_dsl(
        prelude
        + "\nfrom tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology\n"
        '\n@func\ndef f(x: Tensor[(8, 4), "f32", ShardLayout('
        "layout=Layout((8, 2, 2), (4, 2, 1)), attrs=(Split(1),), "
        'mesh=Mesh((Topology("gpu", 2),), Layout((2,), (1,)), names=("g",)))], '
        'start: Tensor[(), "i64"]) -> Tensor[(2, 4), "f32"]:\n'
        "    return x[start:start + 2, :]\n"
    )
    assert isinstance(sharded.body.type.layout, ShardLayout)
    assert sharded.body.type.layout.attrs == sharded.body.args[0].type.layout.attrs
    assert sharded.body.type.layout.layout.shape == (2, 2, 2)


def test_a_compile_time_list_is_indexed_where_it_is_written() -> None:
    """A comprehension and a plain list literal both bind a Python list of Exprs.

    Indexing either picks an expression rather than emitting an op, so neither
    list survives into the program a golden could show.
    """
    prelude = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import Tensor\n"
        '\n@func\ndef f(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 1), "f32"]:\n'
    )
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    comprehension = import_dsl(
        prelude
        + "    taps = [x[:, :, j:j + 1] for j in range(4)]\n"
        "    return add(taps[0], taps[-1])\n"
    )
    torch.testing.assert_close(
        evaluate(comprehension, x, device="cpu"), x[:, :, 0:1] + x[:, :, 3:4]
    )

    literal = import_dsl(
        prelude
        + "    ends = [x[:, :, 0:1], x[:, :, 7:8]]\n"
        "    return add(ends[0], ends[1])\n"
    )
    torch.testing.assert_close(
        evaluate(literal, x, device="cpu"), x[:, :, 0:1] + x[:, :, 7:8]
    )


def test_a_closure_int_mesh_dim_resolves_like_the_literal() -> None:
    """A closure int in a mesh-shape sugar prints back to the literal form.

    The parser must resolve the ``ast.Name`` rather than reject it, and the
    trailing reshard names no mesh axis at all, so it resolves its mesh from the
    enclosing scope.
    """
    assert as_script(mesh_dims_reshard_func(4, 32)) == as_script(literal_reshard_func())


def test_a_tile_window_survives_the_print_import_trip() -> None:
    """The canonical source of a windowed loop evaluates to what it started as."""
    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    seed = torch.ones((4, 4), dtype=torch.float32)
    fn = HirExpressions.lookup("full_tile_window")
    expected = evaluate(fn, x, seed, device="cpu")

    torch.testing.assert_close(
        evaluate(import_dsl(as_script(fn)), x, seed, device="cpu"), expected
    )
