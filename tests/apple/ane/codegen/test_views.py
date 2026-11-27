"""ANE view addresses reject opaque IR and preserve static parent displacements."""

from dataclasses import replace

import pytest

from tilefoundry.ir import dim, expr, types
from tilefoundry.ir.hir import math, tensor
from tilefoundry.ir.tir import memory
from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.codegen.tir import prim_function
from tilefoundry_apple.ane.ir import tir


def source(name="x"):
    """Construct the canonical source of the view audit's trigger inputs."""
    desc = isa.TensorDesc.from_shape("f16", (1, 32, 16, 16))
    return expr.Var(name, type=types.TensorType(
        desc.shape, expr.DType.f16, types.Layout(desc.shape, desc.strides), expr.StorageKind.GMEM,
    ))


def view(window):
    """Wrap a tensor expression in the actual PtrOf and TensorView operations."""
    pointer = expr.Call(memory.PtrOf(), (window,), type=types.PointerType(expr.DType.f16, expr.StorageKind.GMEM))
    return expr.Call(memory.TensorView(layout=window.type.layout), (pointer,), type=window.type)


def sliced(value, starts):
    """Make a statically sized window with explicit start expressions."""
    starts = tuple(dim.i64_const(start) if type(start) is int else start for start in starts)
    coordinates = expr.Tuple(starts, type=types.TupleType(tuple(start.type for start in starts)))
    return expr.Call(tensor.Slice(sizes=(1, 1, 1, 1), strides=(1, 1, 1, 1)),
                     (value, coordinates), type=value.type)


def test_static_slice_uses_all_layout_strides():
    x = source()
    root, address = prim_function.view_binding(view(sliced(x, (0, 2, 3, 4))))
    assert root is x
    assert address == (2 * 512 + 3 * 32 + 4) * 2


@pytest.mark.parametrize("position", ("view", "pointer", "window"))
def test_same_arity_wrong_operation_is_refused(position):
    window = sliced(source(), (0, 0, 0, 0))
    value = view(window)
    if position == "view":
        value = replace(value, target=tir.Unary())
    elif position == "pointer":
        value = replace(value, args=(replace(value.args[0], target=tir.Unary()),))
    else:
        value = view(replace(window, target=tir.Unary()))
    with pytest.raises(ValueError, match="expression is not decoded yet"):
        prim_function.view_binding(value)


@pytest.mark.parametrize("arithmetic", (False, True))
def test_nonconstant_slice_start_is_refused(arithmetic):
    start = (expr.Call(math.Binary(kind=expr.BinaryKind.ADD), (dim.i64_const(1), dim.i64_const(1)),
                       type=dim.i64_const(0).type) if arithmetic
             else expr.Var("i", type=dim.i64_const(0).type))
    with pytest.raises(ValueError, match="nonconstant slice start is not decoded yet"):
        prim_function.view_binding(view(sliced(source(), (0, 0, start, 0))))


def test_missing_slice_axes_are_refused():
    with pytest.raises(ValueError, match="axes must match the tensor rank"):
        prim_function.view_binding(view(sliced(source(), (1,))))


@pytest.mark.parametrize("start", (dim.i64_const(-1), expr.Constant(1.5, type=types.TensorType.from_shape((), expr.DType.f32))))
def test_slice_start_must_be_a_nonnegative_integer(start):
    with pytest.raises(ValueError, match="non-negative integer constants"):
        prim_function.view_binding(view(sliced(source(), (0, 0, start, 0))))


def test_slice_window_must_stay_in_its_source():
    with pytest.raises(ValueError, match="Slice window leaves source axis 2"):
        prim_function.view_binding(view(sliced(source(), (0, 0, 16, 0))))


def test_nested_bound_views_accumulate_parent_offsets_and_keep_the_root():
    x = source()
    v, alias, child = (source(name) for name in ("v", "alias", "child"))
    ctx = context.Context()
    ctx.bind(v, x, 64)
    ctx.bind(alias, *prim_function.view_binding(view(v)))
    ctx.bind(child, *prim_function.view_binding(view(sliced(alias, (0, 0, 1, 0)))))
    assert ctx.address_of(alias) == 64
    assert ctx.address_of(child) == 128
    assert ctx.symbols[alias] is x and ctx.symbols[child] is x


def test_unbound_nested_view_expression_is_refused():
    with pytest.raises(ValueError, match="slice source expression is not decoded yet"):
        prim_function.view_binding(view(sliced(view(source()), (0, 0, 1, 0))))
