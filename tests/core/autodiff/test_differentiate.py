"""differentiate: gradients against autograd through the evaluator, saved values, refusals."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from tests.core._corpus import Case, case, check_case, special
from tilefoundry import cli, inspection, runtime
from tilefoundry.interfaces import autodiff, evaluator
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import sharding
from tilefoundry.passes.transforms import differentiate

FIXTURES = Path(__file__).parents[1] / "fixtures" / "autodiff"


def loaded(module: core.Module):
    return module.load(runtime.DictResource({}))


def inputs_for(type_: types.Type, generator: torch.Generator):
    if isinstance(type_, types.TupleType):
        return tuple(inputs_for(item, generator) for item in type_.fields)
    if isinstance(type_.dtype, expr.FloatDType):
        return torch.randn(tuple(type_.shape), generator=generator) * 0.5 + 0.1
    return torch.zeros(tuple(type_.shape), dtype=torch.int64)


def leaves(value):
    if isinstance(value, tuple):
        for item in value:
            yield from leaves(item)
    else:
        yield value


def detached(value):
    if isinstance(value, tuple):
        return tuple(detached(item) for item in value)
    return value.detach() if isinstance(value, torch.Tensor) else value


def ones_like(type_: types.Type):
    if isinstance(type_, types.TupleType):
        return tuple(ones_like(item) for item in type_.fields)
    return torch.ones(tuple(type_.shape))


def is_float(type_: types.Type) -> bool:
    if isinstance(type_, types.TupleType):
        return any(is_float(item) for item in type_.fields)
    return isinstance(type_.dtype, expr.FloatDType)


def flat(type_: types.Type):
    """A tuple type as nested Python tuples of its tensor types."""
    return tuple(flat(item) for item in type_.fields) if isinstance(type_, types.TupleType) else type_


def fields(type_: types.Type) -> tuple[types.Type, ...]:
    return type_.fields if isinstance(type_, types.TupleType) else (type_,)


def check_gradients(module: core.Module, forward: core.Module, backward: core.Module, wrt) -> None:
    """The backward's gradients equal autograd's through the evaluator, for ones (D36)."""
    function = module.entry_function()
    generator = torch.Generator().manual_seed(0)
    inputs = [inputs_for(param.type, generator) for param in function.params]
    names = [param.name for param in function.params]
    trained = [leaf for name in wrt for leaf in leaves(inputs[names.index(name)])]
    for leaf in trained:
        leaf.requires_grad_(True)
    results = evaluator.evaluate(loaded(module), *inputs)
    results = results if isinstance(function.return_type, types.TupleType) else (results,)
    total = sum(
        value.sum() for value, type_ in zip(results, fields(function.return_type)) if is_float(type_)
    )
    reference = torch.autograd.grad(total, trained, allow_unused=True)
    saved = detached(evaluator.evaluate(loaded(forward), *detached(tuple(inputs)))[-1])
    gradients = [ones_like(type_) for type_ in fields(function.return_type) if is_float(type_)]
    computed = evaluator.evaluate(loaded(backward), *gradients, saved)
    computed = computed if len(wrt) > 1 else (computed,)
    for want, got in zip(reference, leaves(computed), strict=True):
        want = torch.zeros_like(got) if want is None else want
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def reshards(backward: core.Module) -> list[tuple[tuple, tuple]]:
    """``(source attrs, result attrs)`` of every Reshard in the backward's entry."""
    found = []
    for value in core.collect_exprs(backward.entry_function().body):
        if isinstance(value, expr.Call) and isinstance(value.target, sharding.Reshard):
            source = types.shard_layout_of(value.args[0].type.layout)
            found.append((source.attrs, types.shard_layout_of(value.type.layout).attrs))
    return found


class Differentiated(Case):
    """A forward program whose generated backward is checked against autograd."""

    wrt: tuple[str, ...] = ()
    recompute: tuple[str, ...] = ()
    saved: frozenset[str] | None = None
    reshards: tuple[tuple[tuple, tuple], ...] = ()

    def guard(self, module, problems):
        before = inspection.as_script(module)
        forward, backward = differentiate.differentiate(module, wrt=self.wrt, recompute=self.recompute)
        assert inspection.as_script(module) == before, "differentiate changed its input Module"
        assert (forward.name, backward.name) == (f"{module.name}Forward", f"{module.name}Backward")
        check_gradients(module, forward, backward, self.wrt)
        if self.saved is not None:
            assert {value.name for value in differentiate.saved_values(forward)} == self.saved
        params = {param.name: param for param in module.entry_function().params}
        returned = fields(backward.entry_function().return_type) if len(self.wrt) > 1 else (
            backward.entry_function().return_type,
        )
        for name, type_ in zip(self.wrt, returned):
            for got, held in zip(leaves(flat(type_)), leaves(flat(params[name].type))):
                assert got.layout == autodiff.gradient_layout(held.layout), name
        found = reshards(backward)
        for expected in self.reshards:
            assert expected in found, (expected, found)


S0, P, B = types.Split(0), types.Partial("sum"), types.Broadcast()


@case
class Elementwise(Differentiated):
    path: Path = FIXTURES / "elementwise.py"
    wrt: tuple[str, ...] = ("x", "w", "b")


@case
class Matmul(Differentiated):
    path: Path = FIXTURES / "matmul.py"
    wrt: tuple[str, ...] = ("x", "a", "b", "c")


@case
class Reduce(Differentiated):
    path: Path = FIXTURES / "reduce.py"
    wrt: tuple[str, ...] = ("x",)


@case
class ReshapeTranspose(Differentiated):
    path: Path = FIXTURES / "reshape_transpose.py"
    wrt: tuple[str, ...] = ("x", "w")


@case
class SiblingCall(Differentiated):
    path: Path = FIXTURES / "sibling_call.py"
    wrt: tuple[str, ...] = ("x", "w1", "w2")


@case
class ChildCall(Differentiated):
    path: Path = FIXTURES / "child_call.py"
    name: str | None = "ChildCall"
    wrt: tuple[str, ...] = ("x", "w1", "w2")


@case
class InactiveCall(Differentiated):
    """No gradient reaches either call; both callees still get a forward version."""

    path: Path = FIXTURES / "inactive_call.py"
    name: str | None = "InactiveCall"
    wrt: tuple[str, ...] = ("x",)


@case
class TupleParam(Differentiated):
    path: Path = FIXTURES / "tuple_param.py"
    wrt: tuple[str, ...] = ("w", "x")


@case
class Sharded(Differentiated):
    path: Path = FIXTURES / "sharded.py"
    wrt: tuple[str, ...] = ("x", "w")
    saved: frozenset[str] | None = frozenset({"x", "h", "full"})
    reshards: tuple = (((P,), (S0,)),)


@case
class Hsdp(Differentiated):
    path: Path = FIXTURES / "hsdp.py"
    wrt: tuple[str, ...] = ("x", "w")
    reshards: tuple = (((P, P), (B, S0)),)


@case
class DataParallel(Differentiated):
    path: Path = FIXTURES / "data_parallel.py"
    wrt: tuple[str, ...] = ("w",)
    reshards: tuple = (((P,), (B,)),)


@case
class MeshRegions(Differentiated):
    path: Path = FIXTURES / "mesh_region.py"
    wrt: tuple[str, ...] = ("x", "w")


CASES = special(Differentiated)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_gradients_match_autograd(case: Differentiated) -> None:
    check_case(case)


def test_recompute_reads_the_parameter_instead_of_the_gathered_weight() -> None:
    module = cli.load_authored_ir(str(FIXTURES / "sharded.py"))
    forward, backward = differentiate.differentiate(module, wrt=("x", "w"), recompute=("Layer.forward:full",))
    assert {value.name for value in differentiate.saved_values(forward)} == {"x", "h", "w"}
    assert ((S0,), (B,)) in reshards(backward)
    check_gradients(module, forward, backward, ("x", "w"))


def test_saved_values_report_name_type_and_origin() -> None:
    module = cli.load_authored_ir(str(FIXTURES / "sharded.py"))
    forward, _ = differentiate.differentiate(module, wrt=("x", "w"))
    rows = {value.name: value for value in differentiate.saved_values(forward)}
    assert rows["x"].origin == "parameter"
    assert rows["full"].origin == "reshard  sharded.py:24"
    assert rows["full"].type.shape == (6, 4)
    assert types.shard_layout_of(rows["full"].type.layout).attrs == (B,)


def test_hand_written_step_calls_the_generated_modules() -> None:
    """D27: the author's step calls the forward and the backward as two child slots."""
    path = FIXTURES / "step.py"
    training = cli.load_authored_ir(f"{path}:Training")
    layer = cli.load_authored_ir(f"{path}:Layer")
    generator = torch.Generator().manual_seed(0)
    x, w, target = (torch.randn(shape, generator=generator) for shape in ((4, 3), (3, 3), (4, 3)))
    loss, updated = evaluator.evaluate(loaded(training), x, w, target)
    trained = w.clone().requires_grad_(True)
    reference = ((evaluator.evaluate(layer, x, trained) - target) ** 2).sum()
    (gradient,) = torch.autograd.grad(reference, (trained,))
    torch.testing.assert_close(loss, reference.detach())
    torch.testing.assert_close(updated, w - 0.1 * gradient)


REFUSED = (
    ("elementwise.py", (), (), "wrt names no parameter"),
    ("elementwise.py", ("q",), (), "wrt 'q' is not a parameter"),
    ("int_param.py", ("index",), (), "wrt 'index' is not a floating point"),
    ("const_param.py", ("w",), (), "wrt 'w' is a ConstTensor"),
    ("no_rule.py", ("x",), (), "ReLU has no autodiff rule"),
    ("unsupported_kind.py", ("x",), (), "Binary MIN has no autodiff rule"),
    ("loop.py", ("x",), (), "LoopRegion is not differentiated"),
    ("backward_name.py", ("x",), (), "declares a function named 'backward'"),
    ("sharded.py", ("w",), ("Nope.forward:full",), "no Module named 'Nope'"),
    ("sharded.py", ("w",), ("Layer.nope:full",), "'nope' must resolve to exactly one function"),
    ("sharded.py", ("w",), ("Layer.forward:nope",), "binds no value named 'nope'"),
    ("sharded.py", ("w",), ("Layer.forward:w",), "'w' is a parameter"),
    ("sharded.py", ("w",), ("Layer.forward.inner:full",), "not by a path through child slots"),
    ("sharded.py", ("w",), ("full",), "expected Module[.function]:variable"),
    ("rebound.py", ("x",), ("Rebound.forward:h",), "2 values are named 'h'"),
    ("same_name.py:SameName", ("x",), ("Block:h",), "2 different Modules are named 'Block'"),
    ("sibling_call.py", ("x",), ("SiblingCall.forward:a",), "result of calling 'layer'"),
)


@pytest.mark.parametrize(
    ("fixture", "wrt", "recompute", "message"), REFUSED, ids=lambda value: str(value)
)
def test_refusals_name_what_they_refuse(fixture, wrt, recompute, message) -> None:
    module = cli.load_authored_ir(str(FIXTURES / fixture))
    with pytest.raises(ValueError, match=re.escape(message)):
        differentiate.differentiate(module, wrt=wrt, recompute=recompute)


def test_refusals_point_at_the_source_line() -> None:
    module = cli.load_authored_ir(str(FIXTURES / "no_rule.py"))
    with pytest.raises(ValueError, match=r"no_rule\.py:11 \(r\)"):
        differentiate.differentiate(module, wrt=("x",))


def test_second_derivative_is_refused() -> None:
    module = cli.load_authored_ir(str(FIXTURES / "elementwise.py"))
    _, backward = differentiate.differentiate(module, wrt=("x",))
    with pytest.raises(ValueError, match="second derivative is not supported"):
        differentiate.differentiate(backward, wrt=("dy",))
