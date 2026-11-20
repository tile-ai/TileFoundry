"""differentiate: gradients against autograd through the evaluator, saved values, refusals."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry import cli, inspection, runtime
from tilefoundry.interfaces import autodiff, evaluator
from tilefoundry.ir import core, dim, expr, types, utils
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
    return torch.randint(0, 3, tuple(type_.shape), generator=generator)


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


def pieces(value) -> tuple:
    """Each unit's own piece of a local view, which the loss sums; a tensor is one piece."""
    if isinstance(value, torch.Tensor) or value.is_global:
        return (value,)
    return tuple(data for _, data in value.tensor)


def check_gradients(
    module: core.Module, forward: core.Module, backward: core.Module, wrt, rtol: float = 1e-5, atol: float = 1e-6,
    dims: Mapping[str, int] = MappingProxyType({}),
) -> None:
    """The backward's gradients equal autograd's through the evaluator, for ones (D36).

    *dims* binds each open dimension to the extent its inputs are drawn at.
    """
    function = module.entry_function()
    generator = torch.Generator().manual_seed(0)
    inputs = [inputs_for(dim.substitute_dims(param.type, dims), generator) for param in function.params]
    names = [param.name for param in function.params]
    trained = [leaf for name in wrt for leaf in leaves(inputs[names.index(name)])]
    for leaf in trained:
        leaf.requires_grad_(True)
    results = evaluator.evaluate(loaded(module), *inputs)
    results = results if isinstance(function.return_type, types.TupleType) else (results,)
    total = sum(
        piece.sum()
        for value, type_ in zip(results, fields(function.return_type))
        if is_float(type_)
        for piece in pieces(value)
    )
    reference = torch.autograd.grad(total, trained, allow_unused=True)
    saved = detached(evaluator.evaluate(loaded(forward), *detached(tuple(inputs)))[-1])
    returned = dim.substitute_dims(function.return_type, dims)
    gradients = [ones_like(type_) for type_ in fields(returned) if is_float(type_)]
    computed = evaluator.evaluate(loaded(backward), *gradients, saved)
    computed = computed if len(wrt) > 1 else (computed,)
    for want, got in zip(reference, leaves(computed), strict=True):
        want = torch.zeros_like(got) if want is None else want
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)


def where(value: expr.Expr) -> str:
    return expr.binding_name(value) or type(value.target).__name__


def reshards(backward: core.Module) -> list[tuple[tuple, tuple]]:
    """``(source attrs, result attrs)`` of each Reshard in the backward's entry; plain is B."""
    found = []
    for value in core.collect_exprs(backward.entry_function().body):
        if isinstance(value, expr.Call) and isinstance(value.target, sharding.Reshard):
            source = types.shard_layout_of(value.args[0].type.layout)
            if source is not None:
                found.append((source.attrs, tuple(utils.attrs_by_axis(value.type, source.mesh).values())))
    return found


class Differentiated(Case):
    """A forward program whose generated backward is checked against autograd."""

    wrt: tuple[str, ...] = ()
    recompute: tuple[str, ...] = ()
    saved: frozenset[str] | None = None
    saved_tensors: Mapping[str, int] | None = None
    slots: Mapping[str, tuple[str, ...]] | None = None
    loops: int | None = None
    same_text: bool = True
    """False: both Modules load, but the printer spells a dim operand that is not a dim inline."""
    reshards: tuple[tuple[tuple, tuple], ...] = ()
    rtol: float = 1e-5
    atol: float = 1e-6
    dims: Mapping[str, int] = MappingProxyType({})

    @property
    def id(self) -> str:
        return f"{super().id}[{','.join(self.recompute)}]" if self.recompute else super().id

    def guard(self, module, problems):
        before = inspection.as_script(module)
        forward, backward = differentiate.differentiate(module, wrt=self.wrt, recompute=self.recompute)
        assert inspection.as_script(module) == before, "differentiate changed its input Module"
        assert (forward.name, backward.name) == (f"{module.name}Forward", f"{module.name}Backward")
        check_gradients(module, forward, backward, self.wrt, self.rtol, self.atol, self.dims)
        if self.saved is not None:
            assert {value.name for value in differentiate.saved_values(forward)} == self.saved
        if self.saved_tensors is not None:
            counted = {value.name: len(tuple(leaves(flat(value.type)))) for value in differentiate.saved_values(forward)}
            assert counted == dict(self.saved_tensors)
        if self.slots is not None:
            elements = {key: tuple(each.name for each in core.instances(slot)) for key, slot in forward.children.items()}
            assert elements == dict(self.slots)
        if self.loops is not None:
            body = forward.entry_function().body
            assert sum(isinstance(each, core.LoopRegion) for each in core.collect_exprs(body)) == self.loops
        for generated in (forward, backward):
            named: dict[str, core.Module] = {}
            for each in core.module_definition_order(generated):
                assert named.setdefault(each.name, each) is each, f"two Modules are named {each.name!r}"
        text = inspection.as_script((forward, backward))
        assert text.count("from __future__ import annotations") == 1
        classes = [line for line in text.splitlines() if line.startswith("class ")]
        assert len(classes) == len(set(classes)), "a class is written twice"
        reloaded = tuple(reload(text, generated.name) for generated in (forward, backward))
        assert not self.same_text or inspection.as_script(reloaded) == text, "the reloaded Modules print other text"
        params = {param.name: param for param in module.entry_function().params}
        returned = fields(backward.entry_function().return_type) if len(self.wrt) > 1 else (
            backward.entry_function().return_type,
        )
        for name, type_ in zip(self.wrt, returned):
            for got, held in zip(leaves(flat(type_)), leaves(flat(params[name].type))):
                assert got.layout == autodiff.gradient_layout(held.layout), name
        for value in core.collect_exprs(backward.entry_function().body):
            if isinstance(value, expr.Call) and isinstance(value.target, sharding.Reshard):
                source = value.args[0].type
                assert (source.layout, source.storage) != (value.type.layout, value.type.storage), where(value)
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
class ChildCallOneElement(ChildCall):
    recompute: tuple[str, ...] = ("ChildCall.rest.1:h",)
    saved_tensors: Mapping[str, int] | None = MappingProxyType({"first.saved": 4, "rest.0.saved": 4, "rest.1.saved": 3})
    slots: Mapping[str, tuple[str, ...]] | None = MappingProxyType(
        {"first": ("LayerForward",), "rest": ("LayerForward", "Layer_1Forward")}
    )


@case
class ChildCallOneElementSaved(ChildCall):
    """The element's own ``:None`` replaces the whole-layer choice of its definition."""

    recompute: tuple[str, ...] = ("Layer", "ChildCall.rest.0:None")
    saved_tensors: Mapping[str, int] | None = MappingProxyType({"first.saved": 2, "rest.0.saved": 4, "rest.1.saved": 2})
    slots: Mapping[str, tuple[str, ...]] | None = MappingProxyType(
        {"first": ("LayerForward",), "rest": ("Layer_1Forward", "LayerForward")}
    )


@case
class ChildCallWholeSlot(ChildCall):
    recompute: tuple[str, ...] = ("ChildCall.first",)
    saved_tensors: Mapping[str, int] | None = MappingProxyType({"first.saved": 2, "rest.0.saved": 4, "rest.1.saved": 4})
    slots: Mapping[str, tuple[str, ...]] | None = MappingProxyType(
        {"first": ("Layer_1Forward",), "rest": ("LayerForward", "LayerForward")}
    )


@case
class ChildCallSharedCopy(ChildCall):
    """Two elements, in two slots, with one choice share one copy."""

    recompute: tuple[str, ...] = ("ChildCall.first:h", "ChildCall.rest.1:h")
    saved_tensors: Mapping[str, int] | None = MappingProxyType({"first.saved": 3, "rest.0.saved": 4, "rest.1.saved": 3})
    slots: Mapping[str, tuple[str, ...]] | None = MappingProxyType(
        {"first": ("Layer_1Forward",), "rest": ("LayerForward", "Layer_1Forward")}
    )


def stacked(slot: str, *indices: int) -> frozenset[str]:
    return frozenset(f"{slot}.*.saved[{index}]" for index in indices)


ONE = MappingProxyType({"layers": ("LayerForward",) * 3})
CUT = MappingProxyType(
    {"layers_0": ("LayerForward",), "layers_1": ("Layer_1Forward",), "layers_2": ("LayerForward",)}
)


@case
class ModuleListLoop(Differentiated):
    path: Path = FIXTURES / "module_list_loop.py"
    name: str | None = "Stack"
    wrt: tuple[str, ...] = ("x", "ws")
    saved: frozenset[str] | None = stacked("layers", 0, 1, 3) | {"ws", "v4"}
    slots: Mapping[str, tuple[str, ...]] | None = ONE
    loops: int | None = 1
    same_text: bool = False


@case
class ModuleListOneLayer(ModuleListLoop):
    recompute: tuple[str, ...] = ("Stack.layers.1:h",)
    saved: frozenset[str] | None = (
        stacked("layers_0", 0, 1, 3) | stacked("layers_1", 0, 1) | stacked("layers_2", 0, 1, 3) | {"ws", "v9"}
    )
    slots: Mapping[str, tuple[str, ...]] | None = CUT
    loops: int | None = 3


@case
class ModuleListWholeLayer(ModuleListLoop):
    recompute: tuple[str, ...] = ("Stack.layers.1",)
    saved: frozenset[str] | None = (
        stacked("layers_0", 0, 1, 3) | stacked("layers_1", 0) | stacked("layers_2", 0, 1, 3) | {"ws", "v8"}
    )
    slots: Mapping[str, tuple[str, ...]] | None = CUT
    loops: int | None = 3


@case
class ModuleListSavedEnds(ModuleListLoop):
    """Two elements apart keep what they compute; the one between recomputes all of it.

    A ``:None`` written twice is one choice.
    """

    recompute: tuple[str, ...] = ("Layer", "Stack.layers.0:None", "Stack.layers.2:None", "Stack.layers.2:None")
    saved: frozenset[str] | None = (
        stacked("layers_0", 0, 1, 3) | stacked("layers_1", 0) | stacked("layers_2", 0, 1, 3) | {"ws", "v8"}
    )
    slots: Mapping[str, tuple[str, ...]] | None = MappingProxyType(
        {"layers_0": ("Layer_1Forward",), "layers_1": ("LayerForward",), "layers_2": ("Layer_1Forward",)}
    )
    loops: int | None = 3


@case
class ModuleListNeighbours(ModuleListLoop):
    """Neighbouring elements with one choice are one segment and one loop."""

    recompute: tuple[str, ...] = ("Stack.layers.1:h", "Stack.layers.2:h")
    saved: frozenset[str] | None = stacked("layers_0", 0, 1, 3) | stacked("layers_1", 0, 1) | {"ws", "v6"}
    slots: Mapping[str, tuple[str, ...]] | None = MappingProxyType(
        {"layers_0": ("LayerForward",), "layers_1": ("Layer_1Forward",) * 2}
    )
    loops: int | None = 2


@case
class ModuleListEveryLayer(ModuleListLoop):
    """Every element with one choice keeps the slot and the loop whole."""

    recompute: tuple[str, ...] = ("Stack.layers.0", "Stack.layers.1", "Stack.layers.2")
    saved: frozenset[str] | None = stacked("layers", 0) | {"ws", "v2"}
    slots: Mapping[str, tuple[str, ...]] | None = MappingProxyType({"layers": ("Layer_1Forward",) * 3})


@case
class ModuleListConstantCall(Differentiated):
    """A call by a constant index of a cut slot calls the element in its segment."""

    path: Path = FIXTURES / "module_list_constant_call.py"
    name: str | None = "Stack"
    wrt: tuple[str, ...] = ("x", "w")
    recompute: tuple[str, ...] = ("Stack.layers.1:h",)
    slots: Mapping[str, tuple[str, ...]] | None = CUT


@case
class ModuleListNested(Differentiated):
    """The loop over the microbatches stays one loop; the loop over the layers inside it is cut."""

    path: Path = FIXTURES / "module_list_nested.py"
    name: str | None = "Stack"
    wrt: tuple[str, ...] = ("x", "ws")
    recompute: tuple[str, ...] = ("Stack.layers.1:h",)
    slots: Mapping[str, tuple[str, ...]] | None = CUT
    loops: int | None = 4
    same_text: bool = False


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


@case
class RmsNorm(Differentiated):
    path: Path = FIXTURES / "rms_norm.py"
    wrt: tuple[str, ...] = ('x', 'w')


@case
class Softmax(Differentiated):
    path: Path = FIXTURES / "softmax.py"
    wrt: tuple[str, ...] = ('x',)


@case
class Activations(Differentiated):
    path: Path = FIXTURES / "sigmoid_silu_softplus.py"
    wrt: tuple[str, ...] = ('x',)


@case
class Rope(Differentiated):
    path: Path = FIXTURES / "rope.py"
    wrt: tuple[str, ...] = ('q', 'k')


@case
class SliceConcat(Differentiated):
    path: Path = FIXTURES / "slice_concat.py"
    wrt: tuple[str, ...] = ('x', 'y')


@case
class CastFullLike(Differentiated):
    path: Path = FIXTURES / "cast_full_like.py"
    wrt: tuple[str, ...] = ('x',)


@case
class BitcastAgreed(Differentiated):
    path: Path = FIXTURES / "bitcast_agreed.py"
    wrt: tuple[str, ...] = ("x", "w")


@case
class Pipeline(Differentiated):
    path: Path = FIXTURES / "pipeline.py"
    wrt: tuple[str, ...] = ("x", "w0", "w1")


@case
class ExpertParallel(Differentiated):
    path: Path = FIXTURES / "expert_parallel.py"
    wrt: tuple[str, ...] = ("tokens", "w")
    dims: Mapping[str, int] = MappingProxyType({"n": 4})


@case
class IndexAdd(Differentiated):
    path: Path = FIXTURES / "index_add.py"
    wrt: tuple[str, ...] = ("dst", "src")


@case
class IndexSelect(Differentiated):
    path: Path = FIXTURES / "index_select.py"
    wrt: tuple[str, ...] = ('table',)


@case
class RepeatInterleave(Differentiated):
    path: Path = FIXTURES / "repeat_interleave.py"
    wrt: tuple[str, ...] = ('x',)


@case
class Topk(Differentiated):
    path: Path = FIXTURES / "topk.py"
    wrt: tuple[str, ...] = ('x',)


@case
class LoopRange(Differentiated):
    path: Path = FIXTURES / "loop_range.py"
    wrt: tuple[str, ...] = ('x', 'w', 'h0')


@case
class FusedViews(Differentiated):
    path: Path = FIXTURES / "fused_views.py"
    wrt: tuple[str, ...] = ("x", "w")


@case
class ParamView(Differentiated):
    path: Path = FIXTURES / "param_view.py"
    wrt: tuple[str, ...] = ("x", "w")


@case
class TwoLoops(Differentiated):
    path: Path = FIXTURES / "two_loops.py"
    wrt: tuple[str, ...] = ("x", "w")


@case
class LoopTwoCarries(Differentiated):
    path: Path = FIXTURES / "loop_two_carries.py"
    wrt: tuple[str, ...] = ('a0', 'b0', 'w')


@case
class LoopTile(Differentiated):
    path: Path = FIXTURES / "loop_tile.py"
    wrt: tuple[str, ...] = ('x', 'w')


@case
class LoopCall(Differentiated):
    path: Path = FIXTURES / "loop_call.py"
    wrt: tuple[str, ...] = ('keys', 'decay')


@case
class LoopEmpty(Differentiated):
    path: Path = FIXTURES / "loop_empty.py"
    wrt: tuple[str, ...] = ('x', 'w')


@case
class LoopSharded(Differentiated):
    path: Path = FIXTURES / "loop_sharded.py"
    wrt: tuple[str, ...] = ("h0", "w")


@case
class LoopNested(Differentiated):
    path: Path = FIXTURES / "loop_nested.py"
    wrt: tuple[str, ...] = ("x", "w")


@case
class MeshDataParallel(Differentiated):
    path: Path = FIXTURES / "mesh_data_parallel.py"
    wrt: tuple[str, ...] = ("x", "w")
    reshards: tuple = (((P,), (B,)),)


@case
class MeshTwoRegions(Differentiated):
    path: Path = FIXTURES / "mesh_two_regions.py"
    wrt: tuple[str, ...] = ("x", "w")


@case
class VoidCall(Differentiated):
    path: Path = FIXTURES / "void_call.py"
    wrt: tuple[str, ...] = ("x",)


BLOCK_WEIGHTS = ("wq", "wk", "wv", "wo", "wu", "wd")


@case
class HsdpBlock(Differentiated):
    path: Path = FIXTURES / "hsdp_block.py"
    wrt: tuple[str, ...] = ("x", *BLOCK_WEIGHTS)
    rtol: float = 1e-4
    atol: float = 1e-5


@case
class MegatronBlock(Differentiated):
    """The Partial gradient of a Broadcast value is reduced before a rule reads it."""

    path: Path = FIXTURES / "megatron_block.py"
    wrt: tuple[str, ...] = ("x", "wn1", "wq", "wk", "wv", "wo", "wn2", "wg", "wu", "wd")
    reshards: tuple = (((P,), (B,)),)
    rtol: float = 1e-4
    atol: float = 1e-5


@case
class SplitIndexSelect(Differentiated):
    path: Path = FIXTURES / "split_index_select.py"
    wrt: tuple[str, ...] = ("table",)


@case
class LayersLoop(Differentiated):
    """Each layer's weights and index are computed again, not stacked; the stacks are saved once."""

    path: Path = FIXTURES / "layers_loop.py"
    wrt: tuple[str, ...] = ("x", "wn0", "norms", "up", "down")
    saved: frozenset[str] | None = frozenset({
        "x", "wn0", "x0", "v8", "v9", "norms", "up", "down",
        "layer.saved[0]", "layer.saved[2]", "layer.saved[5]", "layer.saved[6]",
    })
    rtol: float = 1e-4
    atol: float = 1e-5


@case
class FsdpLoop(Differentiated):
    """The backward gathers each layer again from the split stacked weights, which it saves."""

    path: Path = FIXTURES / "fsdp_loop.py"
    saved: frozenset[str] | None = frozenset(
        {"up", "down", "v5", "layer.saved[1]", "layer.saved[3]", "layer.saved[4]"}
    )
    wrt: tuple[str, ...] = ("x", "up", "down")
    rtol: float = 1e-4
    atol: float = 1e-5


@case
class QwenBlock(Differentiated):
    path: Path = FIXTURES / "qwen_block.py"
    wrt: tuple[str, ...] = ("x", "wn1", "wq", "wk", "wv", "wo", "wn2", "wg", "wu", "wd")
    recompute: tuple[str, ...] = ("Qwen:p", "Qwen:h2", "Qwen:gate")
    rtol: float = 1e-4
    atol: float = 1e-5


@case
class Moe(Differentiated):
    path: Path = FIXTURES / "moe.py"
    wrt: tuple[str, ...] = ("x", "wg", "experts")


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
    assert rows["full"].origin == "reshard  sharded.py:18"
    assert rows["full"].buffer is None
    assert rows["full"].type.shape == (6, 4)
    assert types.shard_layout_of(rows["full"].type.layout) is None


def test_a_loop_stacks_what_changes_per_step_and_keeps_one_copy_of_the_rest() -> None:
    module = cli.load_authored_ir(str(FIXTURES / "loop_range.py"))
    forward, _ = differentiate.differentiate(module, wrt=("x", "w", "h0"))
    rows = differentiate.saved_values(forward)
    stacks = [row for row in rows if row.origin.startswith("stack of 5: ")]
    assert stacks and all(row.type.shape[0] == 5 for row in stacks)
    assert {row.name for row in rows if row.origin == "parameter"} == {"w"}
    assert next(row for row in rows if row.name == "w").type.shape == (3,)


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
    ("no_rule.py", ("x",), (), "Tanh has no autodiff rule"),
    ("unsupported_kind.py", ("x",), (), "Binary MIN has no autodiff rule"),
    ("loop_dynamic.py", ("x",), (), "trip count is not static"),
    ("loop_range.py", ("x",), ("LoopRange.forward:row",), "'row' is computed inside a loop"),
    ("strided_slice.py", ("x",), (), "Slice with strides (2, 1) has no autodiff rule"),
    ("filled_index_select.py", ("table",), (), "IndexSelect with a fill_value has no autodiff rule"),
    ("topk_inner_axis.py", ("x",), (), "TopK on an axis that is not the last has no autodiff rule"),
    ("backward_name.py", ("x",), (), "declares a function named 'backward'"),
    ("sharded.py", ("w",), ("Nope.forward:full",), "no Module named 'Nope'"),
    ("sharded.py", ("w",), ("Layer.nope:full",), "'Layer' has no function or child slot named 'nope'"),
    ("sharded.py", ("w",), ("Layer.forward:nope",), "binds no value named 'nope'"),
    ("sharded.py", ("w",), ("Layer.forward:w",), "'w' is a parameter"),
    ("rebound.py", ("x",), ("Rebound.forward:h",), "2 values are named 'h'"),
    ("same_name.py:SameName", ("x",), ("Block:h",), "2 different Modules are named 'Block'"),
    ("sibling_call.py", ("x",), ("SiblingCall.forward:a",), "result of calling 'layer'"),
    ("child_call.py:ChildCall", ("x",), ("ChildCall.nope.0",), "'ChildCall' has no function or child slot named 'nope'"),
    ("child_call.py:ChildCall", ("x",), ("ChildCall.rest.5",), "slot 'rest' has 2 elements"),
    ("child_call.py:ChildCall", ("x",), ("ChildCall.rest",), "slot 'rest' is a ModuleList; name one element"),
    ("child_call.py:ChildCall", ("x",), ("ChildCall.first.0",), "slot 'first' is not a ModuleList"),
    ("child_call.py:ChildCall", ("x",), ("ChildCall.rest.0.inner.1:h",), "an element path has one child slot"),
    ("child_call.py:ChildCall", ("x",), ("Layer.forward.inner:full",), "an element path starts at the root Module 'ChildCall'"),
    ("child_call.py:ChildCall", ("x",), ("Layer:None",), "':None' names one element"),
    ("child_call.py:ChildCall", ("x",), ("ChildCall.rest.0:None", "ChildCall.rest.0:h"), "':None' and another selector"),
    ("child_call.py:ChildCall", ("x",), ("Nope",), "no Module named 'Nope'"),
    ("child_call.py:ChildCall", ("x",), (":h",), "expected Module[.function][:variable]"),
    ("child_call.py:ChildCall", ("x",), ("",), "expected Module[.function][:variable]"),
    ("child_call.py:ChildCall", ("x",), ("Layer.forward",), "name a variable after the function"),
    ("module_list_shifted.py:Stack", ("x",), ("Stack.layers.1:h",), "slot 'layers' is indexed by a value that is not the loop index"),
    ("module_list_step.py:Stack", ("x",), ("Stack.layers.1:h",), "loop over 'layers' has step 2"),
    ("module_list_overrun.py:Stack", ("x",), ("Stack.layers.1:h",), "loop over 'layers' runs [0, 4); the slot has 3 elements"),
    ("module_list_empty.py:Stack", ("x",), ("Stack.layers.1:h",), "loop over 'layers' runs [2, 2); the slot has 3 elements"),
    ("module_list_dynamic.py:Stack", ("x",), ("Stack.layers.1:h",), "loop over 'layers': trip count is not static"),
    ("module_list_const.py:Stack", ("x",), ("Stack.layers.1:h",), "'Layer' holds a ConstTensor"),
    ("module_list_name_taken.py:Stack", ("x",), ("Stack.layers.1:h",), "'layers_0' is already a name in 'Stack'"),
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
