"""``ConstTensor[...]`` annotation-only parameter form.

``ConstTensor[(M, K), dtype]`` resolves to the identical ``TensorType`` as
``Tensor[(M, K), dtype]``; only the parsed parameter ``Var.is_const`` flag
differs. Locks: equal ``TensorType`` semantics, printer round-trip, and
preservation through call elaboration (`ir.hir.function.elaborate`).
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor
from tilefoundry.dsl.tf import add  # noqa: F401 -- bound via `dsl.tf` import
from tilefoundry.inspection import as_script
from tilefoundry.ir.hir.function import elaborate
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.parser.hir_parser import parse_script


@func
def _uses_weight(
    x: Tensor[(8,), "f32"],
    weight: ConstTensor[(8,), "f32"],
) -> Tensor[(8,), "f32"]:
    return add(x, weight)


def test_const_tensor_matches_tensor_type_and_sets_is_const() -> None:
    x_param, weight_param = _uses_weight.params
    assert weight_param.type == x_param.type
    assert x_param.is_const is False
    assert weight_param.is_const is True


def test_const_tensor_round_trips_through_printer() -> None:
    printed = as_script(_uses_weight)
    assert 'weight: ConstTensor[(8,), "f32"]' in printed
    assert 'x: Tensor[(8,), "f32"]' in printed
    reparsed = parse_script(printed)
    assert reparsed.params[0].is_const is False
    assert reparsed.params[1].is_const is True


def test_const_tensor_preserved_through_forced_reelaboration() -> None:
    @func
    def leaf(w: ConstTensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return add(w, w)  # noqa: F821

    @func
    def outer_fn(w: ConstTensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return leaf(w)

    w_split = make_shard_tensor_type((8, 64), mesh=make_mesh((4,)), attrs=(Split(0),))
    new_outer = elaborate(outer_fn, (w_split,))
    tgt = new_outer.body.target
    assert tgt is not leaf
    assert tgt.params[0].is_const is True
    assert tgt.params[0].type == w_split
