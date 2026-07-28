"""``ConstTensor[...]`` annotation-only parameter form.

``ConstTensor[(M, K), dtype]`` resolves to the identical ``TensorType`` as
``Tensor[(M, K), dtype]``; only the parsed parameter ``Var.is_const`` flag
differs. The corpus prints and re-imports models whose weights are declared
``ConstTensor``, so equal ``TensorType`` semantics and the printer round-trip
ride on that; what it does not exercise is the flag surviving a call
elaboration that rebuilds the callee.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor
from tilefoundry.dsl.tf import add  # noqa: F401 -- bound via `dsl.tf` import
from tilefoundry.ir.hir.function import elaborate
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split


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
