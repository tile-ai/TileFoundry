"""One child IR reached from two owners binds per owner, not once for the last.

No corpus model can reach this state, which is why it is here rather than in a
model's own tests. Normal construction cannot produce it either: ``__post_init__``
claims a child that already belongs to another owner as a clone, so two
``renamed`` copies never share one. The aliasing is built deliberately below with
``copy.copy``, which skips ``__post_init__`` and therefore skips that claim.

It is worth pinning on its own terms because the isolation currently holds twice
over -- once by that cloning, once by constants living per loading -- and a change
to the first (a ``__copy__``, a different ``renamed``) should not be able to take
the property with it. Under the in-place binding this replaces, the aliased child
was bound once per owner with the last owner winning, silently and with no error.
"""

from __future__ import annotations

import copy

import torch

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor
from tilefoundry.runtime.resource import DictResource


@module(entry="scale")
class _Leaf:
    @func
    def scale(x: Tensor[(2,), "f32"], w: ConstTensor[(2,), "f32"]):
        return x * w


@module(entry="scale_through")
class _Owner:
    leaf = _Leaf

    @func
    def scale_through(x: Tensor[(2,), "f32"]):
        return x + x


def _aliasing_owner(name: str) -> object:
    """An owner sharing ``_Owner``'s very child object, which the public API
    cannot build."""
    node = copy.copy(_Owner)
    object.__setattr__(node, "name", name)
    return node


def test_one_shared_child_binds_once_per_owner() -> None:
    left, right = _aliasing_owner("left"), _aliasing_owner("right")
    assert left.modules[0] is right.modules[0], "the aliased state was not built"

    ones = torch.ones(2, dtype=torch.float32)
    loaded_left = left.load(DictResource({"leaf.w": torch.full((2,), 3.0)}))
    loaded_right = right.load(DictResource({"leaf.w": torch.full((2,), 10.0)}))

    # Same child IR on both sides, and each loading reads only its own subtree --
    # including after the other has loaded, which is where the old binding lost.
    assert loaded_left.leaf.module is loaded_right.leaf.module
    assert loaded_left.leaf.scale(ones).float().cpu().tolist() == [3.0, 3.0]
    assert loaded_right.leaf.scale(ones).float().cpu().tolist() == [10.0, 10.0]
