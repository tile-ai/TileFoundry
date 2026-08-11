"""Which reading supplies a child-Module call's constants, and from where.

A composed root holds no constants: each reached child fills its own from the
reading it was loaded into. What that makes checkable before anything runs is a
child whose bindings do not cover its declared weights, and a child weight that
disagrees with the activations about where it lives. Preparation runs the same
rule in reverse, staging children before the converters that may call them.

No GPU: a second device is spelled ``meta``, which needs no card.
"""

from __future__ import annotations

import pytest
import torch

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.target import CudaTarget


class _Weights:
    """A resource whose subtree is the dotted prefix, like a checkpoint's."""

    def __init__(self, values: dict) -> None:
        self.values = values

    def load(self, name: str):
        return self.values[name]

    def load_group(self, name: str):
        return None

    def subtree(self, name: str) -> "_Weights":
        prefix = f"{name}."
        return _Weights(
            {k[len(prefix):]: v for k, v in self.values.items() if k.startswith(prefix)}
        )


@module(entry="run")
class _Scaled:
    @func
    def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.mul(x, w)


@module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
class _Fused:
    scaled = _Scaled

    @func
    def fused(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return scaled(x)  # noqa: F821


def test_reading_a_resource_without_a_child_weight_is_refused() -> None:
    with pytest.raises(KeyError, match="missing declared weight 'w'"):
        _Fused.load(_Weights({}))


def test_a_reached_child_that_cannot_cover_its_constants_is_named() -> None:
    """The check before execution, for a reading assembled outside ``load``.

    ``prepare`` builds one as it stages, so the guard is not reachable only
    through the loader that already refuses a resource with the weight absent.
    """
    reading = _Fused.load(_Weights({"scaled.w": torch.ones(4)}))
    stripped = type(reading)(
        module=reading.module,
        constants=reading.constants,
        modules=tuple(
            type(child)(module=child.module, constants={}, modules=child.modules)
            for child in reading.modules
        ),
    )

    with pytest.raises(KeyError, match="scaled.*'w'"):
        stripped.fused(torch.ones(4))


def test_an_attached_child_no_call_reaches_has_no_say_in_placement() -> None:
    @module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
    class _WithSpare:
        scaled = _Scaled
        spare = _Scaled

        @func
        def fused(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return scaled(x)  # noqa: F821

    reading = _WithSpare.load(
        _Weights({"scaled.w": torch.ones(4), "spare.w": torch.ones(4, device="meta")})
    )

    assert torch.equal(reading.fused(torch.ones(4)), torch.ones(4))


def test_a_child_weight_elsewhere_is_refused_before_anything_runs() -> None:
    elsewhere = torch.ones(4, device="meta")
    reading = _Fused.load(_Weights({"scaled.w": elsewhere}))

    with pytest.raises(ValueError, match="more than one device"):
        reading.fused(torch.ones(4))
    assert str(reading.scaled.constants["w"].device) == "meta"


def test_a_converter_may_call_a_child_staged_before_it(tmp_path) -> None:
    @module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
    class _Converted:
        scaled = _Scaled

        @func
        def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.add(x, w)

        @run.converter("w")
        def _convert_w(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return scaled(w)  # noqa: F821

    raw = _Weights({"w": torch.full((4,), 3.0), "scaled.w": torch.full((4,), 2.0)})
    _Converted.prepare(raw, str(tmp_path), device="cpu")

    from safetensors.torch import load_file  # noqa: PLC0415 — optional runtime dep

    prepared = load_file(str(tmp_path / "model-00001-of-00001.safetensors"))
    assert torch.equal(prepared["w"], torch.full((4,), 6.0))
    assert torch.equal(prepared["scaled.w"], torch.full((4,), 2.0))
