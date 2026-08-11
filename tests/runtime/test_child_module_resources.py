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
from tilefoundry.dsl import ConstTensor, DimVar, DimVarRangePat, Tensor, tf
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


_N = DimVar("n_resource", 1, 9)


@module(entry="run")
class _Broadcast:
    @func
    def run(x: Tensor[(_N,), "f32"], w: ConstTensor[(1,), "f32"]) -> Tensor[(_N,), "f32"]:
        return tf.mul(x, w)


def test_a_variant_this_dispatch_did_not_select_has_no_say() -> None:
    """Placement follows the body that runs, not every body that could have."""

    @module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
    class _Dispatch:
        near = _Broadcast
        far = _Broadcast

        @func
        def dispatch(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
            pass

        @dispatch.specialize(DimVarRangePat("n_resource", 1, 5))
        def _(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
            return near(x)  # noqa: F821

        @dispatch.specialize(DimVarRangePat("n_resource", 5, 9))
        def _(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
            return far(x)  # noqa: F821

    reading = _Dispatch.load(
        _Weights({"near.w": torch.full((1,), 2.0), "far.w": torch.ones(1, device="meta")})
    )

    assert torch.equal(reading.dispatch(torch.ones(4)), torch.full((4,), 2.0))


def test_a_child_only_a_converter_reaches_has_no_say_at_run_time() -> None:
    """A converter runs offline, so what it reaches is not reached by forward."""

    @module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
    class _ConverterOnly:
        scaled = _Scaled

        @func
        def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.add(x, w)

        @run.converter("w")
        def _convert_w(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return scaled(w)  # noqa: F821

    reading = _ConverterOnly.load(
        _Weights({"w": torch.zeros(4), "scaled.w": torch.ones(4, device="meta")})
    )

    assert torch.equal(reading.run(torch.ones(4)), torch.ones(4))


def test_preparation_stages_on_the_device_it_was_given() -> None:
    """A converter runs on the requested device, and what it produced stays there."""

    @module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
    class _Doubled:
        @func
        def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.add(x, w)

        @run.converter("w")
        def _convert_w(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.mul(w, w)

    staged: dict[str, object] = {}
    reading = _Doubled._prepare_into(
        _Weights({"w": torch.full((4,), 3.0)}), "", staged, "meta"
    )

    assert str(staged["w"].device) == "meta"
    assert str(reading.constants["w"].device) == "meta"


@module(entry="pick")
class _Nested:
    near = _Broadcast
    far = _Broadcast

    @func
    def pick(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
        pass

    @pick.specialize(DimVarRangePat("n_resource", 1, 5))
    def _(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
        return near(x)  # noqa: F821

    @pick.specialize(DimVarRangePat("n_resource", 5, 9))
    def _(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
        return far(x)  # noqa: F821


@module(entry="root", target=CudaTarget("nvidia.h200_sxm"))
class _OverDispatch:
    nested = _Nested

    @func
    def root(x: Tensor[(_N,), "f32"]) -> Tensor[(_N,), "f32"]:
        return nested(x)  # noqa: F821


def _over_dispatch_reading():
    return _OverDispatch.load(
        _Weights(
            {
                "nested.near.w": torch.full((1,), 2.0),
                "nested.far.w": torch.ones(1, device="meta"),
            }
        )
    )


def test_a_dispatch_inside_a_child_reaches_only_the_branch_it_selects() -> None:
    """The extents the run was given resolve a nested dispatch as execution will."""
    reading = _over_dispatch_reading()

    assert torch.equal(reading.root(torch.ones(3)), torch.full((3,), 2.0))


def test_a_branch_a_nested_dispatch_selects_is_placed_before_it_runs() -> None:
    """Its child's weight is what the run needs, so it decides the device too."""
    reading = _over_dispatch_reading()

    with pytest.raises(ValueError, match="nested.far.w"):
        reading.root(torch.ones(6))
