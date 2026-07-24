"""Leaf implementation registry + post_init caching (M1 of
docs/plans/agent-kernel-loop/P0a-tonight-nested-module-e2e.md).

A tiny, self-contained module tree (independent of any real model fixture)
demonstrates the mechanism the DeepSeek V4 flash decode-step fixture (M2/M3)
then reuses: (a) plain evaluator recursion and (b) partial / full leaf
registration produce the same result, a registered leaf is intercepted at
any nesting depth (not only at the top-level ``evaluate()`` call), and a
module's ``post_init`` weight hook runs exactly once regardless of how many
times its (already-loaded) weights are reused.
"""
from __future__ import annotations

import torch

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.leaf import ImplementationPackage, LeafRegistry, WeightLoader, leaf_paths
from tilefoundry.ir.core.module import Module


@func
def scale_leaf(x: Tensor[(4,), "f32"], w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.mul(x, w)


@func
def double(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.add(x, x)


@func
def root_fn(x: Tensor[(4,), "f32"], w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    scaled = scale_leaf(x, w)
    return double(scaled)


def _double_weight_post_init(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Stand-in for a real quantized-weight conversion: doubles ``w`` once."""
    return {"w": weights["w"] * 2}


def _build_tree() -> Module:
    leaf_module = Module(
        name="leaf", functions=(scale_leaf,), entry="scale_leaf",
        post_init=_double_weight_post_init,
    )
    mid_module = Module(name="mid", functions=(double,), entry="double")
    return Module(
        name="root", functions=(root_fn,), modules=(leaf_module, mid_module), entry="root_fn",
    )


def _torch_scale_leaf(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * w


def _torch_double(x: torch.Tensor) -> torch.Tensor:
    return x + x


class TestWeightLoaderPostInit:
    def test_post_init_transforms_only_its_own_namespace(self) -> None:
        root = _build_tree()
        loader = WeightLoader(root)
        raw = {"leaf.w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        loaded = loader.load(raw)
        torch.testing.assert_close(loaded["leaf.w"], raw["leaf.w"] * 2)

    def test_post_init_runs_once_and_is_cached(self) -> None:
        root = _build_tree()
        loader = WeightLoader(root)
        raw = {"leaf.w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        first = loader.load(raw)
        second = loader.load(raw)
        assert loader.post_init_runs == 1
        torch.testing.assert_close(first["leaf.w"], second["leaf.w"])


class TestLeafInterception:
    def test_pure_evaluator_and_partial_leaf_registration_agree(self) -> None:
        """(a) plain evaluator vs (b) only ``scale_leaf`` intercepted (nested
        inside root_fn's body — ``double`` still falls back to the plain
        evaluator) must agree exactly (the leaf impl is bit-identical math)."""
        root = _build_tree()
        loader = WeightLoader(root)
        raw = {"leaf.w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        w = loader.load(raw)["leaf.w"]
        x = torch.tensor([5.0, 6.0, 7.0, 8.0])

        reference = evaluate(root_fn, x, w, device="cpu")

        paths = leaf_paths(root)
        registry = LeafRegistry()
        registry.register(
            paths["scale_leaf"], "scale_leaf",
            ImplementationPackage(language="torch", fn_or_source=_torch_scale_leaf, entry="scale_leaf"),
        )
        assert len(registry) == 1
        partial = evaluate(root_fn, x, w, device="cpu", leaves=registry.by_function_name())
        torch.testing.assert_close(partial, reference)

    def test_fully_registered_agrees_too(self) -> None:
        """(b) every leaf registered (both ``scale_leaf`` and ``double``)
        still agrees with the plain evaluator."""
        root = _build_tree()
        loader = WeightLoader(root)
        raw = {"leaf.w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        w = loader.load(raw)["leaf.w"]
        x = torch.tensor([5.0, 6.0, 7.0, 8.0])

        reference = evaluate(root_fn, x, w, device="cpu")

        paths = leaf_paths(root)
        registry = LeafRegistry()
        registry.register(
            paths["scale_leaf"], "scale_leaf",
            ImplementationPackage(language="torch", fn_or_source=_torch_scale_leaf, entry="scale_leaf"),
        )
        registry.register(
            paths["double"], "double",
            ImplementationPackage(language="torch", fn_or_source=_torch_double, entry="double"),
        )
        full = evaluate(root_fn, x, w, device="cpu", leaves=registry.by_function_name())
        torch.testing.assert_close(full, reference)

    def test_no_registration_is_the_pre_m1_evaluator(self) -> None:
        """``leaves`` omitted is exactly the pre-M1 evaluate() call — the
        backward-compat contract M0/M1 both promise."""
        root = _build_tree()
        w = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x = torch.tensor([5.0, 6.0, 7.0, 8.0])
        assert root is not None  # tree is unused by this call; only its functions are
        without_kw = evaluate(root_fn, x, w, device="cpu")
        with_none = evaluate(root_fn, x, w, device="cpu", leaves=None)
        torch.testing.assert_close(without_kw, with_none)

    def test_leaf_registered_at_top_level_call(self) -> None:
        """A registered leaf intercepts even when it is itself the top-level
        ``evaluate()`` target, not only when nested inside a caller."""
        w = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x = torch.tensor([5.0, 6.0, 7.0, 8.0])
        registry = LeafRegistry()
        registry.register(
            (), "scale_leaf",
            ImplementationPackage(language="torch", fn_or_source=_torch_scale_leaf, entry="scale_leaf"),
        )
        out = evaluate(scale_leaf, x, w, device="cpu", leaves=registry.by_function_name())
        torch.testing.assert_close(out, x * w)
