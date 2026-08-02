"""``check`` — comparing what an implementation produced against a reference.

Every output states its own predicates and their bounds; there is no default
bound. See [runtime §1.6](docs/spec/runtime.md#16-check).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Mapping, Sequence

import torch

from tilefoundry.evaluator.value import from_torch_dtype

_NEAR_ZERO = 1e-12

_SIGNED_BITS = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


@dataclass(frozen=True)
class Predicate:
    """One comparison, carrying the bound it holds the outputs to."""

    #: how the command line and the report name it
    name: ClassVar[str] = ""
    #: this predicate's own bound fields, in the order they are stated
    bounds: ClassVar[tuple[str, ...]] = ()
    #: false for a predicate that judges the candidate alone
    needs_reference: ClassVar[bool] = True
    #: true when it is meaningful on an integer or boolean output
    discrete: ClassVar[bool] = True
    #: one line, for `check --help`
    guidance: ClassVar[str] = ""

    def measure(
        self, candidate: torch.Tensor, reference: torch.Tensor | None
    ) -> "PredicateResult":
        raise NotImplementedError

    def _result(
        self, values: Mapping[str, float], passed: bool, note: str | None = None
    ) -> "PredicateResult":
        return PredicateResult(predicate=self, values=dict(values), passed=passed, note=note)


@dataclass(frozen=True)
class PredicateResult:
    """What one predicate measured on one output."""

    predicate: Predicate
    values: Mapping[str, float]
    passed: bool
    note: str | None = None


@dataclass(frozen=True)
class AllClose(Predicate):
    atol: float
    rtol: float

    name = "allclose"
    bounds = ("atol", "rtol")
    discrete = False
    guidance = "per-element; catches the one bad element an aggregate hides"

    def measure(self, candidate, reference):
        current, expected = candidate.float(), reference.float()
        slack = self.atol + self.rtol * expected.abs()
        violation = ((current - expected).abs() - slack).max().item()
        worst = max(violation, 0.0) if violation == violation else violation
        return self._result({"max_violation": worst}, violation <= 0.0)


@dataclass(frozen=True)
class RelL2(Predicate):
    max: float

    name = "rel_l2"
    bounds = ("max",)
    discrete = False
    guidance = "aggregate magnitude"

    def measure(self, candidate, reference):
        current, expected = candidate.float(), reference.float()
        distance = (current - expected).norm().item()
        reference_norm = expected.norm().item()
        if reference_norm <= _NEAR_ZERO:
            # Dividing by nothing produced a number with no scale to read it
            # against, so the bound applies to the distance itself.
            return self._result(
                {"absolute_l2": distance},
                distance <= self.max,
                "the reference norm is zero; this is an absolute L2 distance",
            )
        value = distance / reference_norm
        return self._result({"rel_l2": value}, value <= self.max)


@dataclass(frozen=True)
class Cosine(Predicate):
    min: float

    name = "cosine"
    bounds = ("min",)
    discrete = False
    guidance = "aggregate direction; catches a systematic bias"

    def measure(self, candidate, reference):
        current = candidate.float().flatten()
        expected = reference.float().flatten()
        current_norm, reference_norm = current.norm().item(), expected.norm().item()
        if current_norm <= _NEAR_ZERO and reference_norm <= _NEAR_ZERO:
            return self._result(
                {"cosine": 1.0}, 1.0 >= self.min, "both sides are entirely zero"
            )
        if current_norm <= _NEAR_ZERO or reference_norm <= _NEAR_ZERO:
            return self._result(
                {"cosine": 0.0}, False, "one side is entirely zero and the other is not"
            )
        value = (torch.dot(current, expected) / (current.norm() * expected.norm())).item()
        return self._result({"cosine": value}, value >= self.min)


@dataclass(frozen=True)
class Equal(Predicate):
    name = "equal"
    guidance = "bitwise; for discrete outputs, where one wrong value is total failure"

    def measure(self, candidate, reference):
        mismatched = int((candidate != reference).sum().item())
        return self._result(
            {"mismatched": float(mismatched), "elements": float(candidate.numel())},
            mismatched == 0,
        )


@dataclass(frozen=True)
class Ulp(Predicate):
    max: float

    name = "ulp"
    bounds = ("max",)
    discrete = False
    guidance = "distance in units of last place"

    def measure(self, candidate, reference):
        distance = (_ordered_bits(candidate) - _ordered_bits(reference)).abs().max().item()
        return self._result({"max_ulp": float(distance)}, distance <= self.max)


@dataclass(frozen=True)
class MaxAbs(Predicate):
    max: float

    name = "max_abs"
    bounds = ("max",)
    discrete = False
    guidance = "largest absolute difference"

    def measure(self, candidate, reference):
        value = (candidate.float() - reference.float()).abs().max().item()
        return self._result({"max_abs": value}, value <= self.max)


@dataclass(frozen=True)
class MaxRel(Predicate):
    max: float

    name = "max_rel"
    bounds = ("max",)
    discrete = False
    guidance = "largest relative difference"

    def measure(self, candidate, reference):
        current, expected = candidate.float(), reference.float()
        difference = (current - expected).abs()
        magnitude = expected.abs()
        # An element whose reference is zero has no relative error to state: it
        # either agrees exactly or it is unbounded.
        relative = torch.where(
            magnitude > 0,
            difference / magnitude.clamp_min(_NEAR_ZERO),
            torch.where(
                difference == 0, torch.zeros_like(difference), torch.full_like(difference, float("inf"))
            ),
        )
        value = relative.max().item()
        return self._result({"max_rel": value}, value <= self.max)


@dataclass(frozen=True)
class NanInf(Predicate):
    name = "nan_inf"
    needs_reference = False
    discrete = False
    guidance = "no NaN and no Inf; the one predicate needing no reference"

    def measure(self, candidate, reference=None):
        nans = int(torch.isnan(candidate).sum().item())
        infinities = int(torch.isinf(candidate).sum().item())
        return self._result(
            {"nan": float(nans), "inf": float(infinities)}, nans == 0 and infinities == 0
        )


#: Every predicate a caller may state, by the name it is stated under. `check
#: --help` lists these, so a predicate added here cannot be missing from it.
PREDICATES: Mapping[str, type[Predicate]] = {
    predicate.name: predicate
    for predicate in (AllClose, RelL2, Cosine, Equal, Ulp, MaxAbs, MaxRel, NanInf)
}


def _ordered_bits(tensor: torch.Tensor) -> torch.Tensor:
    """A float tensor's bits as integers ordered the way the floats are, so their
    difference is a count of representable values."""
    try:
        as_integer = tensor.view(_SIGNED_BITS[tensor.dtype])
    except KeyError:
        raise ValueError(
            f"check: ulp is defined on a float output, not on {tensor.dtype}"
        ) from None
    bits = as_integer.to(torch.int64)
    sign = 1 << (tensor.element_size() * 8 - 1)
    # A negative float's magnitude grows as its signed bit pattern falls, so the
    # negative half is reflected rather than used as it stands.
    return torch.where(bits >= 0, bits, -(bits + sign))


@dataclass(frozen=True)
class OutputCheck:
    """One output: what it is, and what every predicate said about it."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    ref_norm: float | None
    results: tuple[PredicateResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


@dataclass(frozen=True)
class Report:
    """What one `check` established, per output."""

    outputs: tuple[OutputCheck, ...]

    @property
    def passed(self) -> bool:
        return all(output.passed for output in self.outputs)


def _flatten(x, path: str = "output") -> list[tuple[str, torch.Tensor]]:
    """Flatten a tensor or nested tuple-of-tensors into ``[(path, tensor), ...]``;
    the path list doubles as a structural signature for comparing outputs."""
    if isinstance(x, torch.Tensor):
        return [(path, x)]
    if isinstance(x, tuple):
        leaves: list[tuple[str, torch.Tensor]] = []
        for i, item in enumerate(x):
            leaves.extend(_flatten(item, f"{path}[{i}]"))
        return leaves
    raise TypeError(f"check: {path} must be a torch.Tensor or tuple thereof, got {type(x).__name__}")


def _admit(path: str, tensor: torch.Tensor, predicates: Sequence[Predicate]) -> None:
    """Refuse a predicate that says nothing about an output of this dtype."""
    if tensor.dtype.is_floating_point:
        return
    for predicate in predicates:
        if predicate.discrete:
            continue
        raise ValueError(
            f"check: {path} is {from_torch_dtype(tensor.dtype).name}; "
            f"{predicate.name} is not meaningful on a discrete output. One wrong "
            f"value is a tiny numerical deviation and a completely wrong result. "
            f"Use equal."
        )


def _expectations(
    paths: Sequence[str], expect: Mapping[str, Sequence[Predicate]], has_reference: bool
) -> None:
    """Refuse an expectation set that does not cover exactly what was produced."""
    unknown = sorted(set(expect) - set(paths))
    if unknown:
        raise ValueError(
            f"check: {unknown} name no output; this run produced {list(paths)}"
        )
    for path in paths:
        stated = expect.get(path) or ()
        if not stated:
            raise ValueError(
                f"check: no comparison requested for output {path!r}. Every output "
                f"needs at least one predicate. There is no default: a default bound "
                f"that cannot be met trains you to ignore FAIL. A single f32->bf16 "
                f"rounding is already rel_l2 = 1.66e-3."
            )
        if has_reference:
            continue
        two_sided = sorted({p.name for p in stated if p.needs_reference})
        if two_sided:
            raise ValueError(
                f"check: {path!r} asks for {two_sided} with no reference to compare "
                f"against. Without a reference only "
                f"{sorted(n for n, p in PREDICATES.items() if not p.needs_reference)} "
                f"can be measured."
            )


def check(
    candidate: Callable,
    reference: Callable | None,
    inputs: tuple,
    *,
    expect: Mapping[str, Sequence[Predicate]],
) -> Report:
    """Run *candidate* — and *reference* when there is one — on the same *inputs*,
    and measure each output against the predicates *expect* states for it.

    A result may be a bare tensor or an arbitrarily nested tuple of tensors. With
    a reference, the two must flatten to the same structure, shapes and dtypes.
    """
    produced = _flatten(candidate(*inputs))
    if not produced:
        raise ValueError("check: the candidate produced no tensor, so nothing was compared")
    paths = [path for path, _ in produced]
    _expectations(paths, expect, reference is not None)

    expected: dict[str, torch.Tensor] = {}
    if reference is not None:
        reference_leaves = _flatten(reference(*inputs))
        reference_paths = [path for path, _ in reference_leaves]
        if paths != reference_paths:
            raise ValueError(
                f"check: candidate/reference output structures differ: "
                f"{paths} vs {reference_paths}"
            )
        for (path, current), (_, other) in zip(produced, reference_leaves):
            if current.shape != other.shape or current.dtype != other.dtype:
                raise ValueError(
                    f"check: {path} shape/dtype mismatch: candidate "
                    f"{tuple(current.shape)}/{current.dtype} vs reference "
                    f"{tuple(other.shape)}/{other.dtype}"
                )
            expected[path] = other

    outputs = []
    for path, current in produced:
        predicates = tuple(expect[path])
        _admit(path, current, predicates)
        other = expected.get(path)
        outputs.append(
            OutputCheck(
                path=path,
                shape=tuple(current.shape),
                dtype=from_torch_dtype(current.dtype).name,
                ref_norm=None if other is None else other.float().norm().item(),
                results=tuple(predicate.measure(current, other) for predicate in predicates),
            )
        )
    return Report(outputs=tuple(outputs))


__all__ = [
    "PREDICATES",
    "AllClose",
    "Cosine",
    "Equal",
    "MaxAbs",
    "MaxRel",
    "NanInf",
    "OutputCheck",
    "Predicate",
    "PredicateResult",
    "RelL2",
    "Report",
    "Ulp",
    "check",
]
