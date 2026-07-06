"""DSL parsing — happy-path uses the real ``@func`` decorator (returns ``hir.Function``).

Negative / error-diagnostic tests live in ``tests/parser/test_errors.py``
where dynamic source-string + ``pytest.raises`` is the natural fit.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.relu import ReLU

# ── Typical op shapes via real @func authoring ───────────────────────────


@func
def _relu_call(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return relu(x)


@func
def _add_call(
    a: Tensor[(8,), "f32"], b: Tensor[(8,), "f32"],
) -> Tensor[(8,), "f32"]:
    return add(a, b)


@func
def _matmul_call(
    a: Tensor[(4, 8), "f32"], b: Tensor[(8, 16), "f32"],
) -> Tensor[(4, 16), "f32"]:
    return matmul(a, b)


def test_parse_typical_op_call_shapes() -> None:
    """``relu(x)`` / ``add(a, b)`` / ``matmul(a, b)`` parse to the
    expected ``Call(target=Op, args=...)`` IR shape."""
    body = _relu_call.body
    assert isinstance(body, Call) and isinstance(body.target, ReLU)

    body = _add_call.body
    assert isinstance(body, Call) and isinstance(body.target, Binary)
    assert body.target.kind is BinaryKind.ADD
    assert len(body.args) == 2

    body = _matmul_call.body
    assert isinstance(body, Call) and isinstance(body.target, MatMul)


# ── Namespace callee form (``tf.add(...)`` / ``T.copy(...)``) ───────────


from tilefoundry import dsl  # noqa: E402
from tilefoundry.dsl import tf  # noqa: E402  -- test fixture closure capture
from tilefoundry.ir.core.kinds import BinaryKind  # noqa: E402
from tilefoundry.ir.hir.math.binary import Binary  # noqa: E402


@func
def _tf_namespace_add(
    a: Tensor[(8,), "f32"], b: Tensor[(8,), "f32"],
) -> Tensor[(8,), "f32"]:
    return tf.add(a, b)


def test_parse_tf_namespace_attribute_callee() -> None:
    """``tf.add(a, b)`` parses to the same kinded ``Binary`` Call as
    the bare ``add(a, b)`` form."""
    body = _tf_namespace_add.body
    assert isinstance(body, Call) and isinstance(body.target, Binary)
    assert body.target.kind is BinaryKind.ADD
    assert len(body.args) == 2


# ── insert_slice surface (dynamic-update-slice) ──────────────────────────


@func
def _insert_slice_call(
    dst: Tensor[(8,), "f32"], upd: Tensor[(3,), "f32"], off: Tensor[(1,), "i32"],
) -> Tensor[(8,), "f32"]:
    return tf.insert_slice(dst, upd, off)


def test_parse_insert_slice() -> None:
    """``tf.insert_slice(dst, update, offsets)`` parses to an ``InsertSlice``
    Call with the three tensor inputs."""
    from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice  # noqa: PLC0415

    body = _insert_slice_call.body
    assert isinstance(body, Call) and isinstance(body.target, InsertSlice)
    assert len(body.args) == 3


def test_no_write_row_surface() -> None:
    """The public surface is ``insert_slice`` only — the ``write_row`` sugar is
    not exposed."""
    import pytest  # noqa: PLC0415

    with pytest.raises(AttributeError):
        _ = tf.write_row


# ── TIR DSL surface accessible ───────────────────────────────────────────


def test_tir_effect_ops_resolve_through_dsl_surface() -> None:
    """TIR effect ops resolve through ``T``."""

    # Each TIR Op has an OpSchema and is reachable via T.<name>.
    for name in ("copy", "fill", "mma", "rms_norm", "reduce", "alloc_tensor"):
        builder = getattr(dsl.T, name)
        assert callable(builder), f"tilefoundry.dsl.T.{name} did not resolve"
