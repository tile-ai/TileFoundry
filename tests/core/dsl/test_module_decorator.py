"""Module collection and validation of class-body declarations."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest
import torch

from tilefoundry import cli, runtime, utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core, expr

FIXTURES = Path(__file__).parents[1] / "fixtures"
CLASS_BODIES = FIXTURES / "dsl" / "class_body_modules.py"
WEIGHTED = FIXTURES / "dsl" / "weighted_modules.py"
DIAGNOSTICS = FIXTURES / "diagnostics"


def test_module_collects_functions_in_order():
    """Collected functions keep definition order and resolve by name.

    Members land on the Module in definition order and each collected name
    resolves through ``lookup`` (the IR-node path — bare attribute access
    instead returns a runnable callable, see ``ir.core.module.Module``); a name
    the class never declared raises ``AttributeError``.
    """
    _Demo = cli.load_authored_ir(f"{CLASS_BODIES}:_Demo")
    assert _Demo.name == "_Demo"
    assert [fn.name for fn in _Demo.functions] == ["leaf", "composed"]
    assert _Demo.lookup("leaf").name == "leaf"
    assert _Demo.lookup("composed").name == "composed"
    with pytest.raises(AttributeError):
        _Demo.not_a_function


def test_module_refuses_ambiguous_attribute():
    """A duplicated function name is ambiguous; real Module fields are not.

    A duplicated function name is ambiguous under attribute access (raises),
    real Module fields are never intercepted, and ``lookup`` refuses ambiguous
    matches — the core-ir [parser §2](docs/spec/parser.md#2-syntax-and-rules) ambiguity rule.
    """
    base = cli.load_authored_ir(f"{CLASS_BODIES}:_Demo").lookup("leaf")
    dup_a = dataclasses.replace(base, name="dup")
    dup_b = dataclasses.replace(base, name="dup")
    mod = core.Module(name="Dup", functions=(dup_a, dup_b), entry="dup")
    assert mod.name == "Dup"
    assert len(mod.functions) == 2
    with pytest.raises(AttributeError):
        mod.dup
    with pytest.raises(ValueError, match="exactly one"):
        mod.lookup("dup")


def test_module_accepts_three_member_kinds():
    """A plain Python method is the third member kind.

    A plain Python method is the third member kind — an orchestration method,
    bound on the resulting Module. A member that is none of the three kinds
    (DSL function / child Module / plain function) is still rejected.
    """
    _WithMethod = cli.load_authored_ir(f"{CLASS_BODIES}:_WithMethod")

    assert _WithMethod.describe() == "_WithMethod/only"

    with pytest.raises(TypeError, match="only these three member kinds"):
        cli.load_authored_ir(str(DIAGNOSTICS / "dsl_module_constant_member.py"))


def test_module_refuses_empty_body():
    """Only an empty body is refused: one function, one child, or one plain method is enough.

    Only an empty body is refused: one function, one child, or one plain method
    is enough, so a body of methods alone — refused before for owning no function —
    is valid. A supplied ``entry`` is still checked, and a class-body ``__call__``
    is refused rather than dropped.
    """
    with pytest.raises(TypeError, match="empty class body"):
        cli.load_authored_ir(str(DIAGNOSTICS / "dsl_module_empty_body.py"))

    _MethodsOnly = cli.load_authored_ir(f"{CLASS_BODIES}:_MethodsOnly")

    assert _MethodsOnly.entry is None
    assert _MethodsOnly.walk(3) == 3

    with pytest.raises(ValueError, match=r"entry 'absent' names no collected function"):
        cli.load_authored_ir(str(DIAGNOSTICS / "dsl_module_absent_entry.py"))

    with pytest.raises(TypeError, match=r"__call__ has no effect"):
        cli.load_authored_ir(str(DIAGNOSTICS / "dsl_module_class_call.py"))


def test_stepless_module_reports_no_entry():
    """A bare call is answered by naming what to call.

    A bare call is answered by naming what to call; asking for the entry is
    answered by saying there is no default step. Neither may read as ``entry``
    being wrong, which is what ``None`` reaching the entry lookup produces.
    """
    _NoStep = cli.load_authored_ir(f"{CLASS_BODIES}:_NoStep")

    with pytest.raises(ValueError, match=r"declares no entry"):
        _NoStep()

    with pytest.raises(ValueError, match=r"declares no entry, so it has no default step"):
        _NoStep.entry_function()

    assert _NoStep.lookup("helper").name == "helper"


def test_module_runner_takes_weights():
    """A Module's callable takes every declared param, a ``ConstTensor`` one included.

    A Module's callable takes every declared param, a ``ConstTensor`` one
    included; a ``LoadedModule``'s takes activations alone. Each wrong argument
    list is refused naming the runner it wanted, and neither is sent to the
    other's.
    """
    _Weighted = cli.load_authored_ir(f"{WEIGHTED}:_Weighted")

    ones = torch.ones(2, dtype=torch.float32)
    weight = torch.full((2,), 3.0)

    assert evaluator.evaluate(_Weighted.scale, ones, weight).float().cpu().tolist() == [3.0, 3.0]

    with pytest.raises(evaluator.EvalError, match=r"expects 2 inputs, got 1"):
        evaluator.evaluate(_Weighted.scale, ones)

    loaded = _Weighted.load(runtime.DictResource({"w": weight}))
    with pytest.raises(evaluator.EvalError, match=r"expects 1 activation") as excinfo:
        evaluator.evaluate(loaded.scale, ones, weight)
    assert "load(resource)" not in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        evaluator.evaluate(_Weighted.load(runtime.DictResource({})).scale, ones)
    refused = str(excinfo.value)
    assert "missing declared weight 'w'" in refused
    assert "prepare produces it" in refused
    assert (
        utils.spec_ref_render(
            "[runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)"
        )
        in refused
    )


def test_shared_child_binds_per_owner():
    """Two owners over one child IR read their own subtrees rather than the last one loaded winning.

    Two owners over one child IR read their own subtrees rather than the last
    one loaded winning.
    """
    _Owner = cli.load_authored_ir(f"{WEIGHTED}:_Owner")

    def aliasing(name):

        node = copy.copy(_Owner)
        object.__setattr__(node, "name", name)
        return node

    left, right = aliasing("left"), aliasing("right")
    assert left.children["leaf"] is right.children["leaf"]

    ones = torch.ones(2, dtype=torch.float32)
    loaded_left = left.load(runtime.DictResource({"leaf.w": torch.full((2,), 3.0)}))
    loaded_right = right.load(runtime.DictResource({"leaf.w": torch.full((2,), 10.0)}))

    assert loaded_left.children["leaf"].module is loaded_right.children["leaf"].module
    assert evaluator.evaluate(loaded_left.leaf, ones).float().cpu().tolist() == [3.0, 3.0]
    assert evaluator.evaluate(loaded_right.leaf, ones).float().cpu().tolist() == [10.0, 10.0]


def test_forward_reference_sibling_fails_loudly():
    """Test forward reference sibling fails loudly.

    A method that calls a sibling defined *below* it cannot resolve the
    sibling (only callee-before-caller is supported) and raises rather than
    silently mis-parsing the call.
    """
    with pytest.raises(expr.VerifyError):
        cli.load_authored_ir(str(DIAGNOSTICS / "dsl_module_forward_sibling.py"))
