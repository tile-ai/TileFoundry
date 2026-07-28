"""CLI coverage for the public pipeline scheduling boundary."""

from __future__ import annotations

import textwrap

from tilefoundry import cli

_NESTED_MODULE = """
    from tilefoundry import func, module
    from tilefoundry.dsl import Tensor
    from tilefoundry.dsl.tf import matmul, rms_norm
    from tilefoundry.ir.types.shard import Topology

    @module(entry="root", target="cuda")
    class Model:
        topologies = (Topology("cta", 1), Topology("thread", 128))

        @func
        def root(x: Tensor[(16, 16), "bf16"], w: Tensor[(16, 16), "bf16"], weight: Tensor[(16,), "f32"]) -> Tensor[(16, 16), "bf16"]:
            h = matmul(x, w)
            return rms_norm(h, weight)

        @module(entry="inner")
        class child:
            @func
            def inner(x: Tensor[(16, 16), "bf16"], w: Tensor[(16, 16), "bf16"], weight: Tensor[(16,), "f32"]) -> Tensor[(16, 16), "bf16"]:
                h = matmul(x, w)
                return rms_norm(h, weight)
"""


def test_schedule_selects_a_nested_function_through_public_schedule(tmp_path, capsys) -> None:
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_NESTED_MODULE), encoding="utf-8")

    assert cli.main(["schedule", f"{path}:Model.child.inner", "--topology", "thread", "--json"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert '"architecture_id": "nvidia.sm90"' in captured.out
    assert '"statement_id": "MM"' in captured.out


def test_the_solver_budget_is_stated_and_reaches_the_operation(tmp_path, capsys, monkeypatch) -> None:
    """A budget the caller cannot state is a configuration nobody can reproduce.

    The solver's default worker count sizes itself to the machine, which is right
    for one schedule and wrong for several at once. Checked by intercepting the
    public operation rather than by timing anything: what matters is that the two
    numbers the caller wrote arrive, and a run that merely succeeded would say
    nothing about whether they did.
    """
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_NESTED_MODULE), encoding="utf-8")

    seen = {}
    real = cli.schedule

    def record(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "schedule", record)
    assert (
        cli.main(
            [
                "schedule",
                f"{path}:Model.child.inner",
                "--topology",
                "thread",
                "--solver-timeout=12.5",
                "--solver-workers=3",
            ]
        )
        == 0
    )

    assert seen["options"].timeout_seconds == 12.5
    assert seen["options"].workers == 3
    assert capsys.readouterr().err == ""


def test_an_unstated_solver_budget_leaves_the_operation_its_own(tmp_path, capsys, monkeypatch) -> None:
    """Omitting the flags states nothing, rather than restating a default here.

    Two places holding the same default is one place for them to disagree, and the
    disagreement would be invisible: both runs schedule successfully.
    """
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_NESTED_MODULE), encoding="utf-8")

    seen = {}
    real = cli.schedule

    def record(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "schedule", record)
    assert cli.main(["schedule", f"{path}:Model.child.inner", "--topology", "thread"]) == 0

    assert seen["options"] is None
    assert capsys.readouterr().err == ""


def test_asking_for_the_first_plan_reaches_the_operation(tmp_path, capsys, monkeypatch) -> None:
    """`--first-plan` states that a plan is wanted rather than the best plan.

    Checked at the boundary it crosses rather than by timing: the search that would
    show a difference is one this fixture is too small to have, and a timing
    assertion on a shared machine measures the machine.
    """
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_NESTED_MODULE), encoding="utf-8")

    seen = {}
    real = cli.schedule

    def record(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "schedule", record)
    assert (
        cli.main(
            ["schedule", f"{path}:Model.child.inner", "--topology", "thread", "--first-plan"]
        )
        == 0
    )

    assert seen["options"].stop_at_first_solution is True
    # The bound on a search that has found nothing is still there.
    assert seen["options"].timeout_seconds == 60.0
    assert capsys.readouterr().err == ""
