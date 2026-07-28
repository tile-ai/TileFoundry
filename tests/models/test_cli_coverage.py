"""The CLI, given a real model, through the door a user comes in by.

A user does not hand us a Module. They hand us a file, and the CLI imports it.
So these tests print the corpus model back out as DSL source, write it to a
file, and call `cli.main` with a path -- the import, the parse, the type check,
the analysis and the rendering all run. A test that reached past that and
called an internal helper would still pass on a model whose printed form does
not parse, which is the failure most worth catching: the printed form is the
only artifact anyone can hand to somebody else.

The source is produced from a Target-bound build, so the file states its own
machine and topology levels and the CLI has to read them back rather than be
told. The one thing the file cannot state is the size to ask about: a model
authored for decode leaves its context length open on purpose, so the length
travels as a `--dim` argument, taken from the same registry entry the in-process
tests use rather than written out again here.
"""

from __future__ import annotations

import json

import pytest

from tests.models.corpus import ModelCase, TargetFixture
from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tilefoundry import cli
from tilefoundry.inspection import as_script

_ANALYSES = ("--compute-cost", "--memory", "--roofline", "--timeline")

#: The same solver budget the in-process schedule witnesses use, stated to the CLI
#: rather than left to the library default.
#:
#: The default worker count lets the solver size itself to the machine; with the suite
#: running several of these at once that oversubscribes it and none of them returns an
#: incumbent, which looks like a model that cannot be scheduled and is not -- measured:
#: which models failed changed from run to run.
#:
#: `--first-plan` because what this asks is whether the CLI can schedule the printed
#: model at all. Searching for the best plan spends the whole budget on every one of
#: these, which is time bought for an answer no assertion here reads.
_SOLVER_ARGS = ("--solver-timeout=60", "--solver-workers=4", "--first-plan")


def _dims_of(model: ModelCase) -> dict[str, int]:
    """Every extent *model* states, across all its cases.

    The CLI is given the printed module, and what it analyses is that module's
    entry function -- so the extents it needs are the ones the model states
    anywhere, not the ones one case happens to state. Reading them off a single
    case worked only while the case that stated them was also the last one listed,
    which is a coincidence of ordering rather than a fact about the model.

    A model states one set of extents; a case that disagreed about the same
    dimension would make "the length the CLI is asked about" two answers, so that
    is refused here rather than resolved.
    """
    dims: dict[str, int] = {}
    for case in (*model.analyze, *model.schedule, *model.sized):
        for name, extent in (case.dims or {}).items():
            if dims.setdefault(name, extent) != extent:
                raise AssertionError(
                    f"{model.id} states {name}={dims[name]} and {name}={extent}; "
                    f"one model states one extent per dimension"
                )
    return dims


def _dim_args(model: ModelCase) -> list[str]:
    """The `--dim` arguments for every extent *model* states."""
    return [f"--dim={name}={extent}" for name, extent in _dims_of(model).items()]


def _source_for(model: ModelCase, fixture: TargetFixture, directory) -> str:
    """The model, aimed at one machine, as source a user could have written."""
    built = model.build_for(fixture)
    path = directory / f"{model.id}.py"
    path.write_text(as_script(built, module=model.entry), encoding="utf-8")
    return str(path)


def _models() -> list[ModelCase]:
    return list(CORPUS)


def _identify(model: ModelCase) -> str:
    return model.id


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_the_printed_model_is_source_the_cli_can_import(model, tmp_path) -> None:
    """Print, write, import, type-check. A model that cannot make this trip
    has no form anyone can pass around, however well it analyses in memory."""
    source = _source_for(model, ACCEPTANCE(), tmp_path)

    assert cli.main(["analyze", source, "--compute-cost", *_dim_args(model)]) == 0


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_every_analysis_the_cli_offers_runs_on_a_real_model(
    model, tmp_path, capsys
) -> None:
    source = _source_for(model, ACCEPTANCE(), tmp_path)

    assert cli.main(["analyze", source, *_ANALYSES, *_dim_args(model)]) == 0
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_the_cli_reports_a_real_model_as_json(model, tmp_path, capsys) -> None:
    source = _source_for(model, ACCEPTANCE(), tmp_path)

    assert (
        cli.main(
            ["analyze", source, "--compute-cost", "--json", *_dim_args(model)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_the_cli_schedules_a_real_model_at_a_declared_level(
    model, tmp_path, capsys
) -> None:
    """The level comes from the fixture rather than a literal, so a fixture
    that stops declaring it fails here instead of testing nothing."""
    fixture = ACCEPTANCE()
    case = model.schedule[0]
    source = _source_for(model, fixture, tmp_path)

    assert (
        cli.main(
            [
                "schedule",
                source,
                "--topology",
                fixture.level(case.topology).name,
                *_dim_args(model),
                *_SOLVER_ARGS,
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_the_cli_reads_the_machine_off_the_printed_source(
    model, tmp_path, capsys
) -> None:
    """Nothing tells the CLI which target to use; the file has to say."""
    source = _source_for(model, ACCEPTANCE(), tmp_path)

    assert cli.main(["inspect", "capabilities", source]) == 0
    assert capsys.readouterr().out.strip()
