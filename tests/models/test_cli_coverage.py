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
told.
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

    assert cli.main(["analyze", source, "--compute-cost"]) == 0


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_every_analysis_the_cli_offers_runs_on_a_real_model(
    model, tmp_path, capsys
) -> None:
    source = _source_for(model, ACCEPTANCE(), tmp_path)

    assert cli.main(["analyze", source, *_ANALYSES]) == 0
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_the_cli_reports_a_real_model_as_json(model, tmp_path, capsys) -> None:
    source = _source_for(model, ACCEPTANCE(), tmp_path)

    assert cli.main(["analyze", source, "--compute-cost", "--json"]) == 0
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

    assert cli.main(["schedule", source, "--topology", fixture.level(case.topology).name]) == 0
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("model", _models(), ids=_identify)
def test_the_cli_reads_the_machine_off_the_printed_source(
    model, tmp_path, capsys
) -> None:
    """Nothing tells the CLI which target to use; the file has to say."""
    source = _source_for(model, ACCEPTANCE(), tmp_path)

    assert cli.main(["inspect", "capabilities", source]) == 0
    assert capsys.readouterr().out.strip()
