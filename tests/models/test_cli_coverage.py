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

_ZERO_SIZED_BINDINGS = {
    "qwen3_1_7b": frozenset(("k_cache", "v_cache")),
    "qwen2_5_1_5b": frozenset(("k_cache", "v_cache")),
    "gemma2_2b": frozenset(("k_cache", "v_cache")),
    "minicpm3_4b": frozenset(("k_cache", "v_cache")),
    "deepseek_v4_flash": frozenset(("kv_cache", "k_ctx", "score_ctx", "p_ctx", "weighted")),
    "qwen3_5_35b_a3b_full_attention": frozenset(
        ("k_cache", "v_cache", "k_ctx", "score_ctx", "p_ctx", "weighted")
    ),
    "kimi_linear_48b_a3b": frozenset(("k_cache", "v_cache", "score_ctx", "p_ctx", "weighted")),
}

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
    """The `--dim` arguments for the kernel the CLI is asked about.

    The selected case's own extents, not the model's union: the CLI is asked about
    one kernel, and a length stated for a different kernel of the same model is a
    dimension this one does not have. `_dims_of` still holds the model to one
    extent per dimension, which is a claim about the description rather than about
    any one invocation.
    """
    stated = model.schedule[0].dims or {}
    return [f"--dim={name}={extent}" for name, extent in stated.items()]


def _source_for(
    model: ModelCase, fixture: TargetFixture, directory, selector: str | None = None
) -> str:
    """The model, aimed at one machine, as source a user could have written, and
    the selector naming which kernel of it the CLI is asked about.

    The corpus selection the caller asks about, defaulting to the model's schedule
    selection. Stated rather than left to the CLI's default, because a root that
    composes child Modules declares no step of its own -- there is nothing for a
    default to pick -- and the corpus already says which kernel it means.

    What is printed is the selected execution Module, which carries the machine it
    resolved through its owners.
    """
    selector = model.schedule[0].selector if selector is None else selector
    selected, _ = model.resolve(model.build_for(fixture), selector)
    path = directory / f"{model.id}.py"
    path.write_text(as_script(selected), encoding="utf-8")
    return f"{path}:{selected.name}.{selector.rsplit('.', 1)[-1]}"


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


def _memory_lifetimes(source: str, dims: dict[str, int], capsys) -> dict[str, int]:
    args = [f"--dim={name}={extent}" for name, extent in dims.items()]
    assert cli.main(["analyze", source, "--memory", "--json", *args]) == 0
    report = json.loads(capsys.readouterr().out)
    return {
        item["binding"]: item["bytes"]
        for item in report["function_records"]["memory"]["lifetimes"]
    }


@pytest.mark.parametrize(
    "model",
    [model for model in CORPUS if any(case.dims for case in model.sized)],
    ids=_identify,
)
def test_the_cli_analyzes_open_dimensions_at_zero(model, tmp_path, capsys) -> None:
    case = next(case for case in model.sized if case.dims)
    source = _source_for(model, ACCEPTANCE(), tmp_path, case.selector)
    nonzero = dict(case.dims)
    zero = {name: 0 for name in nonzero}
    bindings = _ZERO_SIZED_BINDINGS[model.id]

    zero_lifetimes = _memory_lifetimes(source, zero, capsys)
    nonzero_lifetimes = _memory_lifetimes(source, nonzero, capsys)
    assert bindings <= zero_lifetimes.keys()
    assert all(zero_lifetimes[binding] == 0 for binding in bindings)
    assert all(nonzero_lifetimes[binding] > 0 for binding in bindings)


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
