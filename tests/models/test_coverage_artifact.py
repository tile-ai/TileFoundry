"""The coverage report is held to the two ways it can lie without failing.

It can say too much -- count a case twice, or record a verdict a test asserted
about itself -- and it can say nothing at all while every test around it is green.
The second is the dangerous one: an empty report is exactly what a run produces
when the model tests were collected and never executed, and nothing else in the run
looks different.

The outcome mapping is checked against a real runner in a subprocess rather than by
constructing report objects. A hand-made report proves the mapping agrees with what
this file believes pytest does; only pytest can say what pytest does, and the last
time that distinction was skipped here a blocked case was being recorded as a plain
failure for weeks.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.summarise_model_coverage import summarise
from tests.models.coverage_artifact import build

_SESSION = '''
import pytest
from tests.models.coverage_artifact import declare


def test_passes(record_property):
    declare(record_property, model="m", target="t", kind="analyze", case="m/a/one",
            function="one")


@pytest.mark.xfail(strict=True, raises=ValueError, reason="a measured limit")
def test_blocked(record_property):
    declare(record_property, model="m", target="t", kind="analyze", case="m/a/two",
            function="two")
    raise ValueError("a measured limit")


def test_fails(record_property):
    declare(record_property, model="m", target="t", kind="analyze", case="m/a/three",
            function="three")
    assert False, "this one really is broken"
'''


def _run_session(tmp_path: Path, *extra: str) -> list[dict]:
    """Run a throwaway session and hand back the records its plugin collected."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    directory = tmp_path / "artifacts"
    test = tmp_path / "test_declared_cases.py"
    test.write_text(textwrap.dedent(_SESSION), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    conftest = tmp_path / "conftest.py"
    conftest.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(root)!r})


            def pytest_configure(config):
                from tests.models.coverage_artifact import ModelCoveragePlugin
                delegating = bool(getattr(config.option, "numprocesses", None))
                config.pluginmanager.register(
                    ModelCoveragePlugin({str(directory)!r}, delegating=delegating),
                    "coverage-under-test",
                )
            """
        ),
        encoding="utf-8",
    )
    import os  # noqa: PLC0415

    environment = {**os.environ, "TILEFOUNDRY_COVERAGE_DIR": str(directory)}
    # This session must not inherit the outer one's worker identity. A worker writes
    # a shard and leaves the merge to its controller, so an inner session that
    # believed it were one would write no report at all -- and would do so only when
    # this file is run in parallel, which is how it is normally run.
    environment.pop("PYTEST_XDIST_WORKER", None)
    environment.pop("PYTEST_XDIST_WORKER_COUNT", None)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "-p", "no:cacheprovider", "-q", *extra],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    written = directory / "model-coverage.json"
    assert written.is_file(), (
        f"the session wrote no report\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    return payload


def _statuses(payload: dict) -> dict[str, str]:
    rows = payload["models"]["m"]["targets"]["t"]["analyze"]["tested"]
    return {row["case"]: row["status"] for row in rows}


def test_a_shard_left_by_an_earlier_run_is_not_reported_as_this_one(
    tmp_path: Path,
) -> None:
    """The dangerous shape of a stale shard: it names a case this session never
    collected, and merging it leaves no trace.

    A shard is a worker handing its records to its own controller. One survives
    when a controller is killed after a worker has written, or when a rerun
    allocates workers differently -- and the merge then unlinks it, so the report
    carries another run's verdicts and looks legitimate.

    The stale record here is deliberately a case the fresh session does not have,
    because filtering by node id would let exactly that one through.
    """
    directory = tmp_path / "artifacts"
    shards = directory / ".model-coverage-shards"
    shards.mkdir(parents=True, exist_ok=True)
    (shards / "gw0.json").write_text(
        json.dumps(
            [
                {
                    "model": "m",
                    "target": "t",
                    "kind": "analyze",
                    "case": "m/a/from-an-earlier-run",
                    "function": "gone",
                    "status": "PASS",
                    "reason": "",
                    "nodeid": "test_that_no_longer_exists.py::test_gone",
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = _run_session(tmp_path)

    assert "m/a/from-an-earlier-run" not in _statuses(payload)
    assert _statuses(payload) == {
        "m/a/one": "PASS",
        "m/a/two": "BLOCKED",
        "m/a/three": "FAIL",
    }
    assert payload["run"]["cases_reported"] == 3


def test_the_runner_decides_the_outcome_not_the_test(tmp_path: Path) -> None:
    """PASS, BLOCKED and FAIL come from what pytest reported, in one real session.

    The blocked case is a strict xfail that raises: what makes it BLOCKED is the
    runner saying `xfailed`, and nothing in the test says so about itself.
    """
    payload = _run_session(tmp_path)

    assert _statuses(payload) == {
        "m/a/one": "PASS",
        "m/a/two": "BLOCKED",
        "m/a/three": "FAIL",
    }


def test_a_blocked_case_carries_the_reason_the_gate_stated(tmp_path: Path) -> None:
    """A block with no reason cannot be reviewed, so the reason has to travel."""
    payload = _run_session(tmp_path)

    rows = payload["models"]["m"]["targets"]["t"]["analyze"]["tested"]
    blocked = next(row for row in rows if row["status"] == "BLOCKED")
    assert "a measured limit" in blocked["reason"]


def test_a_passing_case_carries_no_reason(tmp_path: Path) -> None:
    """A reason belongs to an outcome that needs explaining; on a PASS it reads as
    a caveat on a result that has none."""
    payload = _run_session(tmp_path)

    rows = payload["models"]["m"]["targets"]["t"]["analyze"]["tested"]
    passing = next(row for row in rows if row["status"] == "PASS")
    assert not passing.get("reason")


def test_running_on_several_workers_counts_each_case_once(tmp_path: Path) -> None:
    """The controller is sent every report its workers produced, and the worker that
    produced it has already recorded it. Counting in both doubles the matrix.

    Compared against the same session run in one process, so the assertion is that
    the two agree rather than that either matches a number written here.

    `xdist` is declared in the `test` extra because both CI and the local runs are
    parallel, so this normally executes; the guard is for an environment that
    installed pytest alone, where there is no controller for the double count to
    occur in.
    """
    pytest.importorskip("xdist", reason="the double count needs a controller to occur")

    serial = _run_session(tmp_path / "serial")
    parallel = _run_session(tmp_path / "parallel", "-n", "2")

    assert parallel["run"]["cases_reported"] == serial["run"]["cases_reported"]
    assert _statuses(parallel) == _statuses(serial)


def test_an_empty_report_is_a_failure() -> None:
    """The report a run produces when the model tests never executed.

    Everything else about such a run looks like success, which is why this is the
    one condition the summariser exits non-zero on regardless of content.
    """
    from scripts.summarise_model_coverage import main  # noqa: PLC0415

    empty = {
        "models": {},
        "run": {
            "cases_reported": 0,
            "worker_shards": 0,
            "models_in_corpus": [],
            "modules_in_corpus": [],
        },
    }
    path = Path(__file__).parent / "_empty-coverage.json"
    path.write_text(json.dumps(empty), encoding="utf-8")
    try:
        assert main(["summarise", str(path)]) == 1
    finally:
        path.unlink()


def test_a_reported_failure_is_counted_as_one() -> None:
    """A FAIL in the report is a FAIL of the run that produced it."""
    payload = {
        "models": {
            "m": {
                "inventory": ["one"],
                "targets": {
                    "t": {
                        "reference": [
                            {"case": "m/r", "status": "FAIL", "reason": "broken"}
                        ],
                        "analyze": {"tested": [], "untested": []},
                        "schedule": {"tested": [], "untested": []},
                        "sized": {"tested": [], "untested": []},
                    }
                },
            }
        },
        "run": {
            "cases_reported": 1,
            "worker_shards": 0,
            "models_in_corpus": ["m"],
            "modules_in_corpus": ["m"],
        },
    }

    text, failures = summarise(payload)

    assert failures == 1
    assert "FAIL" in text



def test_the_report_states_what_the_run_covered() -> None:
    """The counts a reader checks first are in the payload, so the summariser and
    any other consumer read them from one place."""
    from tests.models.registry import CORPUS  # noqa: PLC0415

    run = build([], CORPUS, 0)["run"]

    assert run["models_in_corpus"] == sorted({case.model for case in CORPUS})
    assert len(run["modules_in_corpus"]) == len(CORPUS)
    assert run["cases_reported"] == 0


@pytest.mark.parametrize("kind", ["analyze", "schedule", "sized"])
def test_the_report_separates_untested_from_not_asked(kind: str) -> None:
    """`untested` is a gap; `sized`'s `not_applicable` is not one.

    A function with no dimension left open has no context length to be asked about,
    and listing it as untested would report a capability nobody is missing.
    """
    from tests.models.registry import CORPUS  # noqa: PLC0415
    from tests.models.report import CoverageCollector, build_report  # noqa: PLC0415

    collector = CoverageCollector()
    collector.record(
        model=CORPUS[0].model,
        target="t",
        kind=kind,
        case="c",
        status="PASS",
        function=None,
    )

    section = build_report(collector, CORPUS)[CORPUS[0].model]["targets"]["t"][kind]

    assert "untested" in section
    assert ("not_applicable" in section) == (kind == "sized")
