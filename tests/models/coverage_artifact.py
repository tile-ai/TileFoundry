"""The model coverage report, assembled from what the runner actually reported.

A report of what a compiler can do is worth exactly as much as its weakest link to
reality, and the weakest link available here would be a test that records its own
verdict. A test that recorded `PASS` next to a comparison it never ran would
produce a green matrix, and that is not a hypothetical: it happened in this corpus,
and the fix was to stop trusting a test's account of itself.

So nothing here records an outcome. Each corpus-driven test declares only *which
case it is* -- through `record_property`, which travels on pytest's own report --
and the outcome is read off that report: `passed` is PASS, an expected failure is
BLOCKED, anything else is FAIL. The runner is the one party in the room with no
opinion about whether the compiler works.

Under `-n`, results are spread across worker processes, so each worker writes its
own shard and the controller merges them once every worker has finished. A missing
shard would silently shrink the matrix, so the merge states how many it read.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

#: Where the artifacts are written unless the environment names somewhere else.
#: Under `test_results/`, which is gitignored -- a report is evidence from one run
#: on one machine, and committing it would make it look like a property of the code.
DEFAULT_SUBDIR = Path("test_results")

#: The property a test sets to say which case it is. One name, so a test that
#: misspells it contributes nothing rather than contributing something wrong.
PROPERTY = "corpus_case"

_SHARD_DIR = ".model-coverage-shards"


@dataclass(frozen=True)
class Reported:
    """One case, as the runner reported it."""

    model: str
    target: str
    kind: str
    case: str
    function: str | None
    status: str
    reason: str
    nodeid: str


def declare(
    record_property,
    *,
    model: str,
    target: str,
    kind: str,
    case: str,
    function: str | None = None,
) -> None:
    """Say which case this test is, and nothing about how it went.

    Called before the case runs, so a test that fails halfway still appears in the
    matrix as the case that failed. Declaring afterwards would quietly drop exactly
    the rows a reader is looking for.
    """
    record_property(
        PROPERTY,
        {
            "model": model,
            "target": target,
            "kind": kind,
            "case": case,
            "function": function,
        },
    )


def artifact_dir(root: Path) -> Path:
    """Where to write, from the environment or the repository's own results tree."""
    stated = os.environ.get("TILEFOUNDRY_COVERAGE_DIR")
    return Path(stated) if stated else root / DEFAULT_SUBDIR


def status_of(report) -> tuple[str, str] | None:
    """The case's outcome, read off the runner's report.

    `None` for the setup and teardown phases of a test that got as far as running:
    one case is one verdict, and the call phase is where it is decided. A failure
    *in* setup is a verdict, though -- the case did not run, and reporting nothing
    would drop it from the matrix rather than showing it broken.
    """
    if report.when != "call":
        return ("FAIL", f"{report.when} failed") if report.failed else None
    wasxfail = getattr(report, "wasxfail", None)
    if report.skipped and wasxfail is not None:
        return "BLOCKED", str(wasxfail) or "expected failure"
    if report.passed and wasxfail is not None:
        # A strict xfail that passed is reported as a failure by pytest itself; a
        # non-strict one reaching here is a capability the matrix has gone stale on.
        return "FAIL", "recorded as blocked and it passed"
    if report.passed:
        return "PASS", ""
    if report.skipped:
        return "SKIPPED", str(getattr(report, "longrepr", ("", "", ""))[2] or "skipped")
    return "FAIL", "failed"


def declared(report) -> list[dict]:
    """The cases a test declared itself to be, if any."""
    found = []
    for name, value in getattr(report, "user_properties", ()):
        if name != PROPERTY:
            continue
        found.append(value if isinstance(value, dict) else json.loads(value))
    return found


class ModelCoveragePlugin:
    """Collects declared cases and their reported outcomes, and writes the report."""

    def __init__(self, root: Path, *, delegating: bool = False) -> None:
        self._root = root
        self._records: list[Reported] = []
        self._worker = os.environ.get("PYTEST_XDIST_WORKER")
        #: A controller of workers is sent every report its workers produced, and
        #: the worker that produced it has already put it in a shard. Collecting in
        #: both counts every case twice -- which reads as a matrix twice the size of
        #: the corpus, in a report whose whole job is to be countable.
        self._collecting = self._worker is not None or not delegating

    # -- lifecycle -------------------------------------------------------

    def pytest_sessionstart(self, session) -> None:
        """Discard shards left by an earlier run, before any worker writes one.

        A shard is this run's worker handing its records to this run's controller.
        One left behind -- by a controller killed after a worker had written, or by
        a rerun whose worker allocation differs -- would otherwise be merged in as
        if it belonged here, and then unlinked: the report would carry another
        run's PASS and BLOCKED verdicts, name cases this session never collected,
        and keep no trace that it had happened.

        Only the controller may do this. A worker running it would delete the
        shards its siblings had already written.
        """
        if self._worker is not None:
            return
        shards = artifact_dir(self._root) / _SHARD_DIR
        for path in sorted(shards.glob("*.json")):
            path.unlink()

    # -- collection ------------------------------------------------------

    def pytest_runtest_logreport(self, report) -> None:
        if not self._collecting:
            return
        outcome = status_of(report)
        if outcome is None:
            return
        status, reason = outcome
        for case in declared(report):
            self._records.append(
                Reported(
                    model=case["model"],
                    target=case["target"],
                    kind=case["kind"],
                    case=case["case"],
                    function=case.get("function"),
                    status=status,
                    reason=reason or case.get("reason", ""),
                    nodeid=report.nodeid,
                )
            )

    # -- writing ---------------------------------------------------------

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        directory = artifact_dir(self._root)
        if self._worker is not None:
            self._write_shard(directory)
            return
        shards, records = self._merged(directory)
        self._write(directory, records, shards)

    def _write_shard(self, directory: Path) -> None:
        shards = directory / _SHARD_DIR
        shards.mkdir(parents=True, exist_ok=True)
        path = shards / f"{self._worker}.json"
        path.write_text(
            json.dumps([asdict(record) for record in self._records]), encoding="utf-8"
        )

    def _merged(self, directory: Path) -> tuple[int, list[Reported]]:
        """This process's records plus every worker's, and how many shards there were."""
        records = list(self._records)
        shards = directory / _SHARD_DIR
        count = 0
        for path in sorted(shards.glob("*.json")):
            count += 1
            for item in json.loads(path.read_text(encoding="utf-8")):
                records.append(Reported(**item))
            path.unlink()
        if shards.is_dir() and not any(shards.iterdir()):
            shards.rmdir()
        return count, records

    def _write(self, directory: Path, records: list[Reported], shards: int) -> None:
        from tests.models.registry import CORPUS  # noqa: PLC0415

        directory.mkdir(parents=True, exist_ok=True)
        payload = build(records, CORPUS, shards)
        (directory / "model-coverage.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def build(records: list[Reported], corpus, shards: int) -> dict:
    """Group what ran as `model -> target -> reference / analyze / schedule / sized`.

    `untested` is derived from each model's own Modules rather than from a list, so a
    function added to a model appears as untested instead of vanishing. It is
    computed per target from the cases that actually reported there, which is what
    makes it an answer about this run rather than about the registry.
    """
    from tests.models.report import CaseResult, CoverageCollector, build_report  # noqa: PLC0415

    collector = CoverageCollector()
    for record in records:
        if record.kind not in ("reference", "analyze", "schedule", "sized"):
            continue
        # A reason belongs to an outcome that needs explaining. A PASS does not, and
        # giving it one reads as a caveat on a result that carries none.
        reason = record.reason
        if record.status != "PASS" and not reason.strip():
            reason = "no reason reported"
        collector.results.append(
            CaseResult(
                model=record.model,
                target=record.target,
                kind=record.kind,
                case=record.case,
                function=record.function,
                status=record.status,
                reason="" if record.status == "PASS" else reason,
            )
        )
    report = build_report(collector, corpus)
    # A case reported for a model the corpus does not hold is not nothing: something
    # ran and named it. Dropping it would make the report silently narrower than the
    # run, which is the failure this whole file is arranged against, so it is added
    # with no derived `untested` -- there is no inventory to derive one from.
    for record in records:
        if record.model in report:
            continue
        rows = [
            _row(other)
            for other in records
            if other.model == record.model and other.target == record.target
        ]
        report[record.model] = {
            "inventory": [],
            "targets": {
                record.target: {
                    "reference": [r for r in rows if r["kind"] == "reference"],
                    **{
                        kind: {
                            "tested": [r for r in rows if r["kind"] == kind],
                            "untested": [],
                        }
                        for kind in ("analyze", "schedule", "sized")
                    },
                }
            },
        }
    return {
        "models": report,
        "run": {
            "cases_reported": len(records),
            "worker_shards": shards,
            "models_in_corpus": sorted({case.model for case in corpus}),
            "modules_in_corpus": sorted(case.id for case in corpus),
        },
    }






def _row(record: Reported) -> dict:
    """One reported case as a report row, keeping the kind so it can be filed."""
    row: dict = {"case": record.case, "status": record.status, "kind": record.kind}
    if record.function:
        row["function"] = record.function
    if record.reason and record.status != "PASS":
        row["reason"] = record.reason
    return row





__all__ = [
    "PROPERTY",
    "ModelCoveragePlugin",
    "Reported",
    "artifact_dir",
    "build",
    "declare",
    "declared",
    "status_of",
]
