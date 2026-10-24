"""A CPU provider and CLI reconstruction checks shared by target tests."""

from __future__ import annotations

import pytest

import tilefoundry.cli.target as target_cli
from tilefoundry import cli
from tilefoundry.target import CpuTarget


class ListedCpuTarget(CpuTarget):
    name = "tests.cli.listed_cpu"


def listed_identities(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    builtins = target_cli.available_targets()
    monkeypatch.setattr(
        target_cli,
        "available_targets",
        lambda: (*builtins, ListedCpuTarget()),
    )
    assert cli.main(["target", "list"]) == 0
    listed = capsys.readouterr()
    namespace: dict[str, object] = {}
    for import_line in (line for line in listed.out.splitlines() if line.startswith("from ")):
        exec(import_line, namespace)

    rows = [line.strip() for line in listed.out.splitlines() if "  identity: " in line]
    identities = []
    for row in rows:
        expression, identity = row.rsplit("  identity: ", 1)
        target = eval(expression.rstrip(), namespace)
        assert target.identity == identity
        identities.append(identity)

        assert cli.main(["target", "show", identity]) == 0
        shown = capsys.readouterr()
        assert shown.err == ""
        if "device:" in shown.out:
            assert f"device: {identity}\n" in shown.out
            assert shown.out.count("  digest: ") == 2
        else:
            assert shown.out == (
                f"identity: {identity}\n{expression.rstrip()}\nfacts: unavailable\n"
            )

    assert "tests.cli.listed_cpu" in identities
    return identities
