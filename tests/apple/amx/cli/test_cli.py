"""Tests and helpers for the owning target."""

from __future__ import annotations

import tilefoundry.cli.target as target_cli
from tests.core.cli._targets import ListedCpuTarget
from tilefoundry import cli


def test_target_list_expressions_execute_and_show_accepts_their_identities(
    capsys, monkeypatch
) -> None:
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

    assert {
        "apple.m2_pro",
        "cpu",
    } < set(identities)
    assert "tests.cli.listed_cpu" in identities
