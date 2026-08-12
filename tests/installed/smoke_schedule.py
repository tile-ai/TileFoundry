"""`schedule` against a file of one's own."""

from __future__ import annotations

import json

_CTA_SOLVER = ("--first-plan", "--solver-workers=2")


def test_schedule_answers_at_each_level_the_target_schedules(tf, cmine) -> None:
    partition = tf("schedule", f"{cmine}:CMine.root", "--topology", "cta", *_CTA_SOLVER)
    assert partition.returncode == 0, partition.stderr
    assert "nvidia.h200_sxm" in partition.stdout

    pipeline = tf("schedule", f"{cmine}:CMine.root", "--topology", "thread")
    assert pipeline.returncode == 0, pipeline.stderr
    assert "pipeline schedule" in pipeline.stdout


def test_schedule_json_names_the_machine_it_solved_against(tf, cmine) -> None:
    done = tf("schedule", f"{cmine}:CMine.root", "--topology", "thread", "--json")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert payload["target"]["architecture_id"] == "nvidia.sm90"
    assert payload["target"]["device_id"] == "nvidia.h200_sxm"
    scheduled = {item["id"]: item["instruction"] for item in payload["statements"]}
    assert scheduled["MM"] == "SM80_16x8x16_F32BF16BF16F32_TN"


def test_schedule_selects_a_nested_function(tf, cmine) -> None:
    done = tf("schedule", f"{cmine}:CMine.child.inner", "--topology", "thread", "--json")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    payload = json.loads(done.stdout)
    assert payload["target"]["architecture_id"] == "nvidia.sm90"
    assert any(item["id"] == "MM" for item in payload["statements"])


def test_partition_resolves_derived_execution_geometry(tf, derived_prefill) -> None:
    source = f"{derived_prefill}:DerivedPrefill.prefill"
    unbound = tf("schedule", source, "--topology", "cta", *_CTA_SOLVER)
    assert unbound.returncode == 1
    assert "symbolic" in unbound.stderr
    assert "bind every dimension before analysis or scheduling" in unbound.stderr

    bound = tf(
        "schedule",
        source,
        "--topology",
        "cta",
        "--dim",
        "prefill_n=17",
        "--dim",
        "topology_only=32",
        "--json",
        *_CTA_SOLVER,
    )
    assert bound.returncode == 0, bound.stderr
    payload = json.loads(bound.stdout)
    assert payload["extent"] == 3


def test_a_module_without_an_entry_names_its_functions_and_the_rule(tf, tmp_path, cmine) -> None:
    entryless = tmp_path / "entryless.py"
    entryless.write_text(
        cmine.read_text(encoding="utf-8").replace(
            '@module(entry="root", target=', "@module(target="
        ),
        encoding="utf-8",
    )
    done = tf("schedule", f"{entryless}:CMine", "--topology", "cta")
    assert done.returncode == 1
    assert "declares no entry, so it has no default step. It declares root" in done.stderr
    assert "The rule: tilefoundry spec core-ir default-step" in done.stderr


def test_a_level_the_target_does_not_schedule_is_refused(tf, cmine) -> None:
    done = tf("schedule", f"{cmine}:CMine.root", "--topology", "warp")
    assert done.returncode == 1
    assert "warp" in done.stderr


def test_the_solver_flags_are_accepted_by_the_installed_command(tf, cmine) -> None:
    done = tf(
        "schedule",
        f"{cmine}:CMine.root",
        "--topology",
        "cta",
        "--first-plan",
        "--solver-timeout",
        "30",
        "--solver-workers",
        "2",
    )
    assert done.returncode == 0, done.stderr
    assert "nvidia.h200_sxm" in done.stdout
