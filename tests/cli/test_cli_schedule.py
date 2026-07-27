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
