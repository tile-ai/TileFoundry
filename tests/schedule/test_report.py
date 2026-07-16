from __future__ import annotations

from tilefoundry.ir.target import CudaTarget
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.parser import parse_module_source
from tilefoundry.schedule import auto_dist

SOURCE = '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Reported:
    @func
    def branch(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.add(x, x)

    @func
    def main(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        value: where(layout=(H @ cta,)) = branch(x)
        return value
'''


def test_report_and_markdown_are_authoritative_and_deterministic() -> None:
    module = parse_module_source(SOURCE)
    mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    result = auto_dist(module, target=CudaTarget(device="h200_sxm"), mesh=mesh)
    report = result.report

    assert report.status == "OPTIMAL"
    assert report.logical_fingerprint
    assert report.level == "cta"
    assert report.predicted_makespan_ns >= 0
    assert report.function_regions
    assert report.operations
    assert len(report.constraints) == 1
    assert report.constraints[0].satisfied is True
    assert report.render_markdown() == report.render_markdown()
    assert "uncalibrated formula" in report.render_markdown()
    assert "## Operations" in report.render_markdown()
