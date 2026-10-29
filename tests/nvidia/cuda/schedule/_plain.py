"""Plain scheduling candidate report expectations."""

from dataclasses import dataclass

from tilefoundry.ir.core import get_registered_name
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary


@dataclass(frozen=True)
class SiteCounts:
    matmuls: int
    reshards: int
    accepted: int = 0
    target: str = "nvidia.h200_sxm"
    detailed: bool = False
    binaries: int | None = None


def check_plain_report(report, source, sites):
    assert report["source"] == source
    assert report["target"] == sites.target
    matmul_rows = [row for row in report["lines"] if row["op"] == "tf.matmul"]
    reshard_rows = [row for row in report["lines"] if row["op"] == "tf.reshard"]
    assert (len(matmul_rows), len(reshard_rows)) == (sites.matmuls, sites.reshards)
    assert sum(bool(row["candidates"]) for row in matmul_rows) == sites.accepted
    assert all(row["candidates"] or row["refused"] for row in report["lines"])
    assert all(
        row["candidates"] or all(item["reason"].startswith("mesh:") for item in row["refused"])
        for row in reshard_rows
    )
    assert not any(row["candidates"] for row in matmul_rows)
    if sites.detailed:
        for row in matmul_rows:
            assert {item["id"] for item in row["refused"]} == {
                "T.cuda.sm90.Wgmma",
                "T.cuda.sm80.Mma",
                "T.cuda.sm89.Mma",
            }
            assert all(item["reason"] for item in row["refused"])
    if sites.binaries is not None:
        binary_names = {get_registered_name(Binary(kind=kind)) for kind in BinaryKind}
        binaries = [row for row in report["lines"] if row["op"] in binary_names]
        assert len(binaries) == sites.binaries
        assert all(any(c["id"] == "T.binary" for c in row["candidates"]) for row in binaries)
        assert all(row["op"] != "tf.schedule" for row in report["lines"])
