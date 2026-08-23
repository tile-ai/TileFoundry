"""Parser-owned checks for canonical grid-loop bindings."""

from tests.fixtures.placed.gqa_decode import GqaOnline
from tilefoundry.inspection import as_script
from tilefoundry.ir.hir.specialize import specialize_concretely


def test_gqa_correction_reads_the_old_carry_and_unique_yield() -> None:
    function = specialize_concretely(GqaOnline.entry_function(), {"ctx_len": 8})
    printed = as_script(function)

    assert printed.count("        m_new = max(m, score)") == 1
    assert " = sub(m, m_new)" in printed
    lines = printed.splitlines()
    start = lines.index("    for i in range(8):")
    end = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].startswith("    ")
        and not lines[index].startswith("        ")
    )
    yielded = lines[end - 3 : end]
    assert [line.split(" = ", 1)[0] for line in yielded] == [
        "        l",
        "        o",
        "        m",
    ]
    assert yielded[-1] == "        m = m_new"
    assert len({line.split(" = ", 1)[1] for line in yielded}) == 3
    assert lines[end] == '    k_n = cast(k_new, dtype="f32")'
