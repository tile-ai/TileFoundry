from __future__ import annotations

import pytest

from tilefoundry.parser import parse_module_source
from tilefoundry.schedule import (
    UnsupportedDistributionError,
    build_program_schedule_graph,
    generate_distribution_candidates,
)


def _module(body: str) -> str:
    return f'''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Candidates:
    @func
    def main{body}
'''


def test_matmul_candidates_include_m_n_and_partial_k_choices() -> None:
    source = _module('''(x: Tensor[(8, 4), "bf16"], w: Tensor[(4, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
        return tf.matmul(x, w)
''')
    graph = build_program_schedule_graph(parse_module_source(source))
    candidates = generate_distribution_candidates(graph).all_candidates()
    matmul = [candidate for candidate in candidates if candidate.implementation_key == "MatMul"]

    assert any(candidate.output_states[0].layout.split_axis == 0 for candidate in matmul)
    assert any(candidate.output_states[0].layout.split_axis == 1 for candidate in matmul)
    partial = [candidate for candidate in matmul if candidate.output_states[0].partial is not None]
    assert partial
    assert partial[0].input_states[0].layout.split_axis == 1
    assert partial[0].input_states[1].layout.split_axis == 0


def test_exact_divisibility_filters_tail_splits() -> None:
    source = _module('''(x: Tensor[(10, 6), "bf16"]) -> Tensor[(10, 6), "bf16"]:
        return tf.add(x, x)
''')
    graph = build_program_schedule_graph(parse_module_source(source))
    candidates = generate_distribution_candidates(graph).all_candidates()
    counts = {
        candidate.cta_count
        for candidate in candidates
        if candidate.implementation_key == "Binary"
    }
    assert 2 in counts
    assert 4 not in counts
    assert all(candidate.estimated_work.flops >= 0 for candidate in candidates)


def test_reduce_gather_and_topk_rules_preserve_or_create_partial() -> None:
    source = '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Candidates:
    @func
    def main(x: Tensor[(8, 16), "bf16"], indices: Tensor[(4,), "i64"]):
        reduced = tf.reduce(x, axes=(1,), kind="sum")
        gathered = tf.gather(x, indices, axis=0)
        selected = tf.topk(x, k=4, axis=1)
        values = selected[0]
        return (reduced, gathered, values)
'''
    graph = build_program_schedule_graph(parse_module_source(source))
    table = generate_distribution_candidates(graph)
    implementations = {candidate.implementation_key for candidate in table.all_candidates()}
    assert {"Reduce", "Gather", "TopK", "TupleGetItem"} <= implementations
    reduce_candidates = [candidate for candidate in table.all_candidates() if candidate.implementation_key == "Reduce"]
    assert any(candidate.output_states[0].partial is not None for candidate in reduce_candidates)


def test_unknown_operator_fails_closed() -> None:
    source = '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Candidates:
    @func
    def main(x: Tensor[(4,), "bf16"]) -> Tensor[(4,), "bf16"]:
        return tf.rope(x, x, x, x, x)[0]
'''
    graph = build_program_schedule_graph(parse_module_source(source))
    with pytest.raises(UnsupportedDistributionError, match="RoPE"):
        generate_distribution_candidates(graph)
