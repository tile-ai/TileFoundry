#!/usr/bin/env python3
"""Generate the pinned FlashInfer kernel-source survey."""

from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import re
import subprocess
from pathlib import Path, PurePosixPath

FAMILIES = (
    "norm",
    "attention",
    "moe",
    "quant",
    "gemm",
    "comm",
    "recurrent",
    "sampling",
    "unclassified",
)
EXPECTED_COVERED_STRATEGIES = {
    "A03",
    "A07",
    "G01",
    "N01",
    "N03",
    "N05",
    "N06",
    "N07",
    "Q01",
}
SUPPORT_ROW = re.compile(r"^\|\s*([A-Z]\d{2})\s*\|\s*([^|]+?)\s*\|")
CATALOG_ROW = re.compile(r"^\|\s*([A-Z]\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
STRATEGY_ID = re.compile(r"\b[A-Z]\d{2}\b")
REVISION = re.compile(r"\b[0-9a-f]{40}\b")
SOURCE_REFERENCE = re.compile(
    r"(?P<path>(?:csrc|include/flashinfer|flashinfer|benchmarks|tests)/[^\s`:,+]+\.(?:cuh|cu|py|rst))"
    r"(?::(?P<symbol>[A-Za-z_][A-Za-z0-9_.]*))?"
)
GLOBAL_MARKER = re.compile(r"(?:__global__|FLASHINFER_GLOBAL)")
CALL_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ENTRY_TOKEN = re.compile(
    r"(kernel|fused|fusion|attention|prefill|decode|cascade|merge_state|norm|quant|rope|page|"
    r"append|moe|expert|routing|topk|top_k|sampling|sample|gemm|matmul|mm_|_mm|allreduce|"
    r"all_reduce|alltoall|all_to_all|all_gather|kda|gdn|delta_rule|mamba|ssd|state_update|"
    r"mhc|silu_and_mul|packbits)",
    re.IGNORECASE,
)
FUSION_TOKEN = re.compile(
    r"(fused|fusion|and_mul|add_rmsnorm|rmsnorm_silu|quantize_append|all_gather_matmul|"
    r"allreduce_fusion|attention|moe|kda|gdn|mamba|mhc|ssd|sampling)",
    re.IGNORECASE,
)
NON_ENTRY_PREFIXES = (
    "check_",
    "compile_",
    "create_",
    "gen_",
    "get_",
    "has_",
    "is_",
    "make_",
    "set_",
    "validate_",
)
NON_ENTRY_CLASS_SUFFIXES = ("Config", "Info", "Role", "Type", "Workspace")
PUBLIC_ENTRY_DECORATORS = {"custom_op", "flashinfer_api"}


@dataclasses.dataclass(frozen=True)
class CatalogEntry:
    strategy: str
    boundary: str
    paths: tuple[str, ...]
    symbols: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SurveyRow:
    path: str
    family: str
    entries: tuple[str, ...]
    fusion: bool
    strategies: tuple[str, ...]
    fixtures: tuple[str, ...]


def _run_git(upstream: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(upstream), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or "git command failed")
    return completed.stdout


def _tracked_files(upstream: Path) -> tuple[str, ...]:
    return tuple(path for path in _run_git(upstream, "ls-files").splitlines() if path)


def _in_source_scope(path: str) -> bool:
    item = PurePosixPath(path)
    if path.startswith("csrc/"):
        return item.suffix in {".cu", ".cuh"}
    if path.startswith("include/flashinfer/"):
        return item.suffix == ".cuh"
    return path.startswith("flashinfer/") and item.suffix == ".py"


def _parse_support_matrix(path: Path) -> dict[str, str]:
    strategies: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SUPPORT_ROW.match(line)
        if match:
            strategies[match.group(1)] = match.group(2).strip()
    if len(strategies) != 60:
        raise SystemExit(f"expected 60 SUPPORT-MATRIX strategies, found {len(strategies)}")
    return strategies


def _parse_catalog(path: Path, support: dict[str, str]) -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    previous_paths: tuple[str, ...] = ()
    previous_symbols: tuple[str, ...] = ()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CATALOG_ROW.match(line)
        if not match or match.group(1) not in support:
            continue
        strategy, provenance = match.group(1), match.group(3)
        references = list(SOURCE_REFERENCE.finditer(provenance))
        paths = tuple(dict.fromkeys(item.group("path") for item in references))
        symbols = tuple(
            dict.fromkeys(item.group("symbol") for item in references if item.group("symbol"))
        )
        if not paths and re.search(r"\bsame (?:symbol|template|API)\b", provenance, re.IGNORECASE):
            paths, symbols = previous_paths, previous_symbols
        if paths:
            previous_paths, previous_symbols = paths, symbols
        entries.append(CatalogEntry(strategy, support[strategy], paths, symbols))
    found = {entry.strategy for entry in entries}
    if found != set(support):
        missing = ", ".join(sorted(set(support) - found))
        raise SystemExit(f"catalog is missing SUPPORT-MATRIX strategies: {missing}")
    return tuple(entries)


def _fixture_coverage(fixtures: Path) -> dict[str, tuple[str, ...]]:
    coverage: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted(fixtures.glob("*.py")):
        if path.name == "__init__.py" or path.name.endswith(".blocked.py"):
            continue
        for strategy in sorted(set(STRATEGY_ID.findall(path.read_text(encoding="utf-8")))):
            coverage[strategy].append(path.name)
    found = set(coverage)
    if found != EXPECTED_COVERED_STRATEGIES:
        missing = ", ".join(sorted(EXPECTED_COVERED_STRATEGIES - found)) or "none"
        extra = ", ".join(sorted(found - EXPECTED_COVERED_STRATEGIES)) or "none"
        raise SystemExit(f"covered strategy drift: missing [{missing}], extra [{extra}]")
    return {strategy: tuple(paths) for strategy, paths in coverage.items()}


def _catalog_indexes(
    catalog: tuple[CatalogEntry, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    strategies: dict[str, list[str]] = collections.defaultdict(list)
    symbols: dict[str, list[str]] = collections.defaultdict(list)
    for entry in catalog:
        for path in entry.paths:
            strategies[path].append(entry.strategy)
            symbols[path].extend(entry.symbols)
    return (
        {path: tuple(dict.fromkeys(values)) for path, values in strategies.items()},
        {path: tuple(dict.fromkeys(values)) for path, values in symbols.items()},
    )


def _family_from_strategy(strategy: str) -> str:
    return {
        "N": "norm",
        "A": "attention",
        "R": "attention",
        "M": "moe",
        "Q": "quant",
        "G": "gemm",
        "C": "comm",
        "K": "recurrent",
        "S": "recurrent",
        "H": "recurrent",
        "T": "sampling",
        "L": "sampling",
    }[strategy[0]]


def _classify_family(path: str, strategies: tuple[str, ...]) -> str | None:
    if strategies:
        families = {_family_from_strategy(strategy) for strategy in strategies}
        if len(families) == 1:
            return families.pop()
    value = path.lower()
    rooted_rules = (
        ("comm", ("/comm/", "include/flashinfer/comm/")),
        ("moe", ("/fused_moe/", "/moe_ep/", "/grouped_mm/")),
        ("recurrent", ("/kda/", "/kda_", "/gdn_", "/gdn/", "/mamba/", "/mhc")),
        ("attention", ("/attention/", "/mla/", "/xqa/")),
        ("norm", ("/norm/",)),
        ("gemm", ("/gemm/", "/deep_gemm/")),
        ("quant", ("/quantization/",)),
        ("sampling", ("/sampling", "/topk", "/logits_processor/")),
    )
    for family, needles in rooted_rules:
        if any(needle in value for needle in needles):
            return family
    rules = (
        ("comm", ("/comm/", "allreduce", "all_reduce", "alltoall", "all_to_all", "nvshmem")),
        ("moe", ("moe", "expert", "routing", "router", "grouped_mm")),
        ("recurrent", ("kda", "gdn", "delta_rule", "mamba", "ssd", "mhc", "state_update")),
        ("attention", ("attention", "/mla", "decode", "prefill", "page", "rope", "xqa", "pod")),
        ("norm", ("norm", "layernorm", "rmsnorm")),
        ("quant", ("quant", "fp4", "fp8", "mxfp", "activation", "packbits")),
        ("gemm", ("gemm", "matmul", "tinygemm", "bgmv", "/bmm")),
        ("sampling", ("sampling", "topk", "top_k", "top_p", "logits", "air_top_p")),
    )
    for family, needles in rules:
        if any(needle in value for needle in needles):
            return family
    return None


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _python_entries(
    path: Path,
    catalog_symbols: tuple[str, ...],
    *,
    explicit_only: bool = False,
) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return catalog_symbols
    entries: list[str] = list(catalog_symbols)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        explicitly_public = any(
            _decorator_name(decorator) in PUBLIC_ENTRY_DECORATORS
            for decorator in node.decorator_list
        )
        if explicitly_public:
            entries.append(node.name)
            continue
        if explicit_only:
            continue
        if node.name.startswith("_"):
            continue
        if node.name.startswith(NON_ENTRY_PREFIXES):
            continue
        if isinstance(node, ast.ClassDef) and node.name.endswith(NON_ENTRY_CLASS_SUFFIXES):
            continue
        if ENTRY_TOKEN.search(node.name):
            entries.append(node.name)
    return tuple(dict.fromkeys(entries))


def _remove_balanced_call(value: str, name: str) -> str:
    while True:
        start = value.find(name)
        if start < 0:
            return value
        opening = value.find("(", start + len(name))
        if opening < 0:
            return value
        depth = 0
        closing = -1
        for index in range(opening, len(value)):
            if value[index] == "(":
                depth += 1
            elif value[index] == ")":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing < 0:
            return value
        value = value[:start] + value[closing + 1 :]


def _cpp_entries(path: Path, catalog_symbols: tuple[str, ...]) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8", errors="replace")
    entries: list[str] = list(catalog_symbols)
    for marker in GLOBAL_MARKER.finditer(text):
        line_start = text.rfind("\n", 0, marker.start()) + 1
        if text[line_start : marker.start()].lstrip().startswith("#define"):
            continue
        tail = text[marker.end() : marker.end() + 1600]
        brace = tail.find("{")
        semicolon = tail.find(";")
        ends = tuple(index for index in (brace, semicolon) if index >= 0)
        if not ends:
            continue
        end = min(ends)
        header = "\n".join(
            line for line in tail[:end].splitlines() if not line.lstrip().startswith("#")
        )
        for attribute in ("__launch_bounds__", "__cluster_dims__", "__maxnreg__"):
            header = _remove_balanced_call(header, attribute)
        functions = re.findall(r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", header)
        if functions:
            entries.append(functions[0])
            continue
        candidates = CALL_NAME.findall(header)
        if candidates:
            entries.append(candidates[0])
    return tuple(dict.fromkeys(entries))


def _hard_exclusion(path: str) -> tuple[str, str] | None:
    if path.startswith("csrc/nv_internal/"):
        return (
            "vendored third-party",
            "vendored TensorRT-LLM/deep-gemm implementation is not FlashInfer's own kernel surface",
        )
    parts = PurePosixPath(path).parts
    excluded_segments = {
        "jit": (
            "JIT/build plumbing",
            "generates or loads kernels; it is not a semantic kernel boundary",
        ),
        "testing": (
            "test support",
            "test-only generators and reference helpers are not shipped kernels",
        ),
        "autotuner": (
            "tuning/config",
            "search and configuration code does not define kernel semantics",
        ),
        "profiler": (
            "observability",
            "profiling and tracing code records execution rather than computing it",
        ),
        "trace": (
            "observability",
            "profiling and tracing code records execution rather than computing it",
        ),
        "trace_apply": (
            "observability",
            "profiling and tracing code records execution rather than computing it",
        ),
        "tuning_configs": (
            "tuning/config",
            "search and configuration code does not define kernel semantics",
        ),
    }
    for part in parts:
        if part in excluded_segments:
            return excluded_segments[part]
    stem = PurePosixPath(path).stem.lower()
    if stem in {"testing", "test_utils"}:
        return (
            "test support",
            "test-only generators and reference helpers are not shipped kernels",
        )
    if any(token in stem for token in ("benchmark", "reference", "validation")):
        return (
            "benchmark/reference",
            "benchmark and reference implementations are evidence, not kernel APIs",
        )
    return None


def _helper_exclusion(path: str) -> tuple[str, str] | None:
    parts = PurePosixPath(path).parts
    if any(part in {"helpers", "roles"} for part in parts):
        return (
            "helper/utility",
            "shared primitives and configuration are not complete kernel boundaries",
        )
    stem = PurePosixPath(path).stem.lower()
    helper_tokens = (
        "backend",
        "compiler",
        "epilogue",
        "helper",
        "mainloop",
        "registry",
        "runner",
        "scheduler",
        "staging",
        "utils",
    )
    if any(token in stem for token in helper_tokens):
        return (
            "helper/utility",
            "shared primitives and configuration are not complete kernel boundaries",
        )
    if stem in {
        "algo_knobs",
        "api",
        "base",
        "collective_builder",
        "common",
        "compile",
        "config",
        "configs",
        "enums",
        "errors",
        "fusion_rules",
        "layer",
        "mainloop_spec",
        "pipeline_topology",
        "prepare",
        "runner_common",
        "runners",
        "schedule",
        "tensors",
        "tuner",
        "utils",
        "helpers",
        "weights",
        "workspace_base",
    }:
        return (
            "helper/utility",
            "shared primitives and configuration are not complete kernel boundaries",
        )
    if stem in {"__main__", "aot"} or stem.endswith("_enums"):
        return (
            "helper/utility",
            "shared primitives and configuration are not complete kernel boundaries",
        )
    return None


def _is_fusion(
    path: str,
    family: str,
    strategies: tuple[str, ...],
    entries: tuple[str, ...],
) -> bool:
    if strategies:
        return True
    if family in {"attention", "moe", "recurrent"}:
        return True
    return bool(FUSION_TOKEN.search(" ".join((path, *entries))))


def _scan(
    upstream: Path,
    tracked: tuple[str, ...],
    catalog_strategies: dict[str, tuple[str, ...]],
    catalog_symbols: dict[str, tuple[str, ...]],
    fixture_coverage: dict[str, tuple[str, ...]],
) -> tuple[tuple[SurveyRow, ...], dict[tuple[str, str], list[str]]]:
    rows: list[SurveyRow] = []
    excluded: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for relative in tracked:
        if not _in_source_scope(relative):
            continue
        strategies = catalog_strategies.get(relative, ())
        hard_exclusion = _hard_exclusion(relative)
        if hard_exclusion:
            excluded[hard_exclusion].append(relative)
            continue
        path = upstream / relative
        symbols = catalog_symbols.get(relative, ())
        helper_exclusion = _helper_exclusion(relative)
        if path.suffix == ".py":
            entries = _python_entries(
                path,
                symbols,
                explicit_only=helper_exclusion is not None,
            )
        else:
            entries = _cpp_entries(path, symbols)
        if not entries:
            if path.suffix in {".cu", ".cuh"} and "__global__" in path.read_text(
                encoding="utf-8", errors="replace"
            ):
                excluded[
                    (
                        "CUDA marker without entry",
                        "contains __global__ only in a macro or unparseable declaration, not a kernel entry",
                    )
                ].append(relative)
                continue
            if helper_exclusion:
                excluded[helper_exclusion].append(relative)
                continue
            excluded[
                ("no kernel entry", "no CUDA global definition or public Python kernel/API entry")
            ].append(relative)
            continue
        family = _classify_family(relative, strategies) or "unclassified"
        fixtures = tuple(
            sorted(
                {
                    fixture
                    for strategy in strategies
                    for fixture in fixture_coverage.get(strategy, ())
                }
            )
        )
        rows.append(
            SurveyRow(
                path=relative,
                family=family,
                entries=entries,
                fusion=_is_fusion(relative, family, strategies, entries),
                strategies=strategies,
                fixtures=fixtures,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.path)), excluded


def _cell(values: tuple[str, ...], empty: str = "-") -> str:
    if not values:
        return empty
    return "<br>".join(f"`{value.replace('|', '&#124;')}`" for value in values)


def _render(
    upstream: Path,
    revision: str,
    tracked: tuple[str, ...],
    rows: tuple[SurveyRow, ...],
    excluded: dict[tuple[str, str], list[str]],
    catalog: tuple[CatalogEntry, ...],
    coverage: dict[str, tuple[str, ...]],
) -> str:
    scoped = tuple(path for path in tracked if _in_source_scope(path))
    outside = len(tracked) - len(scoped)
    covered_rows = tuple(row for row in rows if row.fixtures)
    uncovered = tuple(row for row in rows if not row.fixtures)
    family_counts = collections.Counter(row.family for row in uncovered)
    root_counts = collections.Counter(
        PurePosixPath(path).parts[0] for path in tracked if path not in scoped
    )
    global_marker_paths = {
        path
        for path in scoped
        if PurePosixPath(path).suffix in {".cu", ".cuh"}
        and "__global__" in (upstream / path).read_text(encoding="utf-8", errors="replace")
    }
    included_paths = {row.path for row in rows}
    global_exclusions = {
        reason: len(global_marker_paths.intersection(paths))
        for (reason, _), paths in excluded.items()
        if global_marker_paths.intersection(paths)
    }
    global_accounting = [
        f"{len(global_marker_paths.intersection(included_paths))} included",
        *(f"{count} {reason}" for reason, count in sorted(global_exclusions.items())),
    ]
    lines = [
        "# FlashInfer transcribable-kernel survey",
        "",
        f"Pinned upstream: `flashinfer-ai/flashinfer@{revision}`.",
        "",
        "This file is generated. Reproduce it from the TileFoundry worktree with:",
        "",
        "```bash",
        "PLAN_DIR=/path/to/docs/plans/hir-frontend",
        "python scripts/survey_flashinfer.py --upstream ~/flashinfer \\",
        '  --evidence-dir "$PLAN_DIR/evidence/flashinfer" \\',
        '  --fixtures tests/fixtures/flashinfer --output "$PLAN_DIR/SURVEY.md"',
        "```",
        "",
        "## Result",
        "",
        f"- Repository accounting: **{len(tracked)} tracked files = {len(scoped)} scoped source files + {outside} outside the source roots**.",
        f"- Scoped accounting: **{len(scoped)} = {len(rows)} transcribable groups + {sum(len(paths) for paths in excluded.values())} excluded source files**.",
        f"- D28 output surface: **{len(rows)} `.py` groups**; **{len(covered_rows)} covered**, **{len(uncovered)} uncovered**.",
        f"- Existing semantic coverage: **{len(coverage)} strategies** across **{len(covered_rows)} upstream source files**. Strategy count and D28 file count are intentionally different.",
        f"- CUDA marker audit: **{len(global_marker_paths)} files containing `__global__` = "
        + " + ".join(global_accounting)
        + "**.",
        "",
        "### Uncovered groups by mechanism family",
        "",
        "| Family | Files |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {family} | {family_counts[family]} |" for family in FAMILIES)
    lines.extend(
        [
            f"| **Total** | **{len(uncovered)}** |",
            "",
            "## Method",
            "",
            "The scan uses every Git-tracked file at the pinned revision. The source roots are",
            "`csrc/**/*.{cu,cuh}`, `include/flashinfer/**/*.cuh`, and `flashinfer/**/*.py`.",
            "After the explicit exclusions below, a C++ file enters the D28 surface when it",
            "declares or defines a CUDA global kernel, or when the 60-strategy catalog names it as",
            "a semantic boundary. A Python file enters when it defines a public kernel/API entry",
            "or is catalog provenance. Family assignment happens only after admission; a file that",
            "does not fit the eight D29 work-splitting labels is retained as `unclassified`.",
            "",
            "The Python test is static and deterministic: in an operation module, a top-level",
            "public Function or Class name must contain a kernel/operation term such as attention,",
            "norm, quant, MoE, GEMM, collective, recurrent, or sampling. A helper/utility module",
            "enters only when the catalog names it or a top-level entry is explicitly exported by",
            "`@flashinfer_api` or `@custom_op`; internal `@cute.jit` primitives do not make the",
            "whole helper file a D28 kernel boundary. Factory/support names and test/JIT paths are",
            "excluded.",
            "",
            "Catalog symbols are retained even when their implementation is a C++ template method",
            "rather than a CUDA global. C++ declarations and definitions both create rows; bare",
            "`#define ... __global__` aliases are accounted separately and do not invent entries.",
            "",
            "`fusion=yes` means either the support catalog names the file as a fusion boundary or",
            "the file/entry names identify a multi-operation attention, MoE, recurrent, sampling,",
            "or explicitly fused implementation. It is a survey classification, not a claim that",
            "TileFoundry already expresses that boundary.",
            "",
            "## Existing coverage",
            "",
            "| Strategy | Boundary | Upstream source | Fixture |",
            "| --- | --- | --- | --- |",
        ]
    )
    catalog_by_strategy = {entry.strategy: entry for entry in catalog}
    for strategy in sorted(coverage):
        entry = catalog_by_strategy[strategy]
        source = _cell(tuple(path for path in entry.paths if _in_source_scope(path)))
        lines.append(f"| {strategy} | {entry.boundary} | {source} | {_cell(coverage[strategy])} |")
    lines.extend(
        [
            "",
            "## D28 source groups",
            "",
            "One row is one prospective corpus `.py`; one row may contain multiple independent",
            "authored Modules when the upstream file exposes multiple kernel entries.",
            "",
            "| Upstream source | Family | Public/kernel entries | Fusion | Strategies | Coverage |",
            "| --- | --- | --- | :---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.path}` | {row.family} | {_cell(row.entries)} | "
            f"{'yes' if row.fusion else 'no'} | {_cell(row.strategies)} | {_cell(row.fixtures)} |"
        )
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            "Every scoped source file not listed above is assigned exactly one reason. Family",
            "classification is never an exclusion. `helper/utility` is considered only after no",
            "CUDA or public Python kernel entry was found:",
            "",
            "| Reason | Files | Why |",
            "| --- | ---: | --- |",
        ]
    )
    for (reason, why), paths in sorted(excluded.items()):
        lines.append(f"| {reason} | {len(paths)} | {why} |")
    lines.extend(
        [
            "",
            "Tracked files outside the three source roots were still counted. They are excluded",
            "because docs, tests, benchmarks, CI, examples, packaging, and vendored dependencies",
            "are not upstream kernel-source grouping units:",
            "",
            "| Top-level path | Files |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{root + '/' if '.' not in root else root}` | {count} |"
        for root, count in sorted(root_counts.items())
    )
    lines.append("")
    for (reason, why), paths in sorted(excluded.items()):
        lines.extend(
            [
                f"<details><summary>{reason}: {len(paths)} files</summary>",
                "",
                f"Reason: {why}.",
                "",
                "```text",
                *sorted(paths),
                "```",
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    upstream = args.upstream.resolve()
    evidence = args.evidence_dir.resolve()
    revision = _run_git(upstream, "rev-parse", "HEAD").strip()
    dirty = _run_git(upstream, "status", "--porcelain")
    if dirty:
        raise SystemExit("upstream worktree must be clean")
    catalog_path = evidence / "CATALOG.md"
    expected_revisions = set(REVISION.findall(catalog_path.read_text(encoding="utf-8")))
    if revision not in expected_revisions:
        raise SystemExit(f"upstream revision {revision} is not pinned by {catalog_path}")
    support = _parse_support_matrix(evidence / "SUPPORT-MATRIX.md")
    catalog = _parse_catalog(catalog_path, support)
    coverage = _fixture_coverage(args.fixtures.resolve())
    catalog_strategies, catalog_symbols = _catalog_indexes(catalog)
    tracked = _tracked_files(upstream)
    rows, excluded = _scan(
        upstream,
        tracked,
        catalog_strategies,
        catalog_symbols,
        coverage,
    )
    known_catalog_paths = {
        path for entry in catalog for path in entry.paths if _in_source_scope(path)
    }
    surveyed_paths = {row.path for row in rows}
    missing_catalog_paths = known_catalog_paths - surveyed_paths
    if missing_catalog_paths:
        raise SystemExit(
            "catalog source paths missing from survey: " + ", ".join(sorted(missing_catalog_paths))
        )
    args.output.write_text(
        _render(upstream, revision, tracked, rows, excluded, catalog, coverage),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
