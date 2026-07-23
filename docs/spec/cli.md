# TileFoundry Command-Line Interface

This file is the normative reference printed by `tilefoundry help cli`. It
defines the command-line contract for Agent-authored HIR analysis. `help dsl`
prints the installed [HIR specification](./hir.md); Python authoring syntax and
grammar productions remain in the [parser specification](./parser.md).

## Commands

```text
tilefoundry analyze model.py[:Module[.function]]
    [--roofline] [--footprint] [--timeline]

tilefoundry inspect capabilities model.py[:Module[.function]]

tilefoundry help dsl
tilefoundry help cli
```

`SOURCE` is a Python file followed optionally by `:Module`, `:Function`, or
`:Module.function`. Without a selector, the source must define exactly one HIR
Module or exactly one HIR Function. A selector chooses the named Module, the
named Function, or a named Function inside a Module.

## Analyze

`analyze` first runs deterministic type inference and then prints complete type
comments, regardless of analysis flags. It never performs candidate search,
layout enumeration, or automatic resharding.

With no analysis flag, `analyze` runs roofline, footprint, and timeline. When
one or more flags are present, it runs only the named analyses. The selected
Function target, or the selected Module entry Function target, determines the
hardware specification; there is no ordinary `--target` option.

On success, stdout begins with the overall analysis summary followed by
annotated HIR. On inference, verification, or analysis failure, stdout is empty
and stderr reports the source location, binding where available, and reason.

## Inspect Capabilities

`inspect capabilities` resolves the target from the selected Function or Module
entry Function and prints the installed compact hardware capability record. It
does not emit compiler operation coverage. Hardware facts identify their unit,
qualification, source, and whether they are direct, derived, runtime queried,
or unavailable.

## Help

`help dsl` writes `share/tilefoundry/spec/hir.md` verbatim; `help cli` writes
`share/tilefoundry/spec/cli.md`. In a source or editable tree, they read the
matching files from `docs/spec/`. Python operation signatures are provided
separately by installed stubs and Python introspection.
