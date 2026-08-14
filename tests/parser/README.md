# tests/parser

Three shapes, and the rule for picking one.

## A new parser feature goes into an existing program

`programs.py` holds five programs, each parsed once at import. Find the one
whose `features` list names the area you are touching — `HirExpressions`,
`HirGrid`, `HirSharded`, `HirModule`, or the `tir_*` prim funcs — and add a
line to its body plus an entry to `features`. Then regenerate the goldens:

```
pytest tests/parser --update-golden
```

`golden/<program>.py` is the program printed back as DSL source by
`tilefoundry.inspection.as_script`. Read the diff before keeping it: that diff
*is* the assertion, so an unexplained change there is the finding. The goldens
are excluded from ruff, because reformatting recorded printer output edits the
evidence rather than the code.

There is no golden for the TIR programs — nothing prints a `PrimFunction` — so
`test_tir_programs.py` asserts on the parsed nodes directly.

## A new refusal goes into the table

`error_cases.py` holds every subject `tests/parser` refuses, as one row each:
the subject, the exception, and the message it must carry. `subject` is DSL
source to import, or a builder for the ones with no source to feed — a
definition that must fail while it is decorated, a hand-forged TIR node, an
operand check that never reaches the parser. `test_refused_programs.py` runs
the table and is the only entry point; nothing else here carries a bare
`pytest.raises`.

## Only what neither can express gets its own test

`test_programs.py` and `test_tir_programs.py` carry the rest, each saying in
its docstring why a golden cannot hold it. In practice that is node identity,
a target's registered `Op` class, a canonicalisation the printer renders back
as the sugar it was written as, and the things that were never one program
parsing: evaluating against torch, printer equality, re-elaboration, and
reading a generated `.pyi`.

A new file is the last resort, not the first. Adding one is a claim that the
subject fits none of the three shapes above.

## Measuring

The gauge is a whole round of `tests/parser`, not per-test contexts — module-level
programs are parsed at import, and that coverage belongs to no test's context.

```
COVERAGE_FILE=<tmp>/parser.coverage python -m pytest tests/parser -q \
  -p no:randomly --cov=tilefoundry --cov-branch --cov-report=
COVERAGE_FILE=<tmp>/parser.coverage python -m coverage report \
  --include='*/tilefoundry/parser/*'
COVERAGE_FILE=<tmp>/parser.coverage python -m coverage report
```

Read both totals. Plenty of what these tests pin lives outside the parser
package — `dsl/`, `evaluator/`, `ir/tir/`, `codegen/` — and the parser-only
number is blind to all of it.
