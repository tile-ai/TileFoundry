# TileFoundry in two steps

TileFoundry is source to source: the reference is source code, the fast
implementation is source code, and either can be pointed at any command.
`check` says whether two of them agree; `analyze` says what one costs.

```text
  step one — describe it, until it agrees

      published model ─────► authored HIR ─────► check ─────┐
                             the reference          │       │
                                                 agrees    fix ─┘

  step two — make it fast; both roads lead back to the same source

        ┌────── change the HIR ◄────── not yet ──────┐
        ▼                                            │
   authored HIR ─────► analyze ─────► predicted performance ok? ──┘
        ▲                                     │
        │                                    yes
        │                                     ▼
        │          write a runtime twin ─────► check ─────► measure
        │                                                     │
        │                                       measured performance ok?
        │                                             ┌───────┴───────┐
        └─────────────────── no ◄─────────────────────┘               └──► ship
```

## Where the other answers are

- `tilefoundry spec <topic>` — the normative contracts: the IR, the parser, the
  runtime, the target.
- `tilefoundry check --help` — the comparison predicates and their bounds, with the
  arithmetic for choosing a tolerance.
- `tilefoundry analyze --help` — flops, traffic, roofline bounds, a predicted time.
- `tilefoundry models` — the models already described, and their authored source to
  copy from.
- `tilefoundry tutorial orchestrator causal_lm` — the shipped autoregressive decode
  loop.
