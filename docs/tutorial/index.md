# TileFoundry in three steps

TileFoundry is source to source: the reference is source code, the fast
implementation is source code, and either can be pointed at any command.
`check` says whether two of them agree; `analyze` says what one costs.

```text
  step one — describe it, until it agrees

        ┌────────── fix the HIR ◄────────── not yet ───────────┐
        ▼                                                      │
   authored HIR ─────► check ─────► agrees? ───────────────────┘
   the reference                       │
        ▲                             yes
        │                              ▼
   published model            the reference is finished

  step two — choose the schedule before writing a kernel

        ┌────── change the HIR ◄────── not yet ───────────────────┐
        ▼                                                         │
   authored HIR ─────► analyze ─────► predicted performance ok? ──┘
                                              │
                                             yes
                                              ▼
       candidates + facts ─────► tf.schedule ─────► finalize ─────► TIR

  step three — implement, check, and measure

        ┌────── revise the schedule ◄────── not yet ──────────────────┐
        ▼                                                             │
       TIR ─────► agent writes CUDA ─────► check ─────► agrees? ──────┘
                                                 yes │
                                                     ▼
                                                  measure
                                                     │
                                       measured performance ok?
                                             ┌───────┴───────┐
                          no: return to step two             └──► ship
```

The [schedule page](schedule.md) takes step two from a CTA-grid HIR through
candidate discovery, instruction facts, persistent scheduling, and finalized TIR.
