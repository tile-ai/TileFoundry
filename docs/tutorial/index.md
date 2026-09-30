# TileFoundry in two steps

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

  step two — choose the schedule, implement, and measure

        ┌────── change the HIR ◄────── not yet ───────────────────┐
        ▼                                                         │
   authored HIR ─────► analyze ─────► predicted performance ok? ──┘
                                              │
                                             yes
                                              ▼
       candidates + facts ─────► tf.schedule ─────► finalize ─────► TIR

        ┌────── revise the schedule ◄────── not yet ──────────────────┐
        ▼                                                             │
       TIR ─────► implement a backend ───► check ─────► agrees? ──────┘
                                                 yes │
                                                     ▼
                                                  measure
                                                     │
                                       measured performance ok?
                                             ┌───────┴───────┐
                          no: return to step two             └──► ship
```
