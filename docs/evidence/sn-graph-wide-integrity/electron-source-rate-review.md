# Electron source-rate ordinary review

## Outcome

Status: **complete**. Ordinary review replaced the volume-integrated identity
`electron_source_rate` with the DD-grounded local identity
`electron_volumetric_source_rate`. The successor is valid and accepted at a
fresh name-quorum score of **0.94375**. No identity was hand-accepted or
hand-promoted, and no graph unit was hand-edited.

The measured before/after identity is:

| State | Identity | Stage | Unit property | `HAS_UNIT` edge | Producing sources |
|---|---|---|---|---|---:|
| Before | `electron_source_rate` | reviewed | `s^-1` | `s^-1` | 2 |
| After | `electron_source_rate` | superseded | `s^-1` | `s^-1` | 0 |
| After | `electron_volumetric_source_rate` | accepted | `m^-3.s^-1` | `m^-3.s^-1` | 2 |

The semantic correction is material rather than cosmetic. The predecessor's
description and documentation defined a volume integral, consistent with a
total rate in `s^-1`. Both attached DD paths instead represent local terms in
the electron-density continuity equation, so their common `m^-3.s^-1` unit
requires the volumetric identity. The successor documentation now states that
the quantity is local and is not integrated.

## DD authority and source reset

The live preflight found exactly the two planned sources, both at
`status='failed'`, `attempt_count=5`, and
`last_error='compose claim-attempt cap reached'`:

- `dd:core_sources/source/profiles_1d/electrons/particles_decomposed/explicit_part`
  is a `FLT_1D` explicit source term added directly to the electron-density
  transport equation. Its DD scalar unit and `HAS_UNIT` edge are both exactly
  `m^-3.s^-1`.
- `dd:edge_sources/source/ggd/electrons/particles/values` is a `FLT_1D` source
  term on grid subsets. Its DD scalar unit and `HAS_UNIT` edge are both exactly
  `m^-3.s^-1`.

The sanctioned `sn retry --failed` dry-run selected **2 of 2** requested
sources. The applied retry reset **2 of 2**. Postflight found both sources at
`status='attached'`, `attempt_count=0`, and `last_error=null`, with each source
carrying exactly one `PRODUCED_NAME` edge to
`electron_volumetric_source_rate`.

## Governed edit and fresh review

The dry-run of `sn edit electron_source_rate --rename
electron_volumetric_source_rate --scope self` previewed exactly one rename and
no family or subtree cascade. The applied invocation carried the mandatory
DD-grounded reason, used scope run `sn-edit-20260821T140829Z`, and passed
`--cost-limit 15`.

Ordinary review wrote **4 fresh quorum rows**, all after the preflight's latest
candidate review:

| Axis | Role | Model | Score | Resolution |
|---|---|---|---:|---|
| names | primary | `openrouter/x-ai/grok-4.5` | 0.8875 | blind primary |
| names | secondary | `openrouter/openai/gpt-5.6-luna` | 1.0000 | quorum consensus |
| docs | primary | `openrouter/anthropic/claude-sonnet-5` | 0.8875 | blind primary |
| docs | secondary | `openrouter/x-ai/grok-4.5` | 0.9500 | quorum consensus |

The name-axis quorum resolved to **0.94375**, promoted the successor to
`name_stage='accepted'`, and left it `validation_status='valid'`. The unit
derivation used the complete DD cohort: successor property
`unit='m^-3.s^-1'`, successor unit edge `m^-3.s^-1`, DD scalar units
`m^-3.s^-1` for **2 of 2** sources, and DD unit edges `m^-3.s^-1` for **2 of
2** sources.

The global `LLMCost` ledger moved from **27,614 rows / $1,366.082862** to
**27,619 rows / $1,366.279974** during the serialized invocation. Actual spend
was therefore **$0.197112 against the $15.00 cap**, leaving **$14.802888**
unused. The CLI independently reported `$0.1971`.

## Integrity tests

The credentialed graph selection was run with `-m graph`, so both named tests
were executed rather than deselected:

- Before: **2 failed / 0 passed**, exit **1**. Both tests reported exactly the
  two `electron_source_rate (s^-1)` to DD `m^-3.s^-1` disagreements.
- After: **2 passed / 0 failed**, exit **0**, in 8.68 seconds.

An earlier selector omitted `-m graph`, was deselected by the repository's
default marker expression, and exited **5** with zero tests executed. That run
is retained as `before-tests.log` for auditability but is explicitly not used
as evidence; `before-tests-graph.log` is the valid baseline.

No raw Cypher mutation was used. No Standard Name or DD unit property or unit
edge was edited by hand. Acceptance came only from the fresh quorum score.

## Artifacts

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/graph-before.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/graph-after.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/retry-dry-run.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/retry-apply.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/edit-dry-run.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/edit-review.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/before-tests-graph.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T140135028608-esrate2/after-tests.log`
