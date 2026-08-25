# Production scalar-mirror repair

## Outcome

**COMPLETE.** On 2026-08-25 the production `codex` graph's complete live
scalar-mirror mismatch class moved from **3 to 0** through the existing signed
`repair_scalar_projection_mismatches` operator. The applying invocation derived
the exact source closure, sole-live targets, actions, and manifest digest from
the live graph inside the same process that applied them. It did not carry an
authority artifact or digest forward from the diagnosis node.

The fresh preview signed exactly three source ids and admitted **3**, refused
**0**, found **0** already clean, required **3** scalar changes, and required
**0** projection additions. Apply reconciled all three scalars and wrote one
manifest-stamped `StandardNameChange` envelope. Replay returned
`already_applied` with `changed=0` and `persistent_writes=0`.

| Required measure | Live result |
|---|---:|
| Requested / signed rows | **3 / 3** |
| Admitted | **3** |
| Refused | **0** |
| Already clean | **0** |
| Scalar changes | **3** |
| Projection additions | **0** |
| Receipt rows recovered by own keys | **1** |
| `StandardNameChange` | **8,516 → 8,517** |
| Scalar-mirror mismatch census | **3 → 0** |
| Replay | **`already_applied`; changed 0; persistent writes 0** |

The refusal array is present and empty. Therefore there is no refusal reason to
paraphrase or omit; no row was worked around.

## Exact source results

Each source had exactly one non-terminal `PRODUCED_NAME` target, no active claim,
and a scalar selecting its other terminal target. The apply changed only the
scalar mirror; every source-to-name relationship and upstream projection stayed
in place.

| Source id | Scalar before | Fresh sole-live authority and scalar after | Other relationship |
|---|---|---|---|
| `dd:plasma_profiles/ggd/mass_density/values` | `mass_density` | `total_plasma_mass_density` (`accepted`) | `mass_density` (`exhausted`) |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | `radial_ion_momentum` | `radial_ion_momentum_source` (`accepted`) | `radial_ion_momentum` (`exhausted`) |
| `derived:conductivity` | `conductivity` | `plasma_electrical_conductivity` (`accepted`) | `conductivity` (`superseded`) |

For the two DD sources, both the terminal and accepted projections already
existed upstream; this transaction added neither. The derived conductivity
source correctly has no DD projection. The source states after apply retained
the same lifecycle, target relationships, and projection sets while their three
scalars selected the accepted identities shown above.

## Fresh signed authority

The driver first ran the global invariant census and required its complete,
ordered result to equal the three source ids above. It then called
`repair_scalar_projection_mismatches` for those ids without a digest. The
operator read current source properties, every `PRODUCED_NAME` relationship and
target lifecycle, source backing, and backing projections to produce this
preview digest:

`0a88f540d8551871ac7bff01900bb4e037bb9c562c2b2df8dfc2d5399523a9c8`

The apply occurred immediately in the same invocation. The operator re-read and
locked the complete participant closure, rebuilt the authority under lock, and
required its canonical SHA-256 to equal that preview digest before any scalar
compare-and-set. Thus the earlier diagnosis supplied the requested source ids,
but did not supply mutation authority; sole-live authority came from the
production graph at apply time.

Applying source commit: `66685405db9c6564e78df4c7c9492a1745ee84ce`.

## Receipt attribution

The durable receipt was recovered using both keys written by this invocation:

- `run_id=r-20260825T081524480843-n-scalarmirrorrepair`
- `manifest_sha256=0a88f540d8551871ac7bff01900bb4e037bb9c562c2b2df8dfc2d5399523a9c8`

That exact query returned one row:

`sn-change:semantic-mirror-repair:0a88f540d8551871ac7bff01900bb4e037bb9c562c2b2df8dfc2d5399523a9c8`

Its `from_name` map contains all three stale scalar values; its `to_name` map
contains the three freshly derived sole-live targets. A second query by this
run id alone returned that same one row and same one digest, excluding a hidden
receipt under another digest. No operation-name query was used to infer the
receipt or the absence of another receipt.

The single receipt node is linked by three `HAS_INTERNAL_CHANGE` relationships,
one to each accepted target: that relationship count moved **5,398 → 5,401**.
The source-to-target and upstream-projection counts were unchanged at **5,351**
and **4,937** respectively. `LLMCost` stayed **34,104 → 34,104**; this
deterministic repair made no provider call.

## Replay and persistent-write proof

Replay used the same three ids, reason, run id, and manifest digest. Before and
after replay, the driver compared the full persistent counters, the exact three
source snapshots, and the receipt rows selected by the run-and-manifest key.
All were equal. The operator returned `already_applied`, `changed=0`, and the
comparison yielded `persistent_writes=0`.

| Persistent measure | Before replay | After replay | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | 8,517 | 8,517 | **0** |
| `PRODUCED_NAME` relationships | 5,351 | 5,351 | **0** |
| `HAS_STANDARD_NAME` relationships | 4,937 | 4,937 | **0** |
| `HAS_INTERNAL_CHANGE` relationships | 5,401 | 5,401 | **0** |
| `LLMCost` rows | 34,104 | 34,104 | **0** |

## Schema-proven closing census

The closing mismatch result is a real zero over declared keys and the authored
relationship direction, not an empty query produced by a wrong property or
reversed edge.

| Sanity probe after replay | Candidates | With queried key |
|---|---:|---:|
| `StandardName.id` | 4,656 | **4,656** |
| `StandardName.name_stage` | 4,656 | **4,656** |
| `StandardNameSource.id` | 9,668 | **9,668** |
| `StandardNameSource.status` | 9,668 | **9,668** |
| `StandardNameSource.source_type` | 9,668 | **9,668** |
| Authored `StandardNameSource -[:PRODUCED_NAME]-> StandardName` target keys | 5,351 | **5,351** |

The reversed `StandardName -[:PRODUCED_NAME]-> StandardNameSource` count was
**0**. Against the 5,351 authored relationships whose targets all carried both
`id` and `name_stage`, the global composed/attached source query found
**0 scalar-mirror mismatches after apply and 0 after replay**.

## Durable machine record

- Result JSON:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T081524480843-n-scalarmirrorrepair/scalar-mirror-repair-result.json`
  — SHA-256
  `df7917960c7221524ae3f13fd07f4c8921da5a23006a902d8ed27b9ec428477b`.
- Invocation diagnostics:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T081524480843-n-scalarmirrorrepair/scalar-mirror-repair.log`
  — SHA-256
  `c91ed578d52763adf99dd773a651217940cadeddf926079523fd6cbd86d88d86`.

The diagnostics contain only `uv`'s notice that an inherited Reckon
`VIRTUAL_ENV` was ignored in favor of the explicitly selected shared
imas-codex project environment. The Python invocation exited zero.

The separate prevention repair for orphan-parent reconciliation remains owned
by its concurrent node. This transaction did not run the ordinary Standard
Names loop after apply; the production mismatch class is zero now, and the
selector guard must be integrated before a later ordinary reconcile can reuse
the migrated `derived:conductivity` source against its terminal predecessor.
