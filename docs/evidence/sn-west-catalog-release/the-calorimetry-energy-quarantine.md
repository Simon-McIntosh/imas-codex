# The calorimetry energy quarantine is revalidated

**Verdict: the quarantine on
`accumulated_total_coolant_absorbed_energy_of_calorimetry_component` was a
stale-state defect, and the sanctioned id-scoped validation drain cleared it.**
The spelling parses cleanly under the installed `imas-standard-names` 0.9.3
(`physical_base='coolant_absorbed_energy'`,
`geometry=CALORIMETRY_COMPONENT`, `aggregation=TOTAL`,
`transformation='accumulated'`), so the recorded `refine_stop_reason:
grammar_invalid` no longer describes it. Re-running the deterministic admission
gate over exactly this one identity (`drain_validation_for_ids([sn_id])`)
moved it `quarantined → valid` with empty `validation_issues` and a freshly
observed `validated_at`, without writing `validation_status` directly and
without touching its accepted sibling
`accumulated_coolant_absorbed_energy_of_calorimetry_component`. The identity
already holds a scored name review (`0.75`, recorded 2026-09-15T00:33:49Z). A
fresh WEST cut would **still not carry it**, but for reasons unrelated to the
quarantine: `name_stage=exhausted` (excluded as `name_not_accepted`, even in a
names-only cut) and `docs_stage=pending` (excluded as
`documentation_not_accepted` in a docs-carrying cut).

![Calorimetry quarantine before / sanctioned drain / after](/imas-codex/figures/sn-west-catalog-release/calorimetry-quarantine-revalidated.svg)

## Starting state, quoted from the live graph (2026-09-15)

Bounded indexed id read on the login node (Neo4j tunnel is login-node-local;
single-node read, well under the ten-second ceiling):

| field | value |
|---|---|
| `id` | `accumulated_total_coolant_absorbed_energy_of_calorimetry_component` |
| `name_stage` | `exhausted` |
| `validation_status` | `quarantined` |
| `reviewer_score_name` | `0.75` |
| `validation_issues` | `null` (field absent from node) |
| `quarantine_reason` | `null` (field absent from node) |
| `validated_at` | `null` (field absent from node) |
| `refine_stop_reason` | `grammar_invalid` |
| `refine_stopped_at` | `2026-09-15T00:34:38.532Z` |
| `docs_stage` | `pending` |
| `source_paths` | `["dd:calorimetry/group/component/energy_total/data"]` |
| `grammar_parse_version` | `0.9.3` |
| `physics_domain` | `mechanical_measurement_diagnostics` |
| `reviewer_scores_name` | `{"grammar":20,"semantic":12,"convention":10,"completeness":18}` |

Its stored review comment (the same rotation that scored it) ends with the
`[grammar_invalid]` refusal: *"refined candidate failed strict grammar
validation: 1 validation error for StandardName object …"*.

## The quarantine reason is stale — verified with the installed parser first

The recorded stop reason claims the refined candidate failed strict grammar
validation. Under the installed `imas-standard-names` 0.9.3 (the same
`grammar_parse_version` the node stores), the stored spelling parses cleanly:

```text
parse_standard_name('accumulated_total_coolant_absorbed_energy_of_calorimetry_component')
  -> PARSE OK
     transformation = 'accumulated'
     aggregation    = <Aggregation.TOTAL: 'total'>
     qualifiers     = ['coolant', 'absorbed']
     physical_base  = 'coolant_absorbed_energy'
     geometry       = <Position.CALORIMETRY_COMPONENT: 'calorimetry_component'>
```

A quarantine whose recorded cause has lapsed is not evidence of a current
defect — it is stale state. The remedy is re-observation through the sanctioned
route, never a direct status write and never a rename.

## The sanctioned revalidation route, and the route's output

Established that re-running validation over a quarantined identity goes through
`drain_validation_for_ids` (`imas_codex/standard_names/workers.py:5016`): it
claims only named nodes whose `validated_at` is null
(`claim_ids_for_validation`, `workers.py:4962` — the identity already qualified,
so no `--revalidate` stamp-clear was needed), runs the shared admission gate
`validate_name_candidate` (ISN round-trip, three ISN layers, post-generation
audits; LLM-free), and re-stamps `validation_issues` / `validation_status` /
`validated_at` under a claim token. A genuine defect re-quarantines instead of
washing to `valid`. The CLI `--revalidate` sweep is the same mechanism scoped
to a domain, and was not needed here.

Drain output (live, 2026-09-15T06:27Z):

```json
{
  "cleared_ids": ["accumulated_total_coolant_absorbed_energy_of_calorimetry_component"],
  "quarantined": 0,
  "requarantined_ids": [],
  "validated": 1
}
```

## State read back from the graph afterwards

| field | after |
|---|---|
| `validation_status` | `valid` (was `quarantined`) |
| `validation_issues` | `[]` (was `null`) |
| `validated_at` | `2026-09-15T06:27:09.573Z` (was `null`) |
| `validation_layer_summary` | `{"pydantic":{"passed":true,"error_count":0},"semantic":{"issue_count":0},"description":{"issue_count":0},"structural":{"issue_count":0},"canonical":{"issue_count":0}}` |
| `name_stage` | `exhausted` (unchanged) |
| `docs_stage` | `pending` (unchanged) |
| `reviewer_score_name` | `0.75` (unchanged; scored review already recorded) |
| `quarantine_reason` | `null` (no code writes this field; it was never the mechanism) |

## Would the WEST cut carry it?

Evaluated through the authentic code path (`_classify_export_population`,
`imas_codex/standard_names/export.py:756`). An identity is carried only when:
`validation_status='valid'` **and** `_validation_observed_at` (validated_at)
present **and** `name_stage` in `{accepted, approved}` **and**, on a
docs-carrying cut, `docs_stage='accepted'`, winning docs review and quorum
present.

- The quarantine is no longer the barrier: `validated_at` observed.
- `name_stage=exhausted` → excluded, reason `name_not_accepted`. This predicate
  is **not** gated on `names_only`, so even a names-only cut withholds it.
- `docs_stage=pending` → excluded, reason `documentation_not_accepted`, on a
  docs-carrying cut.

So the WEST cut would **not** yet carry the identity; the remaining barriers are
lifecycle stages (`exhausted` name, `pending` documentation), not validation.

## Distinct sibling — not collapsed onto

`accumulated_coolant_absorbed_energy_of_calorimetry_component` (the accepted
identity, `name_stage=accepted`, `validation_status=valid`,
`reviewer_score_name=0.9875`, `docs_stage=accepted`, produced by
`dd:calorimetry/group/component/energy_cumulated`) was **not** touched by the
drain and remains distinct. Read against the Data Dictionary, the two derive
from different DD leaves that describe the same coolant-absorbed calorimetric
energy gathered over the discharge but at different granularities: the
quarantined identity's `calorimetry/group/component/energy_total` is a single
per-discharge `FLT_0D` scalar of the total energy extracted from the component
including the post-pulse recovery phase, whereas the accepted sibling's
`calorimetry/group/component/energy_cumulated` is a `FLT_1D` array of the
cumulative thermal energy extracted from the coolant since pulse initiation —
so the `total` token restates the totality `accumulated` already carries and
encodes no additional physical distinction, which is exactly the redundancy the
scored review (0.75) docked under convention and semantic.

## Scope and standing

Only this one `StandardName` node had a validation field touched; the sibling
was verified unchanged by the same bounded read. No `validation_status` write
was authored; the drain wrote it under a claim token from the shared admission
gate. No signed manifest apply and no broad pipeline were run. All graph
queries were bounded id-scoped reads/writes on the login node (graph
connectivity is login-node-local), each well under the ten-second ceiling.
The accepted sibling's release-readiness is unchanged.

## Budget

The revalidation drain is LLM-free (deterministic gate), so this node added
**$0.00** to the campaign. Campaign total before: **$107.18** (against a
$250.00 ceiling); after: **$107.18**. Nothing was deleted.

## Reproduction

Driver at `/tmp/revalidate_calorimetry.py` (opened `GraphClient()`, read
`before`, ran `asyncio.run(drain_validation_for_ids([sn_id]))`, read `after`).
The live before/drain/after JSON captured above reproduces by re-running that
driver verbatim against the live graph.
