# Strain-gauge ordinal identity edit report

## Outcome

All six current accepted identities were submitted individually through the sanctioned `sn edit --rename` path with a mandatory, DD-grounded reason. Every exact non-ordinal measurement-direction proposal was refused before graph mutation because the public ISN grammar has no non-ordinal `measurement_direction_unit_vector` physical base or geometry carrier. The grammar reported the same nearest vocabulary for every axis: `first_measurement_direction_unit_vector`, `second_measurement_direction_unit_vector`, and generic `direction_unit_vector`.

This is the vocabulary-gap outcome anticipated by the live plan. No nearest-object or alternate-locus substitution was made. In particular, the grammar-valid `x|y|z_direction_unit_vector_of_strain_gauge` alternative was not applied: a live attempt for the X first-direction row was transactionally rolled back by the attachment guard because the DD path is owned by `sensor`, not `strain_gauge`. The DD-grounded generic sensor identities already exist, so minting a near-duplicate would also be wrong.

## Dry-run gate

No live edit was attempted until all six final candidate renames had been previewed with `--dry-run`. The previews of the exact non-ordinal `measurement_direction_unit_vector` spellings consistently reported the grammar gap. A second preview series established that the generic `direction_unit_vector` spelling was grammar-valid, after which the attachment/source-owner check demonstrated why it was not an admissible strain-gauge replacement. Dry-run commands made no graph writes and created no edit run IDs.

## Six sanctioned invocations

Each command below was first run with `--dry-run`, inspected, and then issued live without `--dry-run`. Each live invocation terminated at deterministic grammar validation before review or persistence.

| Current accepted identity | DD source binding | Proposed non-ordinal identity | Mandatory reason | Live outcome |
|---|---|---|---|---|
| `x_first_measurement_direction_unit_vector_of_strain_gauge` | `dd:operational_instrumentation/sensor/direction/x` | `x_measurement_direction_unit_vector_of_strain_gauge` | `DD path operational_instrumentation/sensor/direction/x encodes one ordered strain-gauge measurement-direction slot. Ordered sample positions belong in DD provenance, not Standard Name identity; preserve X-axis measurement-direction unit-vector semantics without semantic substitution.` | Vocabulary gap: `x_measurement_direction_unit_vector` does not match a physical base or geometry carrier. |
| `x_second_measurement_direction_unit_vector_of_strain_gauge` | `dd:operational_instrumentation/sensor/direction_second/x` | `x_measurement_direction_unit_vector_of_strain_gauge` | `DD path operational_instrumentation/sensor/direction_second/x encodes one ordered strain-gauge measurement-direction slot. Ordered sample positions belong in DD provenance, not Standard Name identity; preserve X-axis measurement-direction unit-vector semantics without semantic substitution.` | Same vocabulary gap. |
| `y_first_measurement_direction_unit_vector_of_strain_gauge` | `dd:operational_instrumentation/sensor/direction/y` | `y_measurement_direction_unit_vector_of_strain_gauge` | `DD path operational_instrumentation/sensor/direction/y encodes one ordered strain-gauge measurement-direction slot. Ordered sample positions belong in DD provenance, not Standard Name identity; preserve Y-axis measurement-direction unit-vector semantics without semantic substitution.` | Vocabulary gap: `y_measurement_direction_unit_vector` does not match a physical base or geometry carrier. |
| `y_second_measurement_direction_unit_vector_of_strain_gauge` | `dd:operational_instrumentation/sensor/direction_second/y` | `y_measurement_direction_unit_vector_of_strain_gauge` | `DD path operational_instrumentation/sensor/direction_second/y encodes one ordered strain-gauge measurement-direction slot. Ordered sample positions belong in DD provenance, not Standard Name identity; preserve Y-axis measurement-direction unit-vector semantics without semantic substitution.` | Same vocabulary gap. |
| `z_first_measurement_direction_unit_vector_of_strain_gauge` | `dd:operational_instrumentation/sensor/direction/z` | `z_measurement_direction_unit_vector_of_strain_gauge` | `DD path operational_instrumentation/sensor/direction/z encodes one ordered strain-gauge measurement-direction slot. Ordered sample positions belong in DD provenance, not Standard Name identity; preserve Z-axis measurement-direction unit-vector semantics without semantic substitution.` | Vocabulary gap: `z_measurement_direction_unit_vector` does not match a physical base or geometry carrier. |
| `z_second_measurement_direction_unit_vector_of_strain_gauge` | `dd:operational_instrumentation/sensor/direction_second/z` | `z_measurement_direction_unit_vector_of_strain_gauge` | `DD path operational_instrumentation/sensor/direction_second/z encodes one ordered strain-gauge measurement-direction slot. Ordered sample positions belong in DD provenance, not Standard Name identity; preserve Z-axis measurement-direction unit-vector semantics without semantic substitution.` | Same vocabulary gap. |

The live command shape for each row was:

```text
uv run --no-sync imas-codex sn edit <current-identity> \
  --rename <proposed-non-ordinal-identity> \
  --reason <mandatory-DD-grounded-reason> \
  --scope self --cost-limit 2
```

The six per-command ceilings sum to the authorized USD 12.00 ceiling.

## Graph and provenance verification

The post-invocation query used the schema-owned `StandardName.id` key and proved its instrument coverage before trusting any absence: 4,675 `StandardName` candidates and 4,675 with `id`. It then joined each source in the authored direction `(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)`.

Results:

- All six original identities remain `name_stage=accepted` and `validation_status=valid`, each with exactly its original DD source binding.
- Zero nodes exist for the three proposed non-ordinal identities.
- The attempted generic strain-gauge rename was explicitly reported as “rename rolled back” by the attachment fence; it left no successor or source migration.
- No new `sn-edit-*` run ID was created because all six exact proposals failed grammar validation before persistence.
- Therefore there was no identity change to attribute to a direct write. The graph contains zero changes from this node; all attempted changes passed through the sanctioned edit preflight, and none bypassed it. The existing edit fields on the six originals remain their earlier pipeline provenance (`edit_mode=hint`, `edit_status=applied`, with recorded DD-drift reasons), not evidence of a write by this node.

This is the strongest possible no-hand-edit result for a vocabulary-gap outcome: unchanged identities, unchanged source edges, zero proposed nodes, zero new edit runs, and zero direct graph writes.

## Cost evidence

An `LLMCost` query covered all six current identities plus all three proposed identities from the dispatch start time `2026-09-01T14:12:08Z` onward. It returned:

- attributable LLM calls: **0**
- attributable run IDs: **0**
- exact attributable spend: **USD 0.000000**
- authorized ceiling: **USD 12.000000**
- ceiling consumed: **0.0%**

The zero is expected: grammar and attachment validation refused the proposals before the inline review pool could make an LLM call.

## Disposition

The six ordinal identities remain visible and unchanged. Their non-ordinal measurement-direction form requires an ISN vocabulary decision; codex must not redefine the grammar or substitute a nearby object. The appropriate follow-on is an ISN-owned vocabulary addition or an explicit decision to fold these DD sources onto the already-existing generic sensor direction-vector identities through a sanctioned operation that can target an existing identity.
