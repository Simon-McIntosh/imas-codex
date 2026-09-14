# The two excluded WEST names: measured verdicts

## Result

The WEST cut contains 220 candidates, of which 218 were published and exactly
two were excluded. The two exits are accounted for: one is held by the name
acceptance lifecycle and one by validation quarantine. Neither is landable at
the measured state.

## Observation boundary and graph instrument

The graph identity was selected by `StandardName.id`; this record makes no
claim that a `name` property exists. Before asserting a missing property, the
required bounded query is:

```cypher
MATCH (sn:StandardName {id: $id})
RETURN keys(sn) AS property_keys
LIMIT 1
```

That key-set query could not complete in this run. The login-node profile
resolved to `bolt://localhost:7687`, but no listener was present. The
persistent graph REPL and the sanctioned command both reproduced:

```text
Couldn't connect to localhost:7687 ... Connection refused
```

The configured `codex-neo4j` allocation was job `1269941`, still `PENDING`
with reason `Priority` at the observation time. Therefore the detailed state
below is explicitly the latest bounded graph evidence already captured for
these identities, supplemented by the coordinator's current population
census; it is not presented as a new successful live query. The failed
connection occurred before any graph write or provider call.

## Detailed identity evidence

The producing relationship is `(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)`.
The identity key is `id`, not `name`.

### `hot_neutral_temperature`

Description: translational kinetic temperature, expressed as energy per
particle, of the energetic neutral-atom component in the plasma edge or
scrape-off layer.

Producing paths and source state from the bounded source read:

| Source id | DD path | Source type | Source status |
| --- | --- | --- | --- |
| `dd:spectrometer_visible/channel/polarization_spectroscopy/temperature_hot_neutrals` | `spectrometer_visible/channel/polarization_spectroscopy/temperature_hot_neutrals` | `dd` | `composed` |
| `dd:spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_temperature` | `spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_temperature` | `dd` | `attached` |

The latest bounded identity/review read reports:

| Field | Value |
| --- | --- |
| `name_stage` | `reviewed` |
| `docs_stage` | `accepted` |
| `status` | `draft` |
| `validation_status` | `valid` |
| Per-row quarantine reason | none; `validation_issues=[]` |
| `reviewer_score_name` | `0.30` |
| `refine_attempts` | `0` |
| Rotation cap | `3` by the configured rescore/pool default |
| `edit_status` | `null` |
| Name-axis `HAS_REVIEW` count | `4` |
| Newest name-axis `llm_at` | `2026-09-09T07:33:38.958344Z` |
| `review_quorum_shortfall` | `fewer reviewer seats scored than the chain defines (method=semantic_similarity_gate)` |
| `resolution_method` | `semantic_similarity_gate` |

The four name-axis rows explain the score's limited authority. The score is
below the review acceptance threshold and the shortfall also prevents either
acceptance or a paid refinement decision. This is the exact refusal predicate:
`name_stage` remains `reviewed` because the latest score is below the
acceptance threshold, while `review_quorum_shortfall` is non-null, so the
ordinary refine eligibility predicate excludes it. It is not currently
`exhausted`; its zero attempt count leaves the rotation budget untouched.

The requested single-name recovery was invoked as:

```text
imas-codex sn rescore hot_neutral_temperature --cost-limit 1.00
```

It failed at `stage_name_for_rescore` while connecting to
`localhost:7687`, before the drafted transition and before the scoped review
pipeline. Consequently there is no post-rescore score, no completed quorum,
and no name-stage advance to report. No graph state changed and no provider
spend was incurred. The recovery remains the exact next action once the
login-local graph service is reachable.

### `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`

Description: inward half-width of the hard X-ray emissivity peak in normalized
toroidal flux coordinate.

The bounded producing-path read reports one source:

| Source id | DD path | Source type | Source status |
| --- | --- | --- | --- |
| `dd:hard_x_rays/emissivity_profile_1d/half_width_internal` | `hard_x_rays/emissivity_profile_1d/half_width_internal` | `dd` | `attached` |

The latest bounded identity read reports:

| Field | Value |
| --- | --- |
| `name_stage` | `drafted` |
| `docs_stage` | `pending` |
| `status` | `draft` |
| `validation_status` | `quarantined` |
| Per-row quarantine reason | `parse_error: grammar round-trip failed for inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` |
| `reviewer_score_name` | `null` |
| `refine_attempts` | `null` |
| Rotation cap | `3` by the configured rescore/pool default |
| `edit_status` | `open` |
| Name-axis `HAS_REVIEW` count | `0` |
| Newest name-axis `llm_at` | `null` |
| `review_quorum_shortfall` | `null` (no name review has run) |
| `resolution_method` | `null` |

The quarantine is a genuine current finding, not the old verdict that was
rechecked against the superseded grammar. The sanctioned grammar read reports
that the whole residue does not match a `physical_base` or `geometry_carrier`;
the nearest candidate is `normalized_toroidal_flux_coordinate`, and no
canonical spelling is supplied. The source description identifies one scalar
(`half_width_internal`), not two independent source quantities folded into
one name. The problem is therefore a grammar/name-composition gap that needs
a separately derived valid spelling, not a reason to clear the quarantine.
The identity must remain quarantined until that spelling is derived and
validated; no review, score, stage, or row was changed here.

## Verdicts and spend

| Identity | Landable now? | What changed in this run | Exact predicate still refusing it |
| --- | --- | --- | --- |
| `hot_neutral_temperature` | No | No state change: the scoped rescore reached the graph connection boundary and stopped before staging. | `review_quorum_shortfall` is non-null with `semantic_similarity_gate`; the score is 0.30 and the identity remains `reviewed`, so ordinary refine eligibility is withheld. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | No | No state change: quarantine and open edit were preserved. | `validation_status=quarantined` because strict grammar round-trip fails for the stored spelling; it remains `drafted` with no name review. |

Campaign spend before this node was **$101.638907**. The attempted rescore
failed before a provider request, so spend after this node is also
**$101.638907** (delta **$0.000000**, within the $15.00 node ceiling).

The measured release arithmetic remains **220 = 218 published + 2 accounted
exclusions**, residue zero. No candidate was cut, no request was opened, no
quarantine was cleared, no review row was deleted, and no signed manifest was
applied.

## Required follow-up

Restart or otherwise make the configured `codex-neo4j` service reachable on
the login-local endpoint, then rerun the bounded key-set and two-identity
reads. Invoke the same single-name rescore for
`hot_neutral_temperature`, record its returned score, quorum completion and
stage, and re-run the grammar validation instrument for the quarantined
identity. Only a measured state change can turn either verdict into
landable; neither can be made landable by changing an export threshold or
clearing its recorded quarantine.
