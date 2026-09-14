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

The service allocation was job `1269941`, running on
`98dci4-gpu-0002`. No address was hardcoded: `resolve_neo4j()` returned
`bolt://98dci4-gpu-0002:7687`, and `GraphClient.from_profile()` used that
result. The key-set read completed in 0.189009 seconds and returned:

```text
run_id, review_resolution_method, grammar_parse_version, updated_at,
embed_text_hash, reviewer_model_name, reviewed_name_at,
reviewer_comments_per_dim_name, reviewer_scores_name, reviewer_comments_name,
validation_diagnostics_json, review_quorum_shortfall_at, validated_at,
review_quorum_shortfall, semantic_sim, source_dd_resolution_marker,
source_raw_documentation, source_dd_resolution_manifest_digest,
validation_status, reviewer_score_name, reviewer_model_docs, docs_model,
review_mean_score, source_dd_resolution_ids,
source_dd_resolution_converged_ids, reviewer_comments_docs,
review_count, source_paths, source_raw_unit, reviewed_docs_at,
review_disagreement, llm_cost, reviewer_score_docs, link_status,
llm_cost_review_docs, docs_generated_at, review_docs_count,
reviewer_scores_docs, generate_docs_count, reviewer_comments_per_dim_docs,
validation_layer_summary, source_path, docs_chain_length,
review_resubmit_count, claim_seq, source_documentation, source_unit,
harmonized_at, docs_stage, review_input_hash, harmonized_group_signature,
embedded_at, validation_issues, population, embedding, subject, name_stage,
physical_base, kind, origin, physics_domain, documentation, id, created_at,
links, catalog_commit_sha, imported_at, unit, source_types, status,
source_domains, description
```

This confirms that the identity property is `id`, while `name`,
`refine_attempts`, and `edit_status` are absent on this particular node. The
two-id state, source and review reads completed in 0.333534, 0.840012 and
0.359916 seconds respectively, all below the ten-second ceiling. A projected
null on the second identity is reported as null rather than treated as proof
that its property key is absent.

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

The bounded reads immediately before and after recovery report:

| Field | Before rescore | After rescore |
| --- | --- | --- |
| `name_stage` | `reviewed` | `reviewed` |
| `docs_stage` | `accepted` | `accepted` |
| `status` | `draft` | `draft` |
| `validation_status` | `valid` | `valid` |
| Per-row quarantine reason | none; `validation_issues=[]` | none; `validation_issues=[]` |
| `reviewer_score_name` | `0.30` | `0.30` |
| Raw `refine_attempts` | `null` | `null` |
| Effective attempts against cap | `coalesce(null, 0) = 0` of configured cap `3` | `0/3` |
| `edit_status` | `null` | `null` |
| Name-axis `HAS_REVIEW` count | `4` | `5` |
| Newest name-axis `llm_at` | `2026-09-09T07:33:38.958344Z` | `2026-09-14T09:12:20.463703Z` |
| `review_quorum_shortfall` | `fewer reviewer seats scored than the chain defines (method=semantic_similarity_gate)` | unchanged |
| `resolution_method` | `semantic_similarity_gate` | `semantic_similarity_gate` |
| `run_id` | prior run | `sn-rescore-20260914T090836Z` |

Before recovery, all four name-axis rows carried score `0.30` and
`resolution_method=semantic_similarity_gate`. The fifth row created by the
recovery carries the same score and method, with `llm_cost=0.0`. The score's
authority therefore did not improve: the shortfall prevents either acceptance
or a paid refinement decision. The exact refusal predicate remains a non-null
`review_quorum_shortfall`; the ordinary refine eligibility predicate skips the
row, while the score remains below the acceptance threshold. The identity is
not exhausted and its `0/3` rotation budget remains untouched.

The requested single-name recovery was invoked as:

```text
imas-codex sn rescore hot_neutral_temperature --cost-limit 15.00
```

The sanctioned command staged `reviewed -> drafted`, ran only this identity,
and returned exit 3 with the explicit non-accepting receipt:

```text
rescored hot_neutral_temperature
  reviewed -> drafted
  run_id=sn-rescore-20260914T090836Z
Outcome: below threshold
Stage: reviewed
Score: 0.30
Inline review cost: $0.0000
```

The quorum did **not** complete and the name made no net lifecycle advance: it
returned to `reviewed`, added one review row, retained the same score and
shortfall, and spent no refinement attempt. Exit 3 is the command's documented
negative outcome for a successor that did not land, not an infrastructure
failure.

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
| Raw `refine_attempts` projection | `null` |
| Effective attempts against cap | `coalesce(null, 0) = 0` of configured cap `3` |
| `edit_status` | `open` |
| Name-axis `HAS_REVIEW` count | `0` |
| Newest name-axis `llm_at` | `null` |
| `review_quorum_shortfall` | `null` (no name review has run) |
| `resolution_method` | `null` |

The post-rescore bounded read confirms that this row is unchanged. Its
quarantine is a genuine current finding, not the old verdict that was
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
| `hot_neutral_temperature` | No | The designed rescore staged it, added one zero-cost name review, and returned it to `reviewed`; score stayed 0.30, quorum stayed incomplete, and attempts stayed 0/3. | `review_quorum_shortfall` is non-null with `semantic_similarity_gate`; name acceptance and ordinary refinement are both withheld. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | No | No state change: quarantine and open edit were preserved. | `validation_status=quarantined` because strict grammar round-trip fails for the stored spelling; it remains `drafted` with no name review. |

The live `LLMCost` ledger immediately before recovery contained 1,692 rows and
totalled **$101.638907**. Immediately afterward it still contained 1,692 rows
and totalled **$101.638907**. The attributable delta is therefore
**$0.000000**, matching the command's `$0.0000` receipt and remaining within
the $15.00 node ceiling.

The measured release arithmetic remains **220 = 218 published + 2 accounted
exclusions**, residue zero. No candidate was cut, no request was opened, no
quarantine was cleared, no review row was deleted, and no signed manifest was
applied.

## Required follow-up

`hot_neutral_temperature` has now exercised its designed recovery, but the
recovery itself again terminated at `semantic_similarity_gate` rather than a
complete reviewer quorum. The next repair must establish why a rescore that is
defined to buy a fresh quorum is still persisted as a non-quorate semantic-gate
decision; blindly repeating the same command would only append another
identical zero-cost review row. The long hard-X-ray identity separately needs
a valid spelling derived for its one source quantity and then sanctioned
revalidation. Neither can be made landable by lowering an export threshold or
clearing its recorded quarantine.
