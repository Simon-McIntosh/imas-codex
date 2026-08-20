# Ordinary-review gating cohort

## Outcome

The exact remaining review gate contains **19 unique identities**:

- **3 ancestor-fold identities** already exist in the live graph. All three are
  `name_stage=reviewed`, `docs_stage=pending`, and `validation_status=valid`.
- **16 owner/geometry identities** are absent from the live graph: 13 are the
  unique identities behind the authority mapping's 20 absent-target rows, and
  3 are the now-grammatical `polarizer`, `pellet_fragment`, and
  `active_spatial_resolution_zone` identities admitted by the pinned ISN
  vocabulary.

The machine-readable receipt is
[`review-cohort-manifest.json`](review-cohort-manifest.json), SHA-256
`e9631f6b41cd35885eec1fc0b3b508c20aba3bb4273fb1ced3bb73ed669a4c85`.
It records all four requested lifecycle fields for every identity. Null fields
on the 16 owner/geometry rows are the measured result of graph absence, not
missing receipt data.

This is narrower than the approximate live-plan estimate of “23 identities” because the
receipt deduplicates the 20 absent owner/geometry **rows** into 13 identities,
adds three unique identities unblocked by the vocabulary rotation, adds the
three named fold ancestors, and excludes unrelated carried single-name work.
Already accepted owner targets do not gate ordinary review.

## Live ancestor-fold gate

| Identity | Live lifecycle | Score | Refine attempts | Representative source binding | Description |
|---|---|---:|---|---|---|
| `radial_ion_momentum` | reviewed / docs pending / valid | 0.67500 | null | `edge_sources/source/ggd/ion/momentum/r` | Radial component of ion momentum flux, summed over charge states |
| `radial_momentum_flux` | reviewed / docs pending / valid | 0.93125 | null | `plasma_transport/model/ggd/momentum/flux/radial` | Radial momentum flux across grid facets for all species |
| `poloidal_neutral_state_momentum_flux` | reviewed / docs pending / valid | 0.90000 | null | `plasma_transport/model/ggd/neutral/state/momentum/flux/poloidal` | Surface-normal flux density of poloidal momentum carried by a specified neutral state |

Each can advance immediately through the supported recovery CLI; this stages
the same identity for a fresh ordinary quorum draw and never hand-accepts it:

```text
uv run imas-codex sn rescore radial_ion_momentum --cost-limit 1.0
uv run imas-codex sn rescore radial_momentum_flux --cost-limit 1.0
uv run imas-codex sn rescore poloidal_neutral_state_momentum_flux --cost-limit 1.0
```

Acceptance is not presumed. In particular,
`radial_ion_momentum_source` describes a source term while
`radial_ion_momentum` is documented as flux; a fresh quorum may accept, retain
reviewed state, refine, or exhaust the identity. Any result other than accepted
remains a recorded fold refusal.

## Owner/geometry gate

All 16 identities below are currently absent, so their live `name_stage`,
`docs_stage`, `reviewer_score_name`, and `refine_attempts` are all null. The
owner/geometry staging instrument must first create each exact identity at
drafted stage from its signed DD-path authority. Only then is the recorded
name-scoped review invocation executable. This sequencing is intentional:
`sn run --name` cannot mint an absent identity, while reseeding the shared
`toroidal_angle_of_measurement_position` name would have a much broader and
unauthorized blast radius.

| Identity | Authority paths | Origin of gate |
|---|---:|---|
| `toroidal_coordinate_of_active_spatial_resolution_zone` | 2 | Vocabulary-unblocked replacement |
| `toroidal_coordinate_of_beam_tracing_point` | 1 | Absent replacement |
| `toroidal_coordinate_of_bragg_crystal` | 1 | Absent replacement |
| `toroidal_coordinate_of_camera` | 2 | Absent replacement |
| `toroidal_coordinate_of_detector` | 4 | Absent replacement |
| `toroidal_coordinate_of_neutron_detector` | 1 | Absent replacement |
| `toroidal_coordinate_of_pellet` | 1 | Absent replacement |
| `toroidal_coordinate_of_pellet_fragment` | 1 | Vocabulary-unblocked replacement |
| `toroidal_coordinate_of_pellet_path_point` | 1 | Absent replacement |
| `toroidal_coordinate_of_polarizer` | 2 | Vocabulary-unblocked replacement |
| `toroidal_coordinate_of_reciprocating_probe` | 1 | Absent replacement |
| `toroidal_coordinate_of_reflectometer_antenna` | 4 | Absent replacement |
| `toroidal_coordinate_of_shatter_cone` | 1 | Absent replacement |
| `toroidal_coordinate_of_shattering_position` | 1 | Absent replacement |
| `toroidal_coordinate_of_soft_xray_detector` | 1 | Absent replacement |
| `toroidal_coordinate_of_thomson_scattering_laser` | 1 | Absent replacement |

After exact drafted-stage creation, the scoped invocations are:

```text
uv run imas-codex sn run --name toroidal_coordinate_of_active_spatial_resolution_zone --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_beam_tracing_point --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_bragg_crystal --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_camera --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_detector --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_neutron_detector --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_pellet --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_pellet_fragment --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_pellet_path_point --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_polarizer --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_reciprocating_probe --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_reflectometer_antenna --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_shatter_cone --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_shattering_position --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_soft_xray_detector --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
uv run imas-codex sn run --name toroidal_coordinate_of_thomson_scattering_laser --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
```

The exact DD paths behind every identity are retained in the JSON. Examples
include four detector-centre paths for `toroidal_coordinate_of_detector`, four
reflectometer emission/detection antenna paths for
`toroidal_coordinate_of_reflectometer_antenna`, and the two polarizer-centre
paths for `toroidal_coordinate_of_polarizer`. No nearest-owner substitution is
introduced.

## Measured review-seat price

The current default name-review quorum is configured as three seats. Positive
live `LLMCost.llm_cost` observations for `pool=review_name` and
`phase=review_name` give:

| Current review seat | Calls | Mean/call | p95/call | Observed maximum |
|---|---:|---:|---:|---:|
| `openrouter/x-ai/grok-4.5` | 943 | $0.043238 | $0.073792 | $0.119404 |
| `openrouter/openai/gpt-5.6-luna` | 719 | $0.007068 | $0.028930 | $0.056614 |
| `openrouter/anthropic/claude-sonnet-5` | 594 | $0.073854 | $0.136568 | $0.190173 |
| **One three-seat quorum** | — | **$0.124160** | **$0.239290** | **$0.366191** |

Every identity in the JSON carries these measured per-identity prices. For all
19 identities:

| Cohort price basis | Total | Headroom below $150 |
|---|---:|---:|
| Sum of seat means | $2.359036 | $147.640964 |
| Sum of seat p95 values | $4.546510 | $145.453490 |
| Sum of observed seat maxima | $6.957629 | $143.042371 |

The observed-max envelope is **4.6% of the $150 authorization**. This node
spent **$0.00**. These figures price one ordinary name-review quorum per
identity; they deliberately do not disguise composition, refinement, retries,
or docs review as review-seat cost. Any of those later calls needs its own
admission and accounting.

## Write-free proof and exclusions

`StandardNameChange` measured the exact decimal byte string `7492` immediately
before and after the graph read. The values are byte-identical and the delta is
zero. The node made zero graph writes and zero provider calls.

The following are deliberately outside this cohort:

- six owner identities already accepted and valid;
- `toroidal_coordinate_of_field_map_grid`, a policy refusal rather than a
  review candidate;
- the two stale neutron-diagnostic sources, which require the separate
  stale-source lifecycle contract;
- the two complete-source/scalar fold conflicts;
- `electron_source_rate` and the two carried renames, which do not gate the
  ancestor or owner/geometry apply sets.
