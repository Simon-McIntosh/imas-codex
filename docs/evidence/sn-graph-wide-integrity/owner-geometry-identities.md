# Owner-geometry identity composition and review evidence

## Outcome

The sanctioned source-scoped composition and ordinary name-review route closed
**10 of the 16** committed owner-geometry identity gates. All 10 accepted
identities were minted by the configured local
`hosted_vllm/deepseek-v4-flash` compose seat, entered review as `drafted` and
`valid`, and reached `accepted` from one fresh RD quorum with
`resolution_method=quorum_consensus`. Each accepted identity has exactly two
fresh `StandardNameReview` rows in one review group. No refine pool ran.

The node is blocked rather than complete because the same one-pass composition
left six identities without an executable ordinary-review gate:

- three were minted at `drafted` but quarantined by
  `canonical_locus_check`: `toroidal_coordinate_of_beam_tracing_point`,
  `toroidal_coordinate_of_pellet_path_point`, and
  `toroidal_coordinate_of_shattering_position`;
- three were not minted: `toroidal_coordinate_of_pellet_fragment`,
  `toroidal_coordinate_of_reflectometer_antenna`, and
  `toroidal_coordinate_of_shatter_cone`.

None of these six was resubmitted to composition, revalidated, renamed,
refined, or given a second quorum draw. The beam-tracing identity appeared in
two review command scopes because a terminated sequential launcher had already
started its zero-work command when the combined bounded review began. Both
claim filters returned no eligible work, created zero Reviews and zero
`LLMCost` rows, and did not change its stage. This is a duplicated no-op scope,
not a repeated review draw.

## Authority and staging

The committed cohort was read from
`docs/evidence/sn-graph-wide-integrity/review-cohort-manifest.json`; its exact
byte digest was
`e9631f6b41cd35885eec1fc0b3b508c20aba3bb4273fb1ced3bb73ed669a4c85`.
It contained exactly 16 owner-geometry identities over 25 unique DD paths.
The fresh baseline found all 16 identities absent, all 25 sources `attached`
and unclaimed, and the local compose endpoint healthy.

The staging invocation was:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env python /home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T170716004861-sgwi-owner-geometry-identities/stage_cohort.py
```

It used the canonical `detach_one_attachment` and `set_source_compose_hint`
instruments, with a dry-run gate before every write. All **25/25** old-target
detaches and **25/25** exact-source hints were admitted. The pellet-path source
also carried the exhausted, quarantined
`toroidal_angle_of_along_pellet_path` predecessor after its generic binding was
removed. Its release used the canonical hash-bound terminal-attachment
operator: one-row manifest SHA-256
`0a4907ea5ba3b0055619e32cb7bba29abb25d6db2f6af8294a38a402ecc4c24c`,
dry-run refusal count zero, apply mode `applied`.

The shared `toroidal_angle_of_measurement_position` identity was **not
reseeded, reset, reviewed, renamed, or otherwise restaged**. It remained
`accepted` with reviewer score `0.95625`. Its live producer count was 50 at the
fresh baseline, 25 immediately after the 25 source detaches, and 32 after the
composer independently reattached seven paths to it. It therefore retained a
live producer throughout.

## Exact compose and review invocations

The one compose pass was:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env imas-codex sn run --focus "mse/channel/active_spatial_resolution/centre/phi spectrometer_visible/channel/active_spatial_resolution/centre/phi ece/channel/beam_tracing/beam/position/phi spectrometer_x_ray_crystal/channel/crystal/centre/phi camera_x_rays/camera/centre/phi spectrometer_x_ray_crystal/channel/camera/centre/phi bolometer/camera/channel/detector/centre/phi mse/channel/detector/centre/phi spectrometer_uv/channel/detector/centre/phi spectrometer_visible/channel/detector/centre/phi neutron_diagnostic/detector/geometry/centre/phi spi/injector/pellet/position/phi spi/injector/fragment/position/phi pellets/time_slice/pellet/path_geometry/first_point/phi ece/polarizer/centre/phi spectrometer_visible/channel/polarizer/centre/phi langmuir_probes/reciprocating/plunge/position_average/phi reflectometer_fluctuation/channel/antenna_detection_static/centre/phi reflectometer_fluctuation/channel/antenna_emission_static/centre/phi reflectometer_profile/channel/antenna_detection/centre/phi reflectometer_profile/channel/antenna_emission/centre/phi spi/injector/shatter_cone/origin/phi spi/injector/shattering_position/phi soft_x_rays/channel/detector/centre/phi thomson_scattering/laser/end_point/phi" --only compose --names-only --skip-global-maintenance --cost-limit 1.0
```

All 19 local compose calls finished at `$0.00`. The run was interrupted after
300.6 seconds only after it had been idle with one unclaimable source for more
than two minutes; the retained run record is
`41604ed4-ee2d-4171-a9cd-2b24617125ec`, `names_composed=16`,
`stop_reason=interrupted`. No second compose pass was run.

The first accepted identity used its manifest-recorded exact review command:

```text
uv run imas-codex sn run --name toroidal_coordinate_of_active_spatial_resolution_zone --only review_name --names-only --skip-global-maintenance --cost-limit 1.0
```

The remaining 12 minted identities were then submitted together through the
same exact-name, single-pool gate; only the nine valid identities were claimed:

```text
uv run imas-codex sn run --name toroidal_coordinate_of_beam_tracing_point --name toroidal_coordinate_of_bragg_crystal --name toroidal_coordinate_of_camera --name toroidal_coordinate_of_detector --name toroidal_coordinate_of_neutron_detector --name toroidal_coordinate_of_pellet --name toroidal_coordinate_of_pellet_path_point --name toroidal_coordinate_of_polarizer --name toroidal_coordinate_of_reciprocating_probe --name toroidal_coordinate_of_shattering_position --name toroidal_coordinate_of_soft_xray_detector --name toroidal_coordinate_of_thomson_scattering_laser --only review_name --names-only --skip-global-maintenance --cost-limit 12.0 --time 10
```

The three absent identities had no executable `--name` invocation: the exact
name-scope preflight cannot mint an absent identity. No command was substituted
for that fail-closed decision.

## Per-identity results and attributable spend

Costs are the exact `LLMCost.llm_cost / size(sn_ids)` apportionment for rows
written since the node began. Local compose rows are present but cost zero.
The score is null when no fresh quorum was executable.

| Identity | Final name stage | Validation | Fresh quorum score | Attributable USD | Result |
|---|---|---|---:|---:|---|
| `toroidal_coordinate_of_active_spatial_resolution_zone` | `accepted` | `valid` | 0.99375 | 0.061333 | one fresh quorum, accepted |
| `toroidal_coordinate_of_beam_tracing_point` | `drafted` | `quarantined` | null | 0.000000 | review claim refused by validation gate; no draw |
| `toroidal_coordinate_of_bragg_crystal` | `accepted` | `valid` | 1.00000 | 0.052713 | one fresh quorum, accepted |
| `toroidal_coordinate_of_camera` | `accepted` | `valid` | 1.00000 | 0.052990 | one fresh quorum, accepted |
| `toroidal_coordinate_of_detector` | `accepted` | `valid` | 1.00000 | 0.057891 | one fresh quorum, accepted |
| `toroidal_coordinate_of_neutron_detector` | `accepted` | `valid` | 0.99375 | 0.055319 | one fresh quorum, accepted |
| `toroidal_coordinate_of_pellet` | `accepted` | `valid` | 0.99375 | 0.054988 | one fresh quorum, accepted |
| `toroidal_coordinate_of_pellet_fragment` | absent | absent | null | 0.000000 | composer attached source to `toroidal_angle`; no retry |
| `toroidal_coordinate_of_pellet_path_point` | `drafted` | `quarantined` | null | 0.000000 | review claim refused by validation gate; no draw |
| `toroidal_coordinate_of_polarizer` | `accepted` | `valid` | 1.00000 | 0.053679 | one fresh quorum, accepted |
| `toroidal_coordinate_of_reciprocating_probe` | `accepted` | `valid` | 1.00000 | 0.060983 | one fresh quorum, accepted |
| `toroidal_coordinate_of_reflectometer_antenna` | absent | absent | null | 0.000000 | composer reattached all four sources to `toroidal_angle_of_measurement_position`; no retry |
| `toroidal_coordinate_of_shatter_cone` | absent | absent | null | 0.000000 | source left extracted with its open hint and interrupted-run claim; no retry |
| `toroidal_coordinate_of_shattering_position` | `drafted` | `quarantined` | null | 0.000000 | review claim refused by validation gate; no draw |
| `toroidal_coordinate_of_soft_xray_detector` | `accepted` | `valid` | 1.00000 | 0.054016 | one fresh quorum, accepted |
| `toroidal_coordinate_of_thomson_scattering_laser` | `accepted` | `valid` | 1.00000 | 0.026950 | one fresh quorum, accepted |
| **Total** | **10 accepted / 3 drafted / 3 absent** | **10 valid / 3 quarantined / 3 absent** | — | **0.530862** | **10 fresh quorums; 6 no-draw refusals** |

The running coordinator session had spent `$0.857000` before this node. Adding
this node's exact `$0.530862` gives **`$1.387862 / $150.000000`**, leaving
**`$148.612138`** authorized headroom.

## Non-accepted identities

The three validation refusals are all the same current audit decision:

- `toroidal_coordinate_of_beam_tracing_point` is told to use
  `toroidal_coordinate_at_beam_tracing_point`;
- `toroidal_coordinate_of_pellet_path_point` is told to use
  `toroidal_coordinate_at_pellet_path_point`;
- `toroidal_coordinate_of_shattering_position` is told to use
  `toroidal_coordinate_at_shattering_position`.

That `of` versus `at` conflict is not resolved here: the committed authority
names the `of` identities, while the current canonical-locus audit classifies
the point-like owner tokens as evaluation loci. Re-adjudicating the authority
or the audit is ordinary semantic review outside this worker's evidence-only
write fence.

The three missing identities retain the composer outcomes exactly as observed.
The pellet-fragment source is attached to reviewed `toroidal_angle`; the four
reflectometer antenna sources are attached to accepted
`toroidal_angle_of_measurement_position`; and the shatter-cone source remains
`extracted`, hint `open`, with the interrupted run's claim timestamp. No source
was forced onto the desired identity after the model decision.

## Integrity assertions

- **Acceptance path:** PASS for all 10 accepted identities. Every one has one
  fresh review group, two fresh Review rows, `resolution_method=quorum_consensus`,
  and a score at or above the configured threshold. No identity reached
  `accepted` by direct stage mutation, structural inheritance, import, catalog
  edit, or maintenance promotion.
- **No direct graph text edit:** PASS. No raw Cypher `SET`, hand-written graph
  text mutation, `sn edit --rename`, direct acceptance, or direct documentation
  edit was used. Graph writes were performed only by sanctioned detach,
  hash-bound terminal recovery, exact-source hint, compose, and review
  instruments.
- **No acceptance retry:** PASS. Each accepted identity received exactly one
  quorum draw. The six non-accepted identities received no quorum draw and no
  compose retry. The beam-tracing duplicate command scope was a zero-work
  overlap and created no review or cost row.
- **Shared-name preservation:** PASS. `toroidal_angle_of_measurement_position`
  stayed accepted at score 0.95625 and retained 32 live producers after the
  run; it was never reseeded.
- **Ledger:** `StandardNameChange` 7,650 to 7,676 (`+26`: 25 semantic detaches
  plus one terminal recovery); `LLMCost` node count 27,489 to 27,528 (`+39`),
  exact node spend `$0.530862`.

## Durable artifacts

All runtime evidence is under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T170716004861-sgwi-owner-geometry-identities/`:

- `baseline.json` and `baseline.log` — fresh pre-mutation endpoint, graph and
  counter census;
- `staging-receipt.json`, `stage_cohort.log`, and
  `pellet-path-terminal-recovery-manifest.json` — exact 25-path staging and
  hash-bound terminal release;
- `compose.log` — the complete local compose run;
- `review-toroidal_coordinate_of_active_spatial_resolution_zone.log` and
  `review-remaining-combined.log` — complete ordinary review runs;
- `current-census.json` and `final-census.log` — final stages, scores, review
  provenance, costs, counters, sources and run invocations.

The next action is a separate authority decision for the three `of`/`at`
quarantines plus a sanctioned new composition attempt, if authorized, for the
three identities the one-pass local composer did not mint. This node does not
make either decision.
