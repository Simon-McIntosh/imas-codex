# Three WEST sources without an identity

Measured 2026-09-15 against the live Neo4j graph through the login-node-local
tunnel. The cohort is exactly the three requested DD paths. Graph reads were
bounded to those rows and to narrowly filtered representative-name searches;
the only write used the governed `imas-codex sn attach` command.

## Outcome

One source gained an accepted, documentation-accepted, validation-valid
identity. Two remain excluded for measured, source-specific reasons:

| DD path | Before | Final disposition |
|---|---|---|
| `calorimetry/group/component/energy_total/data` | `extracted`, attempts `0`, no scalar identity, zero `PRODUCED_NAME` edges | Excluded. Its prior target is validation-valid but `name_stage=exhausted` at score `0.75`; the lifecycle reconcile deliberately removed the terminal binding and the attachment guard refuses to restore it. |
| `camera_x_rays/camera/camera_dimensions` | `failed`, attempts `5`, no scalar identity, zero `PRODUCED_NAME` edges | Excluded. The source is at the compose-attempt cap with `last_error='compose claim-attempt cap reached'` and `skip_reason_detail='position:camera_dimensions'`; no accepted compatible representative exists, and the closest accepted detector-extent name is rejected by the attachment guard as a different device locus. |
| `spectrometer_visible/channel/isotope_ratios/signal_to_noise` | `skipped`, attempts `2`, no scalar identity, zero `PRODUCED_NAME` edges | Recovered. Attached to accepted `spectral_signal_to_noise_ratio_of_spectrometer_channel`; postflight reads `source_status=attached`, one `PRODUCED_NAME` edge, one DD projection, a matching scalar/cache entry, and accepted/accepted/valid target state. |

The exact three-row release projection after the governed attachment is:

| DD path | Source status | Resolved identity | Terminal stage | Projection reason |
|---|---|---|---|---|
| `calorimetry/group/component/energy_total/data` | `extracted` | none | none | The generic projection reports the retained stale skip text `vocab_gap_nonactionable: :`; the lifecycle diagnosis below is the current measured reason. |
| `camera_x_rays/camera/camera_dimensions` | `failed` | none | none | `compose claim-attempt cap reached` |
| `spectrometer_visible/channel/isotope_ratios/signal_to_noise` | `attached` | `spectral_signal_to_noise_ratio_of_spectrometer_channel` | `accepted` | none |

## The calorimetry edge was dropped, not the identity deleted

The source/name pair existed immediately before this node. The preceding
grammar-invalid census recorded
`accumulated_total_coolant_absorbed_energy_of_calorimetry_component` with the
live producer `dd:calorimetry/group/component/energy_total/data`. The current
read reproduces the changed state:

- source: `status=extracted`, `attempt_count=0`, `produced_sn_id=null`, and zero
  `PRODUCED_NAME` edges;
- name: still present, `name_stage=exhausted`, `docs_stage=pending`,
  `validation_status=valid`, `reviewer_score_name=0.75`, `status=draft`, and
  zero producers;
- the name was created at `2026-09-15T00:31:04.801Z` and its deterministic
  validation observation remains stamped at `2026-09-15T06:27:09.573Z`.

The operation that removed the edge is named by the complete CLI log at
`/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`. At
`2026-09-15 14:46:41` it records:

> `reconcile_source_status_liveness: realigned 0 source(s) with a live target; returned 1 source(s) without one to extracted (1 stale edge(s), 1 projection(s), and 1 source-path entry(s) dropped)`

This cardinality is attributable to the calorimetry pair: the other two rows
in this exact cohort had never acquired identities, while the immediately
preceding census recorded this pair's edge. The code contract agrees with the
receipt: `reconcile_source_status_liveness()` defines `exhausted` as terminal,
returns a bound source with no non-terminal target to `extracted`, resets its
attempt budget to zero, and removes the `PRODUCED_NAME`, DD projection, scalar
mirror, and cached source path together.

The old name is not eligible for reattachment. The guarded dry run refuses:

> `'accumulated_total_coolant_absorbed_energy_of_calorimetry_component' is at name_stage 'exhausted' and may not acquire a source; only accepted, approved, drafted, reviewed can hold a binding`

The closest accepted identity is not interchangeable.
`accumulated_coolant_absorbed_energy_of_calorimetry_component` is accepted and
valid, name score `0.9875`, and is produced by
`dd:calorimetry/group/component/energy_cumulated`; its description says the
quantity accumulates since pulse start. The requested `energy_total/data` leaf
is the whole-discharge scalar, including the terminal total after the pulse.
Binding it to the cumulative array would collapse two DD quantities that the
earlier review explicitly distinguished. The source therefore stays excluded
until a new spelling earns acceptance; restoring its exhausted spelling would
contradict both the lifecycle contract and the guard refusal.

## The camera dimension has no accepted compatible representative

The exact DD source is a `FLT_1D` quantity in metres with documentation
`Total camera dimension in each direction (x1, x2)`. Its enriched description
identifies the two overall detector-array dimensions used for geometry and
field-of-view characterization. The current terminal source state reproduces
the earlier failure exactly:

```text
status=failed
attempt_count=5
last_error=compose claim-attempt cap reached
skip_reason=vocab_gap
skip_reason_detail=position:camera_dimensions
```

The absence check was instrumented. The graph holds `5,130` `StandardName`
nodes and all `5,130` carry `id`; exact candidate lookup found
`extent_of_camera` only at `superseded`, `size_of_camera` and
`total_size_of_camera` only at `exhausted`, and no
`total_extent_of_camera`, `dimensions_of_camera`, or `camera_dimensions` node.

The nearest accepted name is `extent_of_soft_xray_detector`, whose description
is “Full end-to-end physical size of a soft X-ray detector along the
applicable horizontal or vertical detector-plane direction.” It is accepted,
documentation-accepted and validation-valid, but its hardware locus is not the
generic X-ray camera in this source. The guarded dry-run attachment makes that
distinction executable and refuses the pairing:

> `locus/source device mismatch: SN 'extent_of_soft_xray_detector' has hardware locus 'soft_xray_detector' but path 'camera_x_rays/camera/camera_dimensions' shares no device token with it`

No relevant grammar or composition change has landed since the five-attempt
failure: the intervening classifier repair addressed operator bare-prefix
classification, not the stored `position:camera_dimensions` gap. A sixth blind
attempt would spend a governed retry without new information, contrary to the
recorded report-only decision. This source is therefore excluded at the
attempt-cap/vocabulary boundary, with the mechanically tempting existing name
independently refused by the attachment guard.

## The visible-spectrometer ratio is now bound

The source describes a log-scale spectral signal/reference-band power ratio,
unit `dB`. It started `skipped` at two attempts after composition collided with
the superseded generic identity
`signal_to_noise_ratio_of_spectrometer_channel`. The current representative is
the more specific
`spectral_signal_to_noise_ratio_of_spectrometer_channel`, whose description
states the signal-wavelength-interval versus line-free-reference-interval
comparison for the same spectrometer channel. It was already
`name_stage=accepted`, `docs_stage=accepted`, `validation_status=valid`, with
documentation score `0.9125` and a structural derived source.

The dry run first proved that the attachment guard accepted the exact pairing:

> `would attach spectrometer_visible/channel/isotope_ratios/signal_to_noise to spectral_signal_to_noise_ratio_of_spectrometer_channel`

The live command then used the same arguments and recorded
`StandardNameChange sn-change:3395bdb8-1219-49b5-9099-3c7f78657633`, operation
`attach_unbound_standard_name_source`, origin `attachment_judgement`, at
`2026-09-15T16:09:02.479930Z`. Its reason records that the DD leaf and target
are the same channel-local spectral signal/reference power ratio and that the
generic collision is historical.

Postflight verifies the markers introduced by that write rather than only the
absence of an error:

```text
source_status=attached
produced_sn_id=spectral_signal_to_noise_ratio_of_spectrometer_channel
produced_edges=1
dd_projections=1
source_path_cached=true
name_stage=accepted
docs_stage=accepted
validation_status=valid
```

## Commands, spend, and mutation boundary

Read-only evidence came from bounded Cypher through `mcp__imas_cx__repl` and
from `fetch_manifest_source_release_rows()` over exactly these three paths.
The only graph mutation was:

```text
uv run --no-sync imas-codex sn attach spectrometer_visible/channel/isotope_ratios/signal_to_noise spectral_signal_to_noise_ratio_of_spectrometer_channel --reason <recorded-physics-judgement>
```

The two negative guard probes added `--dry-run` and wrote nothing. No `sn run`
recomposition was invoked, so the node spent **$0.00** and the conditional
10 USD/time-limit recomposition fence was not entered. No ad-hoc graph mutation
was used to attach or restore any edge.
