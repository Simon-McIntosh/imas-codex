# Compose probe of the four outstanding sources

`provisional: false` — complete.

The four sources the settled-exclusion declaration records as **not waived** were
returned to the composition queue and recomposed on the free local seat, to
establish whether each one is a name the catalog is missing or a source that
genuinely cannot carry one. Spend: **$0.00** (`local/deepseek-v4.1-flash`,
direct endpoint, proxy bypassed).

## The four were not in one state

The dry run reported only that two of four were eligible for `sn retry --skipped`,
refusing the other two with a message that is an OR of four conditions and so
names none of them. Their actual states differ in kind:

| DD source path | status | refusal recorded | attempts |
|---|---|---|---|
| `summary/disruption/time/value` | `skipped` | `non_nameable_coordinate:time` — *bare non-nameable token: time* | 2 |
| `equilibrium/time_slice/contour_tree/node/z` | `skipped` | `compose_model_skipped` | 2 |
| `camera_x_rays/camera/camera_dimensions` | `failed` | `vocab_gap` → `position:camera_dimensions`, claim-attempt cap reached | 5 |
| `calorimetry/group/component/energy_total/data` | `extracted` | `vocab_gap_nonactionable`, detail is the empty string `':'` | **0** |

Two corrections to the prior reading follow from this. The camera row is not an
attribution defect: it exhausted the claim-attempt cap. The calorimetry row was
never composed at all — zero attempts — and its refusal detail is a bare
separator with nothing either side of it, so whatever gap was recorded, the
record does not say what it was.

`sn retry --skipped` returned the first two and `sn retry --failed` the third;
the fourth needed no retry because `extracted` is already a composable state.

## `camera_x_rays/camera/camera_dimensions` — a livelock, not a vocabulary gap

The composer produces a name for this source on every attempt, and the same one:
**`total_size_of_camera`**. Each attempt is then released:

```
WARNING - Released source dd:camera_x_rays/camera/camera_dimensions after
lifecycle_collision for candidate total_size_of_camera at name stage exhausted
```

That identity already exists, and it belongs to a **different diagnostic**:

| field | value |
|---|---|
| `name_stage` | `exhausted` |
| `validation_status` | `quarantined` |
| `superseded_by` | *(null)* |
| unit | `m` |
| description | Overall horizontal and vertical physical size of the X-ray **crystal spectrometer** camera body. |
| producer | `dd:spectrometer_x_ray_crystal/channel/camera/camera_dimensions` |

So two distinct DD sources — an X-ray camera and a crystal spectrometer's camera
— compose to one spelling, because the spelling does not say which camera it
measures. The existing identity is already quarantined, and nothing supersedes
it, so the name is neither usable nor released.

**The loop is the finding.** The composer cannot see the collision, so it
re-proposes the same colliding spelling on every attempt until the claim-attempt
cap stops it. That is what the earlier five attempts were, and what this probe
reproduced from attempt 1 through attempt 3 under observation. The recorded
`vocab_gap` on `position:camera_dimensions` describes the symptom rather than the
cause: there is no vocabulary gap — there is a name that is not self-descriptive
enough to distinguish two diagnostics, and a retired identity holding it.

The disposition this implies is a self-descriptive spelling for each camera
rather than a waiver, and a decision on the quarantined identity, which has no
successor.

## `calorimetry/group/component/energy_total/data` — a name the catalog was missing

This source composed on its **first** attempt:

```
total_energy_of_calorimetry_component   name_stage=drafted  validation=pending
```

It had never been composed before — `attempt_count` was zero and its status was
`extracted`, which is a composable state, not a refusal. Nothing was blocking it.
The `vocab_gap_nonactionable` on its record is a stale artifact of an earlier
pass, and its detail is the separator with both operands missing.

**That stale string is already published.** It appears in a cut review manifest
as the stated reason this source carries no name:

```
imas_codex/standard_names/manifests/reviews/v0.4.0rc7+west-task-2e.sn_names.yaml:338
  non_nameable_reason: 'vocab_gap_nonactionable: :'
```

So a reviewer opening that manifest is told a source is unnameable because of a
vocabulary gap that is not named, when in fact the source is nameable and was
simply never attempted. **A successful compose does not clear the previous
`skip_reason` / `skip_reason_detail`**, so the record still carries both the
stale refusal and, now, a drafted name.

## The two that remain nameless, and why the reason moved

| path | before | after |
|---|---|---|
| `summary/disruption/time/value` | `non_nameable_coordinate:time` | `compose_model_skipped` |
| `equilibrium/time_slice/contour_tree/node/z` | `compose_model_skipped` | `compose_model_skipped` |

`summary/disruption/time/value` no longer trips the bare-token gate. Previously
the composer proposed the bare token `time` — taken from the DD documentation
*"Time base for disruption."* — and `is_non_nameable_coordinate` suppressed it,
correctly, since it fires only on bare tokens. The composer now declines to
propose anything at all. **The gate was never the defect here**, and the source
is still nameless; what changed is that the refusal is now the composer's own
judgement rather than a downstream suppression of a bad proposal.

This matters for the disposition. A `compose_model_skipped` is not evidence that
a quantity is non-nameable — it is evidence that one model declined once. The
lead's ruling stands that this row is a measured event time of the same shape as
`summary/global_quantities/ip/value` and wants a name; the probe did not produce
one, and steering (`sn source-hint`) rather than waiver is the route.

## Answer to the question the probe was run to settle

**Yes — the batch is missing at least one name it should contain.**
`calorimetry/group/component/energy_total/data` is nameable, composes cleanly,
and was published as non-nameable with a malformed reason. Of the four:

| path | probe outcome | disposition |
|---|---|---|
| `calorimetry/.../energy_total/data` | **named** `total_energy_of_calorimetry_component` | publish; clear the stale skip record |
| `camera_x_rays/camera/camera_dimensions` | livelock on a collision with a quarantined identity | needs self-descriptive spellings for both cameras |
| `summary/disruption/time/value` | composer declined | steer; not a waiver |
| `equilibrium/.../contour_tree/node/z` | composer declined | open |

Cost: **$0.00**. Nine LLM calls, zero failures, free local seat.

## Incidental measurement: graph service re-resolution

The run spawned **5,017 `squeue` subprocesses in 4 m 18 s**, peaking at 41 in a
single second — one per `GraphClient()` construction, each re-resolving the same
service node and logging the same two lines. The resolution is correct and
cached nowhere. This is a load defect on the SLURM controller rather than a
correctness one, and it is proportional to pipeline throughput.

## What this record does not establish

No claim is made here about the embedding server. An earlier reading in this
session diagnosed a compute-node discovery defect; that diagnosis was **wrong**
and is withdrawn — `ensure_embedding_ready` returns ready on a compute node and
resolves `localhost` to the serving host correctly. The one failed run had hit a
server idle for 83.9 hours against a 5-second health check.

Whether the WEST cut now reports `generable True` is a separate live-graph
measurement over that cohort and is not made here.
