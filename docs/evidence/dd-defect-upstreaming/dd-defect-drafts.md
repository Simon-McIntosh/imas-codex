# Data Dictionary defect issue and pull-request drafts

## Outcome

This package contains four complete issue drafts and four complete pull-request
drafts. Nothing was submitted. The duplicate search changes three of the four
submission dispositions:

| Defect | Held Standard Name | Duplicate disposition | Submission gate |
|---|---|---|---|
| Fast-wave cumulative power wording | `fast_ion_charge_state_power_at_inside_flux_surface` | **No existing issue or pull request found** | Eligible for individual review; do not submit without explicit approval |
| Sensor direction, X component | `x_direction_unit_vector_of_sensor` | **Found existing:** merged [PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242) | **Do not submit**; the correction is already on `develop` |
| Sensor direction, Z component | `z_direction_unit_vector_of_sensor` | **Found existing:** merged [PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242) | **Do not submit**; the correction is already on `develop` |
| Charge-state-resolved toroidal ion torque density | `toroidal_ion_charge_state_torque_density` | **Found existing:** the exact source structure was introduced by commit [`d049c043`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/d049c043592505408430549e03682c0f4b0e4dbc), carried into GitHub by merged [PR #9](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/9) | **Do not submit**; the exact path exists on `develop`, so the measured absence must be rechecked downstream |

The sensor X and Z reports cannot truthfully become independent live pull
requests: both components inherit their unit from one shared X/Y/Z template.
The existing upstream change correctly fixed the shared mechanism once. The two
separate pairs below are retained as content-complete archival drafts because
the requested package is one pair per measured identity; their duplicate gate
is binding.

The torque draft is also retained as a content-complete archival draft. Current
source inspection overturns the premise that no exact path exists. The source
already composes `ion/state` + `momentum` + `phi`, so submitting that draft now
would make a false claim about current `develop`.

## Upstream revision and inspection boundary

- Repository: [`iterorganization/IMAS-Data-Dictionary`](https://github.com/iterorganization/IMAS-Data-Dictionary)
- Remote branch inspected: `develop`
- Remote `develop` revision: [`d4c6345f3689a9bc905be527c061a0340a974c61`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/d4c6345f3689a9bc905be527c061a0340a974c61)
- Revision timestamp reported by GitHub: `2026-08-06T09:42:51Z`
- Revision subject: `Update two remaining "introduced_after" metadata tags`
- Verification method: read-only `git ls-remote`, GitHub issue-search GETs,
  GitHub pull-request GETs, commit-association GETs, and content GETs pinned to
  the exact revision. No local branch or remote ref was created or updated.

All live-source claims and all unsuppressed draft prose below were checked
against that revision. The sensor historical drafts also cite the pre-correction
base [`114e4cfccc2049e131b7cd78ec87539d4a6df792`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/114e4cfccc2049e131b7cd78ec87539d4a6df792),
where `direction` still referenced the metre-valued template. The existing
sensor correction was merged as commit
[`cb0d86de`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/cb0d86de388dbbdf62acca36de7b7f8c62bb9889).

## Exhaustive duplicate-search record

### Method and coverage

Every listed GitHub search used the repository qualifier and `in:title,body`.
The unqualified searches intentionally omitted `is:` and `state:` so that a
single query covered issues, pull requests, open records, and closed records.
For each defect, a second four-query matrix explicitly searched open issues,
closed issues, open pull requests, and closed pull requests. `count` is the
GitHub Search Issues API `total_count`, not the number displayed on one page.
No result was discarded; every non-zero hit is listed.

The torque path predates the GitHub migration and is not named in the body of
the migration pull request. A final read-only commit-to-pull-request association
query therefore checked the exact introducing commit. That query returned the
single carrying pull request and is included below.

Total duplicate-search operations: **37**. Unique existing contributions found:
**2 pull requests**. Search coverage: **4/4 defects**, each with all four
issue/PR and open/closed quadrants explicitly queried.

### Fast-wave cumulative power wording — 7 queries

| Query | Count | Hits |
|---|---:|---|
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "power_inside_fast"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "power_inside_thermal" "power_inside_fast"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "Absorbed wave power on thermal species"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:open "power_inside_fast"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:closed "power_inside_fast"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:open "power_inside_fast"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:closed "power_inside_fast"` | 0 | none |

Verdict: **no existing issue or pull request found**. The primary query can be
re-run in the [GitHub search UI](https://github.com/iterorganization/IMAS-Data-Dictionary/issues?q=%22power_inside_fast%22).

### Sensor direction X component — 8 queries

| Query | Count | Hits |
|---|---:|---|
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "operational_instrumentation/sensor/direction/x"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "direction/x" "unit vector"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "xyz0d_static_dimensionless"` | 1 | closed [PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242) |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "Direction of the measurement" "unit vector"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:open "xyz0d_static_dimensionless"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:closed "xyz0d_static_dimensionless"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:open "xyz0d_static_dimensionless"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:closed "xyz0d_static_dimensionless"` | 1 | closed [PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242) |

Verdict: **found existing**. PR #242 was merged on `2026-06-15`, targets
`develop`, changes `schemas/operational_instrumentation/dd_operational_instrumentation.xsd`
and the shared support schema, and explicitly lists `direction` among the
references moved to a unit-`1` template.

### Sensor direction Z component — 8 queries

| Query | Count | Hits |
|---|---:|---|
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "operational_instrumentation/sensor/direction/z"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "direction/z" "unit vector"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "xyz0d_static_dimensionless"` | 1 | closed [PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242) |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "Direction of the measurement" "unit vector"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:open "xyz0d_static_dimensionless"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:closed "xyz0d_static_dimensionless"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:open "xyz0d_static_dimensionless"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:closed "xyz0d_static_dimensionless"` | 1 | closed [PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242) |

Verdict: **found existing**. The same shared-template correction covers Z; a
second report would duplicate PR #242 and fragment one mechanism into two
maintainer discussions.

### Charge-state-resolved toroidal ion torque density — 14 queries

| Query | Count | Hits |
|---|---:|---|
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "toroidal_ion_charge_state_torque_density"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "plasma_sources/source/ggd/ion/state/momentum/phi"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "distributions/distribution/profiles_1d/collisions/ion/state/torque_thermal_phi"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "charge state" "torque density" "toroidal"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "plasma_sources_source_ggd_ion_state"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "Source terms related to the a given state of the ion species"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "Source term for momentum equations, on various grid subsets"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "charge state" "momentum" "plasma_sources"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body "Unify GGD types" "plasma_sources"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:open "plasma_sources_source_ggd_ion_state"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:issue state:closed "plasma_sources_source_ggd_ion_state"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:open "plasma_sources_source_ggd_ion_state"` | 0 | none |
| `repo:iterorganization/IMAS-Data-Dictionary in:title,body is:pr state:closed "plasma_sources_source_ggd_ion_state"` | 0 | none |
| `GET /repos/iterorganization/IMAS-Data-Dictionary/commits/d049c043/pulls` | 1 | merged [PR #9](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/9) |

Verdict: **found existing implementation**. The empty-body migration PR cannot
be discovered by path text, but its associated commit adds
`plasma_sources_source_ggd_ion_state/momentum`. The generic vector type supplies
the `phi` child. On current `develop`, the exact composed path is therefore
`plasma_sources/source/ggd/ion/state/momentum/phi`, with inherited unit
`kg.m^-1.s^-2` (equivalent to `N.m^-2`). The source links are:

- ion-state type and momentum member:
  [`dd_support.xsd` lines 7581–7687](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/d4c6345f3689a9bc905be527c061a0340a974c61/schemas/utilities/dd_support.xsd#L7581-L7687)
- toroidal `phi` component:
  [`dd_support.xsd` lines 5628–5642](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/d4c6345f3689a9bc905be527c061a0340a974c61/schemas/utilities/dd_support.xsd#L5628-L5642)

This is not merely a related near miss: the composition carries ion ownership,
state resolution, momentum-source semantics, and the toroidal projection on one
path. The downstream claim that the path is absent needs remeasurement against
the current DD before any upstream report.

## Draft pair 1 — fast-wave cumulative power wording

Disposition: **content complete; no duplicate found; awaiting explicit review
and authorization**. If an issue is created, replace `ISSUE_NUMBER` in the PR
body with the assigned number. That is the only mechanical substitution.

### Issue title

`waves ion-state power_inside_fast describes thermal rather than fast absorption`

<!-- BODY START fast issue -->
### Issue body

#### Summary

The Data Dictionary path
`waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` is named for the
fast population and is nested under an ion charge-state structure, but its
documentation says that the absorbed wave power is deposited on the thermal
population. A distinct `power_inside_thermal` sibling carries the same thermal
wording, so the recipient population of `power_inside_fast` is contradictory.

#### Current evidence

- Exact path: `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast`
- Source type: `waves_coherent_wave_profiles_1d_ion_state`
- Current prose: “Absorbed wave power on thermal species inside a flux surface
  (cumulative volume integral of the absorbed power density)”
- Declared unit: `W`
- State semantics: the containing structure identifies an ion charge-state
  bundle through `z_min`, `z_max`, and `name`.
- Thermal sibling: `waves/coherent_wave/profiles_1d/ion/state/power_inside_thermal`
- Thermal sibling prose: “Absorbed wave power on thermal species inside a flux
  surface (cumulative volume integral of the absorbed power density)”
- Fast density sibling prose: “Flux surface averaged absorbed wave power
  density on the fast species”

The relevant current source is
[`schemas/waves/dd_waves.xsd`](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/d4c6345f3689a9bc905be527c061a0340a974c61/schemas/waves/dd_waves.xsd#L800-L996).

#### Observed value semantics

This field is a cumulative volume integral inside each flux surface. Its value
is therefore power, not power density, and `W` is dimensionally correct. The
leaf name and the parallel `power_density_fast` field indicate that the
recipient is the fast particle population. The defect is the recipient word in
the prose, not the unit or the cumulative-integral semantics.

#### Proposed correction

Change the prose for this `power_inside_fast` member to:

> Absorbed wave power on the fast particle population inside a flux surface
> (cumulative volume integral of the absorbed power density)

Keep the declared unit `W`. Apply the same recipient-word correction to the
adjacent `power_inside_fast_n_phi` member so the scalar and per-toroidal-mode
descriptions remain consistent; keep its unit `W` as well.

#### Acceptance checks

1. The ion-state `power_inside_fast` prose says `fast`, not `thermal`.
2. `power_inside_thermal` retains its thermal-population wording.
3. Both fast cumulative-power fields retain unit `W`.
4. The generated dictionary validates and exposes the corrected prose at the
   exact path above.
<!-- BODY END fast issue -->

### Pull-request title

`docs(waves): correct fast cumulative-power recipient`

<!-- BODY START fast pr -->
### Pull-request body

Closes #ISSUE_NUMBER

#### Problem

`waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` is declared with
unit `W` and is structurally charge-state resolved, but its current prose is:

> Absorbed wave power on thermal species inside a flux surface (cumulative
> volume integral of the absorbed power density)

That wording duplicates the distinct `power_inside_thermal` sibling and
contradicts both the `power_inside_fast` leaf name and the neighboring
`power_density_fast` description.

#### Change

In `schemas/waves/dd_waves.xsd`, this change describes the ion-state
`power_inside_fast` value as absorbed wave power on the **fast particle
population** inside a flux surface. It makes the same recipient-word correction
to `power_inside_fast_n_phi` so the two members remain parallel.

The proposed scalar prose is:

> Absorbed wave power on the fast particle population inside a flux surface
> (cumulative volume integral of the absorbed power density)

The unit remains `W`; no type, coordinate, path, or dimensional metadata
changes.

#### Exact affected path

- `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast`
- Consistency sibling:
  `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast_n_phi`

#### Verification

- Confirmed the containing ion-state structure carries charge-state bundle
  metadata.
- Confirmed `power_inside_thermal` retains thermal-recipient prose and unit `W`.
- Confirmed `power_inside_fast` and its toroidal-mode sibling retain unit `W`.
- Validate the XSD and regenerated Data Dictionary before merge.
<!-- BODY END fast pr -->

## Draft pair 2 — sensor direction X component

Disposition: **content complete archival draft; found-existing; do not submit**.
Merged PR #242 already applied the proposed mechanism to `develop`.

### Issue title

`operational_instrumentation sensor direction x has a length unit despite unit-vector semantics`

<!-- BODY START sensor-x issue -->
### Issue body

#### Summary

At the pre-correction base, the path
`operational_instrumentation/sensor/direction/x` inherited unit `m` even though
its parent was explicitly documented as a unit vector. A Cartesian component of
a unit direction vector is a dimensionless direction cosine and must have unit
`1`.

#### Evidence at the affected revision

- Exact path: `operational_instrumentation/sensor/direction/x`
- Parent prose: “Direction of the measurement (unit vector)”
- Child prose: “Component along X axis”
- Declared child unit: `m`
- Expected unit: `1`
- Source cause: `direction` referenced the shared `xyz0d_static` position
  template, whose X, Y, and Z members all declared `m`.

The affected source is visible at the pre-correction base in
[`dd_operational_instrumentation.xsd` lines 113–117](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/114e4cfccc2049e131b7cd78ec87539d4a6df792/schemas/operational_instrumentation/dd_operational_instrumentation.xsd#L113-L117)
and
[`dd_support.xsd` lines 13995–14010](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/114e4cfccc2049e131b7cd78ec87539d4a6df792/schemas/utilities/dd_support.xsd#L13995-L14010).

#### Observed value semantics

The X value is a direction cosine contributing to a vector of magnitude one.
It is not an X position or displacement. Treating it as metres breaks
dimensional analysis and prevents binding to a dimensionless direction-vector
component identity.

#### Proposed correction

Move `operational_instrumentation/sensor/direction` to a dedicated
dimensionless Cartesian template whose X, Y, and Z components each declare unit
`1`. Preserve the parent prose and the child prose. Do not change genuine
position structures that correctly use the metre-valued template.

#### Current disposition

This correction is already present on `develop`: `direction` now references
`xyz0d_static_dimensionless`, and its X member declares unit `1`. See merged
[PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242).
<!-- BODY END sensor-x issue -->

### Pull-request title

`fix(operational-instrumentation): make sensor direction components dimensionless`

<!-- BODY START sensor-x pr -->
### Pull-request body

Closes #ISSUE_NUMBER

#### Problem

`operational_instrumentation/sensor/direction/x` was documented as “Component
along X axis” below the parent “Direction of the measurement (unit vector)” but
inherited declared unit `m` from `xyz0d_static`. The value is a direction cosine,
so its declared unit must be `1`.

#### Change

Introduce or reuse a Cartesian direction template whose X, Y, and Z members
declare unit `1`, then change
`schemas/operational_instrumentation/dd_operational_instrumentation.xsd` so
`sensor/direction` references that dimensionless template.

The exact reported path is
`operational_instrumentation/sensor/direction/x`. Its prose remains “Component
along X axis”; its declared unit changes from `m` to `1`. The parent prose
remains “Direction of the measurement (unit vector).” Genuine positions remain
on the metre-valued template.

#### Verification

- Confirm the generated exact path reports unit `1`.
- Confirm the X/Y/Z direction components all share dimensionless units.
- Confirm genuine position components still report `m`.
- Validate the XSD and regenerated Data Dictionary.

#### Duplicate gate

Do not open this pull request: merged PR #242 already implements this shared
template correction on `develop`.
<!-- BODY END sensor-x pr -->

## Draft pair 3 — sensor direction Z component

Disposition: **content complete archival draft; found-existing; do not submit**.
Merged PR #242 already applied the proposed mechanism to `develop`.

### Issue title

`operational_instrumentation sensor direction z has a length unit despite unit-vector semantics`

<!-- BODY START sensor-z issue -->
### Issue body

#### Summary

At the pre-correction base, the path
`operational_instrumentation/sensor/direction/z` inherited unit `m` even though
its parent was explicitly documented as a unit vector. A Cartesian component of
a unit direction vector is a dimensionless direction cosine and must have unit
`1`.

#### Evidence at the affected revision

- Exact path: `operational_instrumentation/sensor/direction/z`
- Parent prose: “Direction of the measurement (unit vector)”
- Child prose: “Component along Z axis”
- Declared child unit: `m`
- Expected unit: `1`
- Source cause: `direction` referenced the shared `xyz0d_static` position
  template, whose X, Y, and Z members all declared `m`.

The affected source is visible at the pre-correction base in
[`dd_operational_instrumentation.xsd` lines 113–117](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/114e4cfccc2049e131b7cd78ec87539d4a6df792/schemas/operational_instrumentation/dd_operational_instrumentation.xsd#L113-L117)
and
[`dd_support.xsd` lines 14024–14033](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/114e4cfccc2049e131b7cd78ec87539d4a6df792/schemas/utilities/dd_support.xsd#L14024-L14033).

#### Observed value semantics

The Z value is a direction cosine contributing to a vector of magnitude one.
It is not a height coordinate or displacement. Treating it as metres breaks
dimensional analysis and prevents binding to a dimensionless direction-vector
component identity.

#### Proposed correction

Move `operational_instrumentation/sensor/direction` to a dedicated
dimensionless Cartesian template whose X, Y, and Z components each declare unit
`1`. Preserve the parent prose and the child prose. Do not change genuine
position structures that correctly use the metre-valued template.

#### Current disposition

This correction is already present on `develop`: `direction` now references
`xyz0d_static_dimensionless`, and its Z member declares unit `1`. See merged
[PR #242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242).
<!-- BODY END sensor-z issue -->

### Pull-request title

`fix(operational-instrumentation): make sensor direction components dimensionless`

<!-- BODY START sensor-z pr -->
### Pull-request body

Closes #ISSUE_NUMBER

#### Problem

`operational_instrumentation/sensor/direction/z` was documented as “Component
along Z axis” below the parent “Direction of the measurement (unit vector)” but
inherited declared unit `m` from `xyz0d_static`. The value is a direction cosine,
so its declared unit must be `1`.

#### Change

Introduce or reuse a Cartesian direction template whose X, Y, and Z members
declare unit `1`, then change
`schemas/operational_instrumentation/dd_operational_instrumentation.xsd` so
`sensor/direction` references that dimensionless template.

The exact reported path is
`operational_instrumentation/sensor/direction/z`. Its prose remains “Component
along Z axis”; its declared unit changes from `m` to `1`. The parent prose
remains “Direction of the measurement (unit vector).” Genuine positions remain
on the metre-valued template.

#### Verification

- Confirm the generated exact path reports unit `1`.
- Confirm the X/Y/Z direction components all share dimensionless units.
- Confirm genuine position components still report `m`.
- Validate the XSD and regenerated Data Dictionary.

#### Duplicate gate

Do not open this pull request: merged PR #242 already implements this shared
template correction on `develop`.
<!-- BODY END sensor-z pr -->

## Draft pair 4 — charge-state-resolved toroidal ion torque density

Disposition: **content complete archival draft; found-existing implementation;
do not submit**. The exact proposed source path is already present on `develop`.

### Issue title

`plasma_sources lacks charge-state-resolved toroidal ion momentum source`

<!-- BODY START torque issue -->
### Issue body

#### Summary

A charge-state-resolved total toroidal ion torque-density quantity needs one
Data Dictionary path that carries ion ownership, charge-state resolution,
process-total source semantics, the toroidal component, and unit `N.m^-2`.

The two originally observed candidates were incomplete:

1. `plasma_sources/source/ggd/ion/momentum/phi` is ion-species resolved and
   process-total, but its ion-level prose says the momentum is the “sum over
   states when multiple states are considered.”
2. `distributions/distribution/profiles_1d/collisions/ion/state/torque_thermal_phi`
   is charge-state resolved, but its prose is “Collisional toroidal torque
   density to the thermal particle population,” restricting both mechanism and
   recipient.

Both candidates declare `kg.m^-1.s^-2` or the equivalent `N.m^-2`, but neither
near miss should be used as authority for a generic charge-state-resolved total.

#### Exact paths, prose, and units

- Species-total near miss:
  `plasma_sources/source/ggd/ion/momentum/phi`
  - parent prose: “Source term for momentum equations (sum over states when
    multiple states are considered), on various grid subsets”
  - leaf prose: “Toroidal component, one scalar value is provided per element
    in the grid subset.”
  - declared/inherited unit: `kg.m^-1.s^-2`
- Collisional-recipient near miss:
  `distributions/distribution/profiles_1d/collisions/ion/state/torque_thermal_phi`
  - prose: “Collisional toroidal torque density to the thermal particle
    population”
  - declared unit: `N.m^-2`
- Required exact path:
  `plasma_sources/source/ggd/ion/state/momentum/phi`
  - intended prose: “Source term for momentum equations, on various grid
    subsets” plus the toroidal-component prose above
  - intended inherited unit: `kg.m^-1.s^-2` (`N.m^-2`)

The distributions evidence is visible in
[`dd_distributions.xsd` lines 1942–2058](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/d4c6345f3689a9bc905be527c061a0340a974c61/schemas/distributions/dd_distributions.xsd#L1942-L2058).

#### Proposed correction

Add a `momentum` member of type `generic_grid_vector_components` to
`plasma_sources_source_ggd_ion_state`, with prose “Source term for momentum
equations, on various grid subsets,” coordinate `1...N`, and declared parent
unit `kg.m^-1.s^-2`. The generic vector supplies `phi` with toroidal-component
prose and unit `as_parent`, producing the exact path
`plasma_sources/source/ggd/ion/state/momentum/phi`.

#### Current disposition

Do not post this report. The proposed structure was already introduced by
commit `d049c043` and is present on current `develop`. The source now exposes
the exact path and unit described above. The missing-path result must be
remeasured in the consuming graph or generated DD instead of duplicated
upstream.
<!-- BODY END torque issue -->

### Pull-request title

`feat(plasma-sources): add charge-state GGD momentum components`

<!-- BODY START torque pr -->
### Pull-request body

Closes #ISSUE_NUMBER

#### Problem

The generic charge-state-resolved toroidal ion torque-density quantity has unit
`N.m^-2` (`kg.m^-1.s^-2`), but the observed alternatives lose essential
semantics:

- `plasma_sources/source/ggd/ion/momentum/phi` is total and toroidal but summed
  over ion states. Its parent prose explicitly says “sum over states when
  multiple states are considered.”
- `distributions/distribution/profiles_1d/collisions/ion/state/torque_thermal_phi`
  is charge-state resolved but says “Collisional toroidal torque density to the
  thermal particle population,” so it is not a process-total source.

#### Change

Add `momentum` to `plasma_sources_source_ggd_ion_state` using
`generic_grid_vector_components`.

- Exact resulting path:
  `plasma_sources/source/ggd/ion/state/momentum/phi`
- Momentum prose: “Source term for momentum equations, on various grid
  subsets”
- Toroidal leaf prose: “Toroidal component, one scalar value is provided per
  element in the grid subset.”
- Parent unit: `kg.m^-1.s^-2`
- Toroidal leaf unit: `as_parent`, resolving to `kg.m^-1.s^-2` (`N.m^-2`)

This keeps total plasma-source semantics while adding the ion-state ownership
needed for charge-state resolution. It does not alter the narrower
collisional-to-thermal fields in `distributions`.

#### Verification

- Generate the Data Dictionary and confirm the exact path exists.
- Confirm `phi` inherits `kg.m^-1.s^-2` from its `momentum` parent.
- Confirm the ion-level `momentum/phi` path remains available as the sum over
  states.
- Validate the XSD and generated dictionary.

#### Duplicate gate

Do not open this pull request: commit `d049c043`, carried by merged PR #9,
already implements this structure on `develop`.
<!-- BODY END torque pr -->

## Verification and zero-write attestation

Quantitative package checks:

- Defects covered: **4/4**.
- Exhaustive duplicate-search operations recorded: **37/37**.
- Explicit issue/PR and open/closed search quadrants: **16/16**.
- Complete issue bodies: **4**.
- Complete pull-request bodies: **4**.
- Draft bodies containing the exact DD path, quoted prose, declared unit, and
  proposed correction: **8/8**.
- Existing upstream contributions found: **2 unique pull requests**.
- Defect dispositions: **1 no-existing + 3 found-existing = 4**.
- Self-attribution scan: a case-insensitive grep for `co-authored-by`,
  `generated with`, `claude`, and `copilot`, restricted to every line between
  the eight `BODY START` / `BODY END` markers, reports **0 hits**.
- Issues created: **0**.
- Pull requests created: **0**.
- Branches created locally: **0**.
- Branches pushed: **0**.
- Commits or tags pushed: **0**.
- Remote comments, reviews, labels, reactions, edits, or other state changes:
  **0**.
- Network writes that mutate remote state: **0**. All network access used GET
  or read-only Git reference discovery for the searches and revision-pinned
  source inspection recorded above.

The string `ISSUE_NUMBER` remains in each PR draft because GitHub assigns an
issue number only after issue creation. Replacing that token with the assigned
number is the sole mechanical step needed to activate `Closes #…`; creating a
number during this node would violate the zero-submission boundary. For the
three found-existing pairs, the duplicate gate prohibits even that step.
