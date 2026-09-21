# Resolving the already-present repair targets to their live ends

provisional: false — all 15 targets carry a class, the four classes sum to 15, and the result table is closed.

[The ordinal revision](west-name-repair-worklist-ordinal-revision) found that
**15 of the 61 proposed repair targets already exist** in the graph — 7 at
`name_stage = 'accepted'` and 8 at `superseded` — and reported that fact
without resolving it. This record resolves it.

**The governing rule, from the project lead: a superseded name is historical
record and is never a target.** No repair binds onto a superseded identity.
Where a proposed target is superseded, the repair follows that identity's
lineage to its live end and binds there; where the lineage has no live end,
the repair mints a fresh identity rather than reviving a retired one.

So each of the 15 falls into one of four classes:

| class | condition |
| --- | --- |
| **ALREADY-LIVE** | the target is `accepted`; nothing to resolve |
| **REDIRECT** | the target is `superseded` and its lineage has exactly one live end; the repair binds there |
| **ADJUDICATE** | the lineage has more than one live end, so a lookup cannot choose |
| **MINT** | the lineage has no live end; the repair mints a fresh spelling |

Neither the worklist nor the revision record is amended. Both are read-only
inputs and this is a third linked record. **No graph write of any kind was
issued** — every statement below is a read.

## The instrument, and why the obvious traversal would have been wrong

Resolving a retired name to its live end looks like a question about where the
name *went*, and the graph cannot answer that question. Three counts measured
on this graph say so: of **2163** superseded names only **18** carry a
`superseded_by` scalar, the whole graph holds **48** `HAS_SUCCESSOR` edges and
**3** `SUPERSEDES` edges, while `REFINED_FROM` holds **1850**. Following the
forward-looking fields would therefore have returned nothing for almost every
target, and *nothing* would have read as "no live end — MINT". Every REDIRECT
below would have been a false MINT.

The lineage lives on `REFINED_FROM`, and it is written by the successor, so it
points **backwards**. The live end is found by asking which name was refined
*from* this one — never by asking where this one went.

### Confirming the direction before trusting it

One statement, printing both ends of five `REFINED_FROM` edges with the
`name_stage` of each:

```cypher
MATCH (a:StandardName)-[:REFINED_FROM]->(b:StandardName)
RETURN a.id, a.name_stage, b.id, b.name_stage LIMIT 5
```

The fourth row is the unambiguous one:

| start | start stage | end | end stage |
| --- | --- | --- | --- |
| `total_fast_particle_count` | **accepted** | `fast_count` | **superseded** |

The accepted name is at the **start** of the arrow and the retired one at its
**end**, so the edge reads "`total_fast_particle_count` was refined from
`fast_count`". A traversal that followed the arrow from a retired name would
walk further into history.

The aggregate over the whole edge type confirms it rather than resting on one
sample. Of the 1850 edges, the end is `superseded` in **1657** (834 from a
superseded start, 687 from an accepted one, 136 from an exhausted one) while
the end is `accepted` in only **95**. Accepted appears at the start of an edge
**seven times more often** than at its end. The direction is not ambiguous.

The 71 edges running `superseded → accepted` are the reason each traversal
below collects *every* reachable accepted name rather than stopping at the
first: a lineage can pass through an accepted node that was later retired, and
a first-hit traversal would stop there.

### Controls on the read itself

A name known to exist returns present — `electron_temperature`, `accepted` —
and a fabricated name, `zzz_not_a_standard_name_control`, returns an empty
result. So an empty lineage below is a measurement and not a silent failure.

### The traversal

For each target, one bounded reverse traversal of at most six hops, anchored on
the named identity:

```cypher
MATCH p = (succ:StandardName)-[:REFINED_FROM*1..6]->(t:StandardName {id: $n})
RETURN succ.id, succ.name_stage, length(p) AS hops ORDER BY hops, id
```

Every statement is an indexed read anchored on a named identity or on the fixed
`REFINED_FROM` type. **21 statements in total, 0.137 s for all of them** — no
whole-label scan, no unbounded traversal, no cartesian product.

## The eight superseded targets

### 1. `ipb98y2_confinement_enhancement_factor` — **REDIRECT**, and it is a round trip

- repair: worklist rename row 11, `energy_confinement_enhancement_factor` → `ipb98y2_confinement_enhancement_factor`
- target stage: `superseded` (`status` superseded, `superseded_by` null — the scalar the instrument section says is almost always absent)
- reverse lineage: 1 reachable name, 1 accepted
- **live end: `energy_confinement_enhancement_factor`, accepted, at hop 1**
- **This is the spelling the repair rejects.** The graph records that `ipb98y2_confinement_enhancement_factor` was refined *into* `energy_confinement_enhancement_factor`; the repair proposes to rename back the other way. Applying it would reverse a decision the pipeline already took, and resolving the target through lineage returns the name the repair started from. The repair is a round trip and cannot be applied as written.

### 2. `hot_neutral_temperature` — **REDIRECT**, and it is a round trip

- repair: worklist rename row 17, `hot_neutral_temperature_at_plasma_boundary` → `hot_neutral_temperature`
- target stage: `superseded`
- reverse lineage: 2 reachable names, 1 accepted
- chain: `hot_neutral_temperature` ← `hot_neutral_temperature_at_plasma_edge` (superseded, hop 1) ← `hot_neutral_temperature_at_plasma_boundary` (**accepted**, hop 2)
- **live end: `hot_neutral_temperature_at_plasma_boundary`, accepted, at hop 2**
- Again the rejected spelling. The intermediate hop is informative: the pipeline went bare → `_at_plasma_edge` → `_at_plasma_boundary`, so the locus qualifier the repair wants to strip was added deliberately and then refined. Two hops is weaker evidence than one, but the chain is linear with no branch.

### 3. `line_integrated_electron_density` — **REDIRECT**, and it is a round trip at five hops

- repair: worklist rename row 18, `line_integrated_electron_number_density` → `line_integrated_electron_density`
- target stage: `superseded`
- reverse lineage: 5 reachable names, 1 accepted
- chain: `line_integrated_electron_density` ← `..._of_interferometer_beam` (superseded) ← `..._of_line_of_sight` (superseded) ← `..._at_line_of_sight` (superseded) ← `accumulated_electron_number_density_at_line_of_sight` (superseded) ← `line_integrated_electron_number_density` (**accepted**, hop 5)
- **live end: `line_integrated_electron_number_density`, accepted, at hop 5**
- **The weakest evidence of the eight and the sharpest finding.** Five hops is a long chain and a distant live end is weaker evidence than an adjacent one — but the chain is linear, every intermediate is superseded, and the single accepted name at its far end is exactly the spelling the repair rejects. Two independent audit halves proposed this rename on the ground that the cohort spells the base `electron_density` in eight names and `electron_number_density` in one; the graph says the pipeline tried `electron_density` first, refined through four spellings, and landed on `electron_number_density`. The rename is not new information to the pipeline, and re-applying it walks the chain backwards.

### 4. `radial_outline_of_plasma_facing_component` — **MINT**

- repair: worklist rename row 29, `radial_outline_of_wall` → `radial_outline_of_plasma_facing_component`
- target stage: `superseded`
- reverse lineage: **0 reachable names at any hop up to six**
- No live end exists, so under the governing rule the repair mints a fresh identity rather than reviving this one. The empty result is a measurement, not a failed read: the same statement returns 5 rows for target 3 and 1 row for target 1, and the fabricated-name control returns empty for a name that genuinely does not exist.

### 5. `thermal_electron_pressure` — **REDIRECT**, and it is a round trip

- repair: worklist rename row 38, `thermal_electron_pressure_at_post_sawtooth_crash` → `thermal_electron_pressure`
- target stage: `superseded`
- reverse lineage: 1 reachable name, 1 accepted
- **live end: `thermal_electron_pressure_at_post_sawtooth_crash`, accepted, at hop 1**
- The rejected spelling again, adjacent this time. The bare `thermal_electron_pressure` was refined into the crash-qualified spelling, so the repair proposes stripping a qualifier the pipeline added.

### 6. `toroidal_coordinate_of_detector_pixel` — **REDIRECT**, and it is a round trip

- repair: worklist rename row 42, `toroidal_coordinate_at_detector_pixel` → `toroidal_coordinate_of_detector_pixel`
- target stage: `superseded`
- reverse lineage: 1 reachable name, 1 accepted
- **live end: `toroidal_coordinate_at_detector_pixel`, accepted, at hop 1**
- The rejected spelling. This pair differs by one preposition, `at` against `of`, so the round trip here is a preposition the pipeline already chose once.

### 7. `radial_coordinate_of_aperture` — **REDIRECT**

- repair: worklist split row 55, one of the three resolutions of `radial_coordinate_of_measurement_position`
- target stage: `superseded`
- reverse lineage: 1 reachable name, 1 accepted
- **live end: `radial_coordinate_of_diagnostic_aperture`, accepted, at hop 1**
- **This one is a genuine redirect and not a round trip**: the live end is a third spelling, neither the rejected name nor the proposed one. The split's aperture binding therefore lands on `radial_coordinate_of_diagnostic_aperture` and the repair is applicable once its target string is rewritten.
- **Independently reproduced.** A peer reported this same resolution; the traversal above was run from the target rather than from the peer's answer, and it agrees exactly, including the hop distance of 1.

### 8. `radial_coordinate_of_toroidal_magnetic_field_probe` — **MINT**

- repair: worklist split row 55, the probe-position resolution of `radial_coordinate_of_measurement_position`
- target stage: `superseded`
- reverse lineage: **0 reachable names at any hop up to six**
- No live end, so the repair mints. The spelling is retired and nothing was ever refined from it.
- **Independently reproduced**, and it agrees with the peer's report of nothing at all. The agreement is worth stating precisely because an empty result is the answer that a wrong instrument would also give: the forward-looking `superseded_by` scalar is null here, as it is on all eight, so a forward traversal would have returned this same emptiness for every one of the six REDIRECTs above.

## The seven accepted targets

All seven are **ALREADY-LIVE** and need no resolution: each was read
individually and each returned `name_stage = 'accepted'`, confirming the
revision record's lookup. Each was also put through the same six-hop reverse
traversal, and **all seven returned zero reachable successors** — which is the
expected shape for a live end and is a second, independent confirmation that
each is the current identity rather than a way-station. A name that had since
been refined into something else would have shown a successor here even while
its own stage still read accepted.

| # | target | own stage | reverse-reachable successors | class |
| --- | --- | --- | --- | --- |
| 9 | `vertical_coordinate_of_measurement_position` | accepted | 0 | **ALREADY-LIVE** |
| 10 | `vertical_coordinate_of_aperture` | accepted | 0 | **ALREADY-LIVE** |
| 11 | `volume_of_plasma_boundary` | accepted | 0 | **ALREADY-LIVE** |
| 12 | `toroidal_angle_of_toroidal_magnetic_field_probe` | accepted | 0 | **ALREADY-LIVE** |
| 13 | `toroidal_angle_of_poloidal_magnetic_field_probe` | accepted | 0 | **ALREADY-LIVE** |
| 14 | `volume_averaged_effective_charge` | accepted | 0 | **ALREADY-LIVE** |
| 15 | `vertical_coordinate_of_x_point` | accepted | 0 | **ALREADY-LIVE** |

All seven carry `status = 'draft'` against `name_stage = 'accepted'`. That is
the catalog lifecycle field rather than the pipeline verdict and the two are
independent — the first audit record established that `status` holds only
`draft` and `superseded` across the whole graph, so `draft` here is the normal
state of an accepted, unreleased name and is not a contradiction.

## The three toroidal loci

The revision record's split row 56 resolves the three bindings of
`toroidal_angle_of_measurement_position`, and one of the audit halves had
proposed a different spelling for the ECE binding. The graph was asked about
all three spellings directly. It answers one of them outright and leaves the
other two open, and the difference matters because only the first is a rename
onto an identity that exists.

### `toroidal_coordinate_of_measurement_position` — **the graph answers it**

- **`toroidal_coordinate_of_measurement_position` is `superseded`** in the
  graph, and its reverse lineage holds exactly one reachable name:
  **`toroidal_angle_of_measurement_position`, accepted, at hop 1.**
- So the audit row that proposed renaming `ece/channel/position/phi` from
  `toroidal_angle_of_measurement_position` onto
  `toroidal_coordinate_of_measurement_position` **is a rename onto a retired
  identity**, and following that identity's lineage to its live end returns the
  very name the proposal rejects. This is the same round-trip shape as the five
  above.
- **The base question is therefore answered by the graph rather than by
  adjudication**: `toroidal_angle_of_measurement_position` is the live spelling
  of this base, the `toroidal_coordinate` spelling was tried and retired, and
  the revision record's decision to leave the ECE binding on
  `toroidal_angle_of_measurement_position` is the one the graph already
  records. No adjudication is required and none should be scheduled.
- **Independently reproduced.** A peer reported this resolution; the traversal
  was run from the target rather than from the peer's answer and agrees
  exactly, at the same hop distance of 1.
- **One discrepancy with the revision record, and it is in the record's favour.**
  This spelling is *not* among the revision record's 15, because that record
  resolved split row 56's ECE binding as a `keeps` and so never looked the
  spelling up. It is a sixteenth already-present target of the *worklist's*
  proposals, though not of the *revised* 59 — the revision had already dropped
  the proposal, and the graph now independently confirms that drop was right.
  The 15 is correct for the revised repair set; the record simply did not know
  that the proposal it dropped pointed at a retired identity.

### `toroidal_coordinate_of_toroidal_magnetic_field_probe` — **the graph leaves it open**

- The identity **does not exist** in the graph at any stage: the single-row read
  returns empty, against the two controls above.
- So there is nothing to redirect and nothing retired to avoid. The spelling is
  free to mint, and the choice between `toroidal_coordinate_of_*` and some other
  form for a probe *position* is a naming decision the graph does not make.
- It does contribute one fact: `toroidal_angle_of_toroidal_magnetic_field_probe`
  is **accepted** (target 12) and denotes the probe's sensing *orientation*, a
  physically different quantity. So the position spelling must differ from the
  orientation spelling, and minting `toroidal_coordinate_of_*` collides with
  nothing.

### `toroidal_coordinate_of_poloidal_magnetic_field_probe` — **the graph leaves it open**

- Also **does not exist** at any stage, and the same reasoning applies against
  the accepted `toroidal_angle_of_poloidal_magnetic_field_probe` (target 13).
- Both of these are ordinary mints. Neither is blocked and neither is settled.

## The lineages

![Reverse REFINED_FROM lineage of each superseded repair target: the target at hop zero, intermediate superseded names along the chain, and the accepted live end with its hop distance, with the five round trips marked](/imas-codex/figures/sn-west-catalog-release/superseded-target-lineage.png)

The figure carries the one thing a table cannot: **hop distance is evidence
strength.** Four of the six redirects land at hop 1, where the target and its
live end are adjacent and the resolution is as strong as the edge itself. One
lands at hop 2 and one at hop 5, and the five-hop chain is the weakest claim
in this record — it is trusted because the chain is linear with no branch and
every intermediate is superseded, not because five hops is close.

## Result

| class | count | targets |
| --- | --- | --- |
| **ALREADY-LIVE** | **7** | the seven accepted targets, each with zero reverse-reachable successors |
| **REDIRECT** | **6** | `ipb98y2_confinement_enhancement_factor`, `hot_neutral_temperature`, `line_integrated_electron_density`, `thermal_electron_pressure`, `toroidal_coordinate_of_detector_pixel`, `radial_coordinate_of_aperture` |
| **ADJUDICATE** | **0** | no target reached more than one live end |
| **MINT** | **2** | `radial_outline_of_plasma_facing_component`, `radial_coordinate_of_toroidal_magnetic_field_probe` |
| **the four classes, summed** | **15** | reconciling against the 15 already-present targets |

| | count |
| --- | --- |
| revised repairs in the ordinal revision | **59** |
| **repairs whose target string can no longer be applied literally** | **7** |
| target strings involved in those 7 repairs | **8** — split row 55 carries two of them |
| of those 8, resolving to a live end that is the repair's own rejected spelling | **5** |
| of those 8, resolving to a genuinely different live end | **1** — `radial_coordinate_of_aperture` → `radial_coordinate_of_diagnostic_aperture` |
| of those 8, resolving to no live end and therefore minting | **2** |
| live ends found at hop 1 | 4 of 6 |
| live ends found at hop 2 or beyond | 2 of 6 — one at hop 2, one at hop 5 |
| graph statements issued | 21, all reads, **0.137 s** in total |
| graph writes issued | **0** |

**The seven repairs whose target cannot be applied literally** are rename rows
11, 17, 18, 29, 38 and 42 of the worklist, plus split row 55, which carries two
superseded targets on its own. Every other one of the 59 either targets an
accepted identity or mints an absent one, and needs nothing from this record.

### Five of the six redirects are round trips, and that is the finding

Only one of the six redirects behaves the way a redirect is supposed to. The
other five resolve to the exact spelling their own repair rejects:

| repair | rejects | proposes | its target's live end |
| --- | --- | --- | --- |
| row 11 | `energy_confinement_enhancement_factor` | `ipb98y2_confinement_enhancement_factor` | **`energy_confinement_enhancement_factor`** |
| row 17 | `hot_neutral_temperature_at_plasma_boundary` | `hot_neutral_temperature` | **`hot_neutral_temperature_at_plasma_boundary`** |
| row 18 | `line_integrated_electron_number_density` | `line_integrated_electron_density` | **`line_integrated_electron_number_density`** |
| row 38 | `thermal_electron_pressure_at_post_sawtooth_crash` | `thermal_electron_pressure` | **`thermal_electron_pressure_at_post_sawtooth_crash`** |
| row 42 | `toroidal_coordinate_at_detector_pixel` | `toroidal_coordinate_of_detector_pixel` | **`toroidal_coordinate_at_detector_pixel`** |

Resolving the target under the governing rule hands each of these repairs back
its own starting point, so **each is a no-op that reverses a decision the
pipeline already recorded.** Whether the audits were right and the pipeline
wrong is a question about the names, and this record does not answer it — it
establishes that these five cannot be executed as renames, because the
identity they would rename onto is retired and the live end of that identity
is the name they would rename away from.

**The right reading is not that the audits were wrong.** An audit that rejects
a spelling without seeing that the catalog already tried and retired the
alternative is making a real argument against the current name; what it cannot
see is that the argument has been had. The remedy for these five is an
adjudication between the audit's reasoning and the pipeline's recorded
decision — which is a different piece of work from applying a rename, and it
belongs to whoever owns the repair schedule.

**The one genuine redirect is worth its own line**, because it is what all six
were assumed to be: `radial_coordinate_of_aperture` is retired and its live end
is the third spelling `radial_coordinate_of_diagnostic_aperture`, which neither
the worklist's rejected name nor its proposed name mentions. Split row 55's
aperture binding lands there and the repair remains applicable once its target
string is rewritten.

### What the instrument choice was worth

Had this record used the forward-looking fields — `superseded_by`,
`HAS_SUCCESSOR`, `SUPERSEDES` — all eight superseded targets would have
returned empty, because `superseded_by` is null on all eight and the three
forward mechanisms together hold 69 edges against `REFINED_FROM`'s 1850. The
record would have read **0 REDIRECT, 8 MINT**, and eight repairs would have
been sent to mint spellings the catalog can in six cases already resolve. The
two genuine MINTs would have been right for the wrong reason, which is the
part that makes the error hard to catch: a wrong instrument agreed with the
correct answer on exactly the two rows where the correct answer was *nothing*.
