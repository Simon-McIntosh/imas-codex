<meta name="docs-project" content="imas-codex">
<meta name="plan-slug" content="sn-grammar-refinement">
<meta name="plan-status" content="active">

# Re-measuring this plan's own headline figures

**Plan:** imas-codex `sn-grammar-refinement`
**Base:** `5ea0d73b` (worktree head at dispatch, clean)
**Graph:** live Neo4j, measured 2026-09-17 from the repository login node
**Grammar package:** `imas_standard_names` 0.9.3 (installed)
**Report-only.** Every graph statement was a `MATCH … RETURN`; no `SET`, `MERGE`,
`CREATE`, or `DELETE` ran. Every figure below is either a Cypher read or an
in-process read of the installed grammar, and every run is captured to a log.

## Why this measurement exists

The plan's headline figures were written before the plan's own migration and
closing work. A sprint rescoped on them would be sized against a graph that no
longer exists. The figures are re-measured here so the rescope rests on the
current graph.

## Method

`live` means a `StandardName` node whose `name_stage` is anything other than
`superseded` — the graph holds both the retired and the current identity of every
migrated name, so counting all nodes double-counts the migration.

Three instruments, each run once in a fresh process:

| Instrument | Log | What it reads |
|---|---|---|
| `sgr_measure.py` | `/tmp/sgr_measure.log` (`/tmp/sgr_measure_out.json`) | census, `HAS_PARENT`, operator census, non-canonical names |
| `sgr_measure2.py` | `/tmp/sgr_measure2.log` (`/tmp/sgr_measure2_out.json`) | parse failures, `with_respect_to` spellings, postfix spellings |
| `sgr_measure3.py` | `/tmp/sgr_measure3.log` (`/tmp/sgr_measure3_out.json`) | per-stage operator splits, `HAS_COMPONENT` population |
| `sgr_probe5.py` | `/tmp/sgr_probe5.log` | component tokens by count, live and retired |
| `sgr_probe6.py` | `/tmp/sgr_probe6.log` | component-set completeness per base; the failure identities |

Exit status was 0 for all five runs.

Two readings are load-bearing and easy to get wrong:

- **A name is non-canonical when `compose(parse(name).ir) != name`.** That is the
  test that decides whether the renderer leaves the stored spelling alone.
- **`ParseResult` exposes the intermediate representation at `.ir`.** Calling
  `compose()` on the `ParseResult` itself raises, and a first pass that did so
  counted six real parse failures plus fifteen composition errors as one
  undifferentiated bucket. The counts below come from the corrected pass; the
  first pass's parse list is discarded rather than quoted.

## Asserted against re-measured

| # | § | Asserted in the plan (verbatim) | Re-measured on the live graph | Verdict |
|---|---|---|---|---|
| 1 | 1 | "2,689 live names exist" | **2,967** live names (5,130 nodes in total, 2,163 of them superseded) | drifted |
| 2 | 1 | "253 carry the defects below, of which 225 are already publishable" | **304** live names carry a grammar operator; **2,528** live names are `accepted` | drifted |
| 3 | 2 | "87 postfix-operator names (74 publishable)" | **89** postfix-operator live names, **79** accepted | drifted |
| 4 | 2 | "149 prefix-operator names (138 publishable)" | **192** prefix-operator live names, **162** accepted | drifted |
| 5 | 3 | "Measured surface: 13 names — 9 `derivative_with_respect_to_*` and 4 `time_derivative_of_*" | **12** live names carry a `with_respect_to` clause (8 accepted, 3 exhausted, 1 reviewed; 21 more retired). The retired order is absent from the live cohort: **0** live names end in `accumulated` | stale |
| 6 | 4 | "1,598 of 2,689 live names carry `HAS_PARENT`" | **1,537** of 2,967 live names (1,441 accepted) | drifted |
| 7 | 4 | "Of 505 component-prefixed stems … 176 are true vector components" | **486** names carry a component token, **265** of them live, over **16** distinct component tokens | drifted |
| 8 | 8 | "the same census found **7 pre-existing parse failures**" | **6** live names fail to strict-parse | drifted |
| 9 | 8 | closing state: "**8 refused**" (census: 2,352 stable + 0 migrating + 8 refused = 2,360) | **8** accepted live names parse, compose, and render away from their stored spelling | current |

**Verdict counts: 1 current, 7 drifted, 1 stale — 9 rows.**

### What each verdict rests on

**Rows 1, 6, 7 — drift in the cohort.** The live cohort grew from 2,689 to 2,967
while the plan ran. `HAS_PARENT` fell in absolute share (1,598 → 1,537) because
newly admitted names are children without being parents. Both figures are the
same instrument at two moments, and the plan's own count was correct when
written.

**Rows 2, 3, 4 — drift in the operator surface.** The operator counts are read by parsing
every live name in-process rather than by string matching: `unary_prefix` covers
163 names, `unary_postfix` 62, `binary` 50, 27 carry both postfix and prefix, and
2 carry binary plus prefix. The plan's §2 counts are subsets by segment family and
grew with the cohort. These are the figures the sprint's remaining renderer work
should be sized from, not §2's.

**Row 5 — stale, and this is the informative verdict.** The plan's §3 surface was
13 names spelled with the index on the operator. That spelling is gone from the
live cohort: the live `with_respect_to` names are the canonical
`derivative_of_<operand>_with_respect_to_<coordinate>` form (10 live names start
`derivative_of`), and the retired order survives only as 21 superseded identities.
A figure that names a defect that no longer exists is *stale*, not drifted — the
work has landed and the number should be retired from any rescope rather than
resized.

**Row 8 — one fewer failure than the plan recorded.** Six, not seven. The six are:
`flux_surface_average_magnetic_field_magnitude`,
`inertial_current_density_due_to_diamagnetic_drift`,
`normalized_perpendicular_gyroaveraged_perturbed_energy`,
`per_toroidal_mode_flux_surface_average_total_absorbed_power_density`,
`tendency_of_runaway_electron_density`, and `toroidal_angle_of_along_pellet_path`.

**Row 9 — current, and worth keeping.** The plan's closing census classified its
2,360-name cohort as 2,352 stable and 8 refused. Eight accepted live names today
parse and compose to a different spelling. These are the same eight identities the
plan refused, still unrefused:

| Refused identity | Renders as |
|---|---|
| `boron_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_boron_density_at_plasma_boundary` |
| `coolant_absorbed_energy_accumulated_of_plasma_facing_component` | `absorbed_energy_accumulated_of_plasma_facing_component_coolant` |
| `krypton_density_..._at_plasma_boundary` | as above, krypton |
| `linear_thermal_electron_decay_time_volume_averaged_due_to_disruption` | `thermal_electron_decay_time_linear_...` |
| `neutral_count_accumulated_at_wall` | `count_accumulated_neutral_at_wall` |
| `oxygen_count_accumulated_due_to_gas_injection` | `count_accumulated_oxygen_due_to_gas_injection` |
| `radiated_energy_accumulated_due_to_impurity_radiation` | `energy_accumulated_radiated_due_to_impurity_radiation` |
| `tungsten_density_..._at_plasma_boundary` | as boron, tungsten |

The agreement is the check that the instrument reads the same surface the plan's
census did: an independent count, on a graph holding 2,163 additional retired
identities and 607 additional live ones, reproduces the plan's refusal count
exactly.

## The first unstarted beat

The plan's §5 orders six beats. Beats 1–4 have landed evidence in the plan's own
record: the shadow diff (350 identities, 350 collision-free), the indexed-operator
demonstrator (grammar `12b5573`), and the tail-gated renames. **Beat 5's schema half
is present** — 486 names carry a component link to a component grammar token, 265
of them live — and its sidecar half was retired by the decision that the name
carries its own decomposition.

**Beat 6, sibling completion across the component families, is the first unstarted
beat. Verdict: still-required.** The measurement that decides it:

> Of the **148 live bases** that carry at least one of the four cardinal component
> tokens, **2 carry all four** and **146 are partial** — missing between one and
> three components.

Representative partials: `magnetic_field` carries poloidal, radial and toroidal but
not vertical; `momentum_flux` carries poloidal, radial and toroidal; `ion_momentum_flux`
carries poloidal only; `halo_current` and `convection_velocity` carry poloidal only;
`back_surface_curvature_of_optical_element` carries vertical only.

**This is an upper bound on the work, not a worklist.** Some of those bases
legitimately have a single component — a toroidal rotation quantity has no poloidal
sibling. Deciding which of the 146 are genuinely incomplete needs the DD geometry
carrier enumeration that the definition of the family already implies, and that
enumeration has not been run. What the figure does establish is that the family
axis is legible in the schema (the completeness query above is expressible today)
and that no completion pass has been applied against it: 98.6% of the cardinal
component bases are partial.

## Remaining effort

Restated in worker-hours against the plan's own 16.0 declared hours:

| Work | Basis | Worker-hours |
|---|---|---|
| Component-family adjudication and completion (beat 6) | 146 partial bases to adjudicate against the DD geometry carriers; composing the genuinely missing members | 10–14 |
| The 8 accepted non-canonical identities | compose a canonical spelling or record a refusal with the reason the plan left open | 2–3 |
| The 6 parse failures | each needs a base or a carrier the grammar does not yet admit | 4–6 |
| **Total** | | **16–23** |

The paid review lane sits on top of that total and is not included: every minted
name reaches a reviewer before it is publishable, so the wall-clock and the spend
of beat 6 are governed by the review lane rather than by the composing.

## Bounds observed

- No graph mutation. Every statement was `MATCH … RETURN`.
- Every graph read was a named, indexed read over `StandardName` or its immediate
  edges, and the largest single cohort read returned 2,967 rows.
- The two reads that iterate the whole live cohort (the operator census and the
  non-canonical check) run in-process over rows already fetched, with no
  per-row query.
- Reported as unmeasured rather than estimated: the plan's §5 beat 1 roster of
  "all 431 roster names" — the roster is a plan-side list with no graph-side
  identity, so its 431 could not be checked from the plan's text alone.