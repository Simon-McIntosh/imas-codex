# The successor of a superseded name survives without the `superseded_by` scalar

**Verdict: the scalar is derivable-around for all but three names.** Of the 2,163
**1,641** have an incoming `REFINED_FROM` successor edge, **517** have no successor
by any route at all, and of those 517 the scalar is the *only* pointer for
**three** — so removing the scalar costs three pointers, not 2,163.

Measured read-only against the live graph on 2026-09-17. No write, no pipeline run.

## The question

The plan asks whether the `superseded_by` scalar can be removed. This record
measures the prerequisite: for each of the 2,163 names in the superseded state,
is a successor reachable *without consulting the scalar*, and does the recorded
baseline of 2,145 null scalars and 517 successor-less names still hold.

## Population and controls

| Measure | Value | Note |
|---|---:|---|
| `name_stage = 'superseded'` names | 2,163 | reproduces the recorded population |
| null `superseded_by` | 2,145 | reproduces the recorded baseline exactly |
| non-null `superseded_by` | 18 | exceeds zero — the instrument sees scalars |
| `successor` scalar set on a superseded name | 0 | the paired scalar is vacant here |
| `REFINED_FROM` edges in the graph | 1,850 | exceeds zero — the edge instrument sees edges |
| `HAS_SUCCESSOR` edges in the graph | 48 | derived from the two scalars, not from lineage |

## Route B — an incoming `REFINED_FROM` edge

The lineage relation carries the successor for the population: **1,641 of 2,163**
names have at least one node pointing at them through `REFINED_FROM`, and 522
have none. Of the 1,641, 867 have a successor that is itself live, and 684 have
an `accepted` successor specifically.

This is the route the plan's derive should read. It is a cardinality-many edge
where the scalar is cardinality-one, which is the finding, not a defect: 1,641
names that the scalar never described are described by the edge.

## Route A — the change ledger does not name a successor

The ledger holds the supersession *event* and not a successor identity. Of the
303 superseded names carrying a `supersede*` row with a non-null `to_name`,
**300 name themselves and only 4 name another identity** (three distinct
targets). The instrument is not blind: the graph holds 15,548
`StandardNameChange` rows, 10,704 `HAS_INTERNAL_CHANGE` edges, 324 `supersede`
rows every one of which carries a `to_name`, and 15,534 rows carry a
`from_name`. The operation producing the self-naming rows is
`supersede_exhausted_orphan` (296 rows), whose `to_name` is the exhausted
identity itself.

**A derive that reads "the ledger row says where this went" therefore reads the
wrong self-reference in 300 of 303 cases.** The `graph_ops.py:3193` edge writer
already avoids this half-way: `HAS_SUCCESSOR` derives from the `successor` or
`superseded_by` scalar and from nothing else, while the lineage walk used by the
fold reads `REFINED_FROM` (`edit.py:2085`). So the derived edge inherits the
scalar's gaps and the lineage carries the population.

## What the scalar is still load-bearing for: five names, three of them uniquely

The 521 names with no `REFINED_FROM` successor and the recorded 517 with no
successor by any route differ by five, and those five are the whole of the
scalar's remaining authority over the successor relation:

| Identity | Scalar successor | Derived edge |
|---|---|---|
| `flux_due_to_thermal_fusion` | `total_neutron_source_rate_due_to_thermal_fusion` | no |
| `area_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | no |
| `lower_energy` | `lower_bound_energy_of_neutron_detector` | no |
| `radial_effective_total_ion_energy_convection_velocity` | none | yes |
| `radial_effective_electron_energy_convection_velocity` | none | yes |

**Three names lose their only recorded pointer if the scalar is dropped without
migration.** Those three are the migration workitem; the 2,145 null scalars are
not, because the edge route already carries 1,641 of the population and nothing
carries the other 517.

## Every source location that reads `superseded_by`

The captured lexical log is the package-wide walk over `imas_codex/**` and
`tests/**` retained at the run's `scratch/readers.txt`. Classified, the
executable sites are:

| Site | Kind | What it does |
|---|---|---|
| `imas_codex/standard_names/graph_ops.py:3193` | read | `successor = n.get("successor") or n.get("superseded_by")` — materialises the `HAS_SUCCESSOR` edge the derived route reads |
| `imas_codex/standard_names/signed_manifest.py:4090` | read | selects `node.superseded_by` into the manifest payload |
| `imas_codex/standard_names/edit.py:2104` | read | consults it only to **refuse** a fold whose lineage disagrees |
| `imas_codex/standard_names/catalog_import.py:332` | write | sets it from a catalog entry that declares one |
| `imas_codex/standard_names/canonical.py:35,69` | list | export/compare field membership |
| `imas_codex/standard_names/protection.py:34` | list | protected-field membership |
| `imas_codex/schemas/standard_name.yaml:925,1996` | declaration | the field and its `HAS_SUCCESSOR` annotation |

Nothing outside those files reads it in the package. The three read sites are
migratable individually: the refuse-only consultation in `edit.py:2104` needs
the lineage walk it already computes from `REFINED_FROM`, the writer at
`graph_ops.py:3193` would derive from lineage only, and the strength of the
signed-manifest read is limited to the 18 names it can currently see.

## What this does not establish

- It does not size the migration of the three uniquely-scalar names — their
  successors exist as nodes, and whether the edge should be written or the
  pointer recorded is a design question this record does not settle.
- It does not measure what a reader loses by following the scalar, only where
  the successor can be found without it. The scalar-vs-edge disagreement is
  recorded separately at
  [[stage-scalar-agreement]].
- It is a single-time read; no write of any kind was attempted here.

## Queries and where they ran

All counted on the live graph through `GraphClient` (login-node tunnel). The
probe scripts and their JSON outputs are retained in the run's `scratch/`
directory (`probes.json` through `probes5.json`, `census1.json` through
`census4.json`). Each zero in this record is printed beside a control returning non-zero: the
scalar census (18 non-null) shows the scalar instrument sees scalars; the
lineage instrument's 1,641 edge-carrying names show `REFINED_FROM` is populated
rather than uniformly absent; and 15,548 ledger rows behind 10,704
`HAS_INTERNAL_CHANGE` edges show the ledger join is not empty, so the 300
self-naming rows are real self-naming and not a broken join.
