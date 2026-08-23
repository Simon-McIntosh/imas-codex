# Deprecation stub lineage audit

## Verdict

**The export implementation contradicts the locked `internal-record-only`
decision, although the contradiction is dormant in the production graph
today.** A release at the measured graph state would emit **0 deprecation
stubs**, of which **0 would carry a successor**, because **0 superseded names
have `catalog_approved_at` set.** That zero is not evidence that the
implementation honors the decision: when its approval and successor
preconditions eventually become true, every emitted stub is constructed with
three public successor references.

The semantic authority says that successors are never released and that
supersede lineage cannot appear in a release
(`docs/plans/sn-release-readiness.html:209-215`,
`docs/plans/sn-release-readiness.html:437-444`). The current export path says
the opposite in executable form.

## The three questions

### 1. Do `_fetch_deprecation_stubs` and `_build_stub_entry` publish a successor?

**Yes.** `_fetch_deprecation_stubs` selects catalog-approved superseded
predecessors and resolves a live successor
(`imas_codex/standard_names/export.py:625-648`). It annotates each returned
predecessor with `_successor` and drops rows for which it cannot resolve one
(`imas_codex/standard_names/export.py:669-679`). `_build_stub_entry` then emits
all of the following into the catalog entry:

| Published field | Emitted value |
|---|---|
| `status` | `deprecated` |
| `superseded_by` | the resolved successor identity |
| `description` | `Deprecated: renamed to <successor>.` |
| `documentation` | prose directing consumers to the successor |
| `links` | `name:<successor>` |

Those mappings are explicit at
`imas_codex/standard_names/export.py:683-705`. `run_export` validates the stub
and appends it to a domain catalog at
`imas_codex/standard_names/export.py:1606-1638`. The focused tests also pin the
public behavior: the entry-level assertions require both `superseded_by` and
the internal successor link
(`tests/standard_names/test_export_deprecation.py:136-151`), while the export
test requires the YAML stub to contain `superseded_by`
(`tests/standard_names/test_export_deprecation.py:236-246`).

### 2. How many stubs would a release emit today, and how many carry a successor?

The live read-only measurement at source commit
`27f28bc773f1012e8e3e228f36ba1e05870311d3` reused `_fetch_candidates`, the
default score gate (`min_score=0.65`, `include_unreviewed=False`), and
`_fetch_deprecation_stubs` rather than approximating the exporter with a
separate query.

| Measure | Live count |
|---|---:|
| `StandardName` nodes before measurement | 4,395 |
| Default release candidates before score gate | 537 |
| Default release candidates after score gate | 537 |
| Superseded predecessors with `catalog_approved_at` | 0 |
| Deprecation stubs the release helper returns | **0** |
| Returned stubs carrying `_successor` | **0** |
| `StandardName` nodes after measurement | 4,395 |

The before/after assertion passed: **4,395 = 4,395, delta 0**. The production
graph was read only. No model was called and LLM spend was **USD 0**.

The reason for the zero is the durable-approval predicate
`old.catalog_approved_at IS NOT NULL`
(`imas_codex/standard_names/export.py:652-656`). Once a published predecessor
exists, zero will cease to protect the catalog: `_build_stub_entry` has no
successor-free branch, so every actual emitted stub carries the successor in
multiple fields.

### 3. Does successor resolution ignore the scalar written by the repair operator?

**Yes.** The Cypher obtains successors exclusively from incoming variable-length
`REFINED_FROM` paths and collects `succ.id`
(`imas_codex/standard_names/export.py:652-664`). Python filters and sorts only
that collected `successors` list
(`imas_codex/standard_names/export.py:669-679`). Although `old {.*}` happens to
return the predecessor's scalar properties, neither stage reads
`old.superseded_by`; there is no scalar fallback.

The live lineage census makes the omission concrete:

| Superseded lineage class | Live count |
|---|---:|
| Superseded names total | 1,643 |
| Any reachable `REFINED_FROM` successor | 1,131 |
| `superseded_by` scalar set | 17 |
| Both edge lineage and scalar | 14 |
| Edge lineage only | 1,117 |
| Scalar only, with no `REFINED_FROM` path | **3** |
| Neither edge nor scalar | 509 |

The three scalar-only before/after identities are:

| Superseded predecessor | Internal scalar successor |
|---|---|
| `area_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` |
| `flux_due_to_thermal_fusion` | `total_neutron_source_rate_due_to_thermal_fusion` |
| `lower_energy` | `lower_bound_energy_of_neutron_detector` |

All three currently have null `catalog_approved_at`, so they do not alter
today's 0-stub result. They nevertheless prove that adding the operator's
scalar did not complete the export resolver: if these predecessors later
satisfied the approval gate, the current resolver would still skip them for
lack of a `REFINED_FROM` path.

## Minimal reconciliation

Do **not** add a `superseded_by` fallback to `_fetch_deprecation_stubs`; that
would make the internal lineage leak more complete and therefore deepen the
contradiction.

The minimal behavioral change is to remove the deprecation-stub assembly from
`run_export` (`imas_codex/standard_names/export.py:1606-1638`), so superseded
internal records never become catalog entries. To enforce the decision at the
only other generic serialization seam, stop `_graph_node_to_entry_dict` from
copying `deprecates` and `superseded_by` into active catalog entries
(`imas_codex/standard_names/export.py:607-611`). The same focused change should
retire the now-dead stub helpers, report counter, and tests that require public
stub lineage. It requires no graph migration: retain `REFINED_FROM` and
`superseded_by` internally for provenance and operator audit, but never project
either into released catalog output.

This audit recommends that change only; it does not apply it.
