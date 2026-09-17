# The node-category skip rule: what it refuses, before and after

**Plan section:** sn-west-cohort-treatment §4 (Group C: the exclusion rule over-reaches)

**Node:** n-the-skip-rule-stops-excluding-a-named-quantity

**Readout:** live graph, login node (the graph is reached through a login-node-local
endpoint and cannot be served from a SLURM partition), 2026-09-17, read-only. Each query
is bounded to a named path list and returned in well under 10 s.

**Rule under audit:** the source-eligibility gate that reads the data-dictionary
classification. `SN_SOURCE_CATEGORIES` is defined at
`imas_codex/core/node_categories.py:30` and applied from one authority in two places: the
candidate query `n.node_category IN $sn_categories`
(`imas_codex/standard_names/sources/dd.py:409,421,435`) and `qualify_dd`
(`imas_codex/standard_names/sources/dd_qualifier.py:73-78`). The second is the only gate
the explicit-path branch meets.

## Summary

`camera_x_rays/detector_humidity` is a documented physical quantity carrying a unit, so
the section requires the rule not to refuse it. The four fit-constraint weights and the two
rows the brief calls solver structure are not quantities, so the rule must keep refusing
them.

**The first half holds. The second half holds for five of the six rows the brief names.
The sixth is a factual error in the brief, and the failing assertion is what proves it.**

The correction the section specified was made by reclassifying the backing nodes under the
one authority, and that reclassification had already landed before this node ran. This
node verifies it rather than re-implementing it, which is what confirm-first requires.

## 1. The classification of each named row

The label is `IMASNode`. This is stated because an earlier probe written against a label
that does not exist returned zero rows for every count, which reads as an empty world
rather than as a typo.

```cypher
MATCH (n:IMASNode)
WHERE any(p IN $paths WHERE n.id = p)
RETURN n.id AS path, n.node_category AS node_category,
       n.data_type AS data_type, n.unit AS unit,
       n.documentation AS documentation
```

| DD path | `node_category` | `data_type` | unit | rule verdict |
|---|---|---|---|---|
| `camera_x_rays/detector_humidity` | **quantity** | STRUCTURE | 1 | admitted |
| `equilibrium/time_slice/constraints/b_field_pol_probe/weight` | fit_artifact | FLT_0D | 1 | refused |
| `equilibrium/time_slice/constraints/faraday_angle/weight` | fit_artifact | FLT_0D | 1 | refused |
| `equilibrium/time_slice/constraints/flux_loop/weight` | fit_artifact | FLT_0D | 1 | refused |
| `equilibrium/time_slice/constraints/n_e_line/weight` | fit_artifact | FLT_0D | 1 | refused |
| `equilibrium/time_slice/convergence/iterations_n` | fit_artifact | INT_0D | — | refused |
| `equilibrium/time_slice/contour_tree/node/z` | **quantity** | FLT_0D | m | admitted |

`SN_SOURCE_CATEGORIES` is `{quantity, geometry, coordinate}`. A category outside that set
is refused by the rule, so `fit_artifact` is refused by name and `quantity` is admitted.

The four weights are one concept under one classification, which is the reconciliation the
sibling evidence document already records
([unnamed roster rows](/imas-codex/evidence/sn-west-cohort-treatment/unnamed-row-causes)).
Their remaining difference is the reason string on the source rows: two report
`not_physical_quantity` / `dd_node_category_ineligible` and two report
`skipped` / `compose_model_skipped`, because the second pair reached the compose model
before the classifier repair. That is a reason-string repair, not a rule change.

## 2. The row where the brief and the measurement disagree

The brief names `equilibrium/time_slice/contour_tree/node/z` as one of the two
solver-structure rows that must **remain excluded** by the node-category rule.

Three measured facts:

1. Its `node_category` is `quantity` (unit `m`, `FLT_0D`, doc "Height"), so it is inside
   `SN_SOURCE_CATEGORIES` and the rule admits it.
2. `qualify_dd` returns eligible for it, which is the assertion that fails in §4.
3. It carries no name anyway: its source row is `skipped` with the reason
   `compose_model_skipped`. The predicate that stopped it is the compose model's free-form
   refusal — the mechanism the section's own audit identified for `detector_humidity`, in
   the opposite direction — not the classifier.

The row read behind fact 3, and behind the positive half's `composed` status, is one
bounded query over the two `source_id` values (a DD source's `source_id` is its DD path):

```cypher
MATCH (s:StandardNameSource)
WHERE s.source_id IN $paths
OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(sn:StandardName)
RETURN s.source_id AS path, s.status AS status, s.skip_reason AS skip_reason,
       sn.id AS produced_name
ORDER BY path
```

It returns two rows. `camera_x_rays/detector_humidity` reads `composed`, with no skip
reason and produced name `relative_humidity_of_detector` — the section's positive half,
asserted at the produced artifact rather than at the classifier alone.
`equilibrium/time_slice/contour_tree/node/z` reads `skipped` with
`compose_model_skipped` and no produced name.

So the row is excluded from naming while being admitted by the rule. The brief asserts its
negative half against the rule, so the assertion fails on that row alone.

**Owner of the residual:** the plan's §4 tail table, in the two-row band that lists this
row beside `convergence/iterations_n`, describes both rows as "solver structure and a
solver diagnostic" and prescribes "Classification repair, as above". That prescription
does not fit this row: the node's classification is already `quantity`, so there is no
mis-classification to repair, and the row's exclusion is a classifier gap, not a
classification error. The classification repair of the four weights, the two reconstructed
containers and both iteration counters is the part that landed. A classifier rule that
excludes a critical-point coordinate remains to be written. It is not this node's to land,
and this node changed no rule.

## 3. The count of sources the rule refuses, before and after

The rule has its own reason code, and that code is the instrument.

```cypher
MATCH (s:StandardNameSource)
WHERE s.skip_reason IS NOT NULL
RETURN s.skip_reason AS reason, count(*) AS n
ORDER BY n DESC
```

Measured 2026-09-17, ranked: 559 temporal_coordinate, 432
local_coordinate_frame, 192 dd_unit_unresolvable, 85 configurable_meaning, 22
compose_model_skipped, 12 vocab_gap_nonactionable, **10 dd_node_category_ineligible**,
9 vocab_gap, 4 non_nameable_coordinate:time.

**The rule refuses 10 sources by its own reason code, and none of them is
`camera_x_rays/detector_humidity`** — the source the section says it wrongly refused is
named and its row is `composed`, so the defect does not reproduce at this instrument.

The qualifier returns `node_category_ineligible` (`dd_qualifier.py:76`); the source rows
persist the same refusal with the source-kind prefix as `dd_node_category_ineligible`
(`signed_manifest.py:6195`). The census reads the persisted form and the two spellings
name one predicate.

**The figure is the same before and after this node's work because no rule text changed,
so both readings come from the one instrument at the one moment.** A true "before" reading
would have to be taken; nothing in this node's change can move the number, which is the
point the confirm-first finding makes. The pre-repair figure would need a snapshot of a
classification that no longer exists in the graph — the "before" column the brief asks for
cannot be produced by this node, and §5 says why rather than bridging it.

Over the DD node population the rule reads: 61,366 `IMASNode` nodes; 0 with a null
`node_category`; 51,132 refused by the rule (`NOT node_category IN {quantity, geometry,
coordinate}`); and 21,101 of those carrying both a non-dimensionless unit and
documentation. The 21,101 is the size of the population a broader audit would sweep, not a
count of this section's rows, and the two instruments count different things: the reason
code counts sources, the population counts DD nodes.

## 4. The executed assertion

Positive half passes, negative half fails on exactly one row. Full log:
`/tmp/skiprule_test.log`; harness: `/tmp/test_skip_rule_scope.py`.

```
collected 2 items
../../../../../../../../tmp/test_skip_rule_scope.py .F                   [100%]
E   AssertionError: the rule admits rows it must refuse: equilibrium/time_slice/contour_tree/node/z category='quantity' eligible=True
E   assert not ["equilibrium/time_slice/contour_tree/node/z category='quantity' eligible=True"]
FAILED ../../../../../../../../tmp/test_skip_rule_scope.py::test_negative_half_remains_excluded
========================= 1 failed, 1 passed in 27.41s ========================
```

The failure is the evidence in both directions: the collected refused-set names only the
sixth row, so the rule still refuses the five it must; and the sixth is admitted.

## 5. Limits

- The brief's "before" figure for the rule cannot be produced by this node: the state it
  describes is the pre-repair classification, which no longer exists in the graph and has
  no snapshot. The only record is the plan's frozen 2026-09-07 census, and comparing it
  would compare two different rules. Stated rather than bridged.
- `contour_tree/node/r` was not measured; the brief named `node/z`.
- No graph mutation, no catalog write and no release action was performed. Every statement
  was a read.